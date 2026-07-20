from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .quantization import BitLinear


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dimension: int, max_sequence_length: int) -> None:
        super().__init__()
        if head_dimension % 2:
            raise ValueError("head dimension must be even for RoPE")
        inverse_frequency = 1.0 / (
            10_000 ** (torch.arange(0, head_dimension, 2, dtype=torch.float32) / head_dimension)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(-2)
        cos = self.cos[:sequence_length].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        sin = self.sin[:sequence_length].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def linear_factory(model_type: str):
    if model_type == "baseline":
        return lambda input_size, output_size: nn.Linear(input_size, output_size, bias=False)
    if model_type == "bitnet":
        return BitLinear
    raise ValueError(f"unknown model type: {model_type}")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, model_type: str) -> None:
        super().__init__()
        make_linear = linear_factory(model_type)
        self.num_heads = config.num_heads
        self.head_dimension = config.hidden_size // config.num_heads
        self.qkv = make_linear(config.hidden_size, config.hidden_size * 3)
        self.output = make_linear(config.hidden_size, config.hidden_size)
        self.rope = RotaryEmbedding(self.head_dimension, config.context_length)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, sequence, self.num_heads, self.head_dimension).transpose(1, 2)

        query = self.rope(split_heads(query))
        key = self.rope(split_heads(key))
        value = split_heads(value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, hidden)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, model_type: str) -> None:
        super().__init__()
        make_linear = linear_factory(model_type)
        self.up = make_linear(config.hidden_size, config.ffn_size)
        self.down = make_linear(config.ffn_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.relu(self.up(x)).square()
        return self.down(activated)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, model_type: str) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config, model_type)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.feed_forward = FeedForward(config, model_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class LanguageModel(nn.Module):
    """Decoder-only causal language model supporting baseline and BitNet linears."""

    def __init__(self, config: ModelConfig, model_type: str = "bitnet") -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.model_type = model_type
        make_linear = linear_factory(model_type)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config, model_type) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = make_linear(config.hidden_size, config.vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.config.context_length:
            raise ValueError("sequence exceeds configured context length")
        hidden = self.dropout(self.embedding(token_ids))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0
            )
        return logits, loss

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_config": asdict(self.config),
            "parameter_count": self.parameter_count,
        }
