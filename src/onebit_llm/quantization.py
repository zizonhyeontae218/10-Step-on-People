from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def ternary_weight_codes(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return abs-mean ternary codes and their scalar dequantization scale."""
    scale = weight.abs().mean().clamp_min(1e-5)
    codes = (weight / scale).round().clamp(-1, 1)
    return codes, scale


def int8_activation_codes(activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each token vector to signed int8 codes and return its scale."""
    scale = activation.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 127.0
    codes = (activation / scale).round().clamp(-127, 127)
    return codes, scale


def fake_quantize_weight(weight: torch.Tensor) -> torch.Tensor:
    codes, scale = ternary_weight_codes(weight)
    dequantized = codes * scale
    return weight + (dequantized - weight).detach()


def fake_quantize_activation(activation: torch.Tensor) -> torch.Tensor:
    codes, scale = int8_activation_codes(activation)
    dequantized = codes * scale
    return activation + (dequantized - activation).detach()


class BitLinear(nn.Linear):
    """Linear layer with W1.58A8 fake quantization and STE gradients."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        quantized_activation = fake_quantize_activation(activation)
        quantized_weight = fake_quantize_weight(self.weight)
        return F.linear(quantized_activation, quantized_weight, None)


def ternary_statistics(module: nn.Module) -> dict[str, float | int]:
    """Measure ternary-code counts across every BitLinear master weight."""
    negative = zero = positive = total = 0
    for child in module.modules():
        if isinstance(child, BitLinear):
            codes, _ = ternary_weight_codes(child.weight.detach())
            negative += int((codes == -1).sum().item())
            zero += int((codes == 0).sum().item())
            positive += int((codes == 1).sum().item())
            total += codes.numel()
    return {
        "ternary_parameters": total,
        "negative": negative,
        "zero": zero,
        "positive": positive,
        "zero_fraction": (zero / total) if total else 0.0,
    }
