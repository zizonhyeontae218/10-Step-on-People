from __future__ import annotations

import pytest
import torch

from onebit_llm.config import ModelConfig
from onebit_llm.model import LanguageModel
from onebit_llm.quantization import BitLinear


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        context_length=8,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        ffn_size=32,
        dropout=0.0,
    )


@pytest.mark.parametrize("model_type", ["baseline", "bitnet"])
def test_forward_shape_and_finite_loss(model_type: str) -> None:
    model = LanguageModel(tiny_config(), model_type)
    inputs = torch.randint(1, 32, (2, 8))
    targets = torch.randint(1, 32, (2, 8))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, 32)
    assert loss is not None and torch.isfinite(loss)


def test_baseline_and_bitnet_have_equal_parameter_counts() -> None:
    baseline = LanguageModel(tiny_config(), "baseline")
    bitnet = LanguageModel(tiny_config(), "bitnet")
    assert baseline.parameter_count == bitnet.parameter_count
    assert not any(isinstance(module, BitLinear) for module in baseline.modules())
    assert any(isinstance(module, BitLinear) for module in bitnet.modules())


@pytest.mark.parametrize("model_type", ["baseline", "bitnet"])
def test_causal_mask_blocks_future_tokens(model_type: str) -> None:
    torch.manual_seed(7)
    model = LanguageModel(tiny_config(), model_type).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = first.clone()
    second[:, 4:] = torch.tensor([9, 10, 11, 12])
    first_logits, _ = model(first)
    second_logits, _ = model(second)
    torch.testing.assert_close(first_logits[:, :4], second_logits[:, :4], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_type", ["baseline", "bitnet"])
def test_fixed_batch_can_overfit(model_type: str) -> None:
    torch.manual_seed(3)
    model = LanguageModel(tiny_config(), model_type)
    inputs = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]).repeat(2, 1)
    targets = torch.tensor([[2, 3, 4, 1, 2, 3, 4, 1]]).repeat(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    _, initial_loss = model(inputs, targets)
    assert initial_loss is not None
    for _ in range(30):
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, final_loss = model(inputs, targets)
    assert final_loss is not None
    assert final_loss.item() < initial_loss.item()
