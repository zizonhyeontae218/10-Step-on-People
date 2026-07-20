from __future__ import annotations

import torch

from onebit_llm.quantization import BitLinear, int8_activation_codes, ternary_weight_codes


def test_ternary_codes_have_only_three_values() -> None:
    weight = torch.tensor([[-2.0, -0.1, 0.0, 0.1, 2.0]])
    codes, scale = ternary_weight_codes(weight)
    assert set(codes.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert scale.item() > 0


def test_activation_codes_are_signed_int8_per_token() -> None:
    activation = torch.tensor([[[0.0, 2.0, -4.0], [1.0, -1.0, 0.5]]])
    codes, scale = int8_activation_codes(activation)
    assert codes.min() >= -127
    assert codes.max() <= 127
    assert scale.shape == (1, 2, 1)


def test_bitlinear_passes_finite_ste_gradients() -> None:
    layer = BitLinear(8, 4)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    layer(inputs).square().mean().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
