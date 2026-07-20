from __future__ import annotations

import pytest
import torch

from onebit_llm.preflight import select_batch_configuration


def test_preflight_falls_back_and_preserves_effective_batch() -> None:
    attempted: list[int] = []

    def probe(batch_size: int) -> float:
        attempted.append(batch_size)
        if batch_size > 2:
            raise torch.OutOfMemoryError("synthetic CUDA out of memory")
        return 777.0

    selected = select_batch_configuration(8, 4, probe)
    assert attempted == [8, 4, 2]
    assert selected.micro_batch_size == 2
    assert selected.gradient_accumulation == 16
    assert selected.effective_batch_size == 32
    assert selected.peak_allocated_mb == 777.0


def test_preflight_does_not_hide_non_oom_errors() -> None:
    with pytest.raises(RuntimeError, match="kernel failed"):
        select_batch_configuration(8, 4, lambda _: (_ for _ in ()).throw(RuntimeError("kernel failed")))


def test_preflight_fails_if_batch_one_cannot_fit() -> None:
    def always_oom(_: int) -> float:
        raise torch.OutOfMemoryError("synthetic CUDA out of memory")

    with pytest.raises(RuntimeError, match="micro-batch 1"):
        select_batch_configuration(8, 4, always_oom)
