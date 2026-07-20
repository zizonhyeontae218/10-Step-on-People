from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from onebit_llm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from onebit_llm.train import train_one


@pytest.mark.parametrize("model_type", ["baseline", "bitnet"])
def test_two_step_training_writes_finite_metrics(tmp_path: Path, model_type: str) -> None:
    rng = np.random.default_rng(2)
    tokens = rng.integers(1, 32, size=(12, 10), dtype=np.uint16)
    tokens_path = tmp_path / "tokens.npy"
    splits_path = tmp_path / "splits.npz"
    np.save(tokens_path, tokens, allow_pickle=False)
    np.savez(splits_path, train=np.arange(10), validation=np.arange(10, 12))
    experiment = ExperimentConfig(
        data=DataConfig(
            tokens_path=str(tokens_path),
            splits_path=str(splits_path),
            tokenizer_path=str(tmp_path / "tokenizer.json"),
            context_length=8,
        ),
        model=ModelConfig(
            vocab_size=32,
            context_length=8,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_size=16,
            dropout=0.0,
        ),
        train=TrainConfig(
            micro_batch_size=2,
            gradient_accumulation=1,
            max_steps=2,
            max_minutes_per_model=1,
            warmup_steps=1,
            eval_interval=1,
            eval_batches=1,
            log_interval=1,
            output_dir=str(tmp_path / "artifacts"),
        ),
    )
    result = train_one(model_type, experiment, device=torch.device("cpu"))
    assert result["completed_steps"] == 2
    checkpoint_dir = tmp_path / "artifacts" / "checkpoints" / model_type
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "last.pt").exists()
    metric_lines = (tmp_path / "artifacts" / "metrics" / f"{model_type}.jsonl").read_text().splitlines()
    assert len(metric_lines) == 3
    assert json.loads(metric_lines[0])["step"] == 0
    assert json.loads(metric_lines[0])["model_family"] == "10-step-on-people"
    assert all(np.isfinite(json.loads(line)["validation_loss"]) for line in metric_lines)
