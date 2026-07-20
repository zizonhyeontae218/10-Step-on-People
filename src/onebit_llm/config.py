from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_FAMILY = "10-step-on-people"


@dataclass(frozen=True)
class ProjectConfig:
    name: str = DEFAULT_MODEL_FAMILY

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("project name must not be empty")


@dataclass(frozen=True)
class DataConfig:
    tokens_path: str = "data/processed/tokens.npy"
    splits_path: str = "data/processed/splits.npz"
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    context_length: int = 128
    pad_token_id: int = 0


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32_000
    context_length: int = 128
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_size: int = 768
    dropout: float = 0.1

    def validate(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    micro_batch_size: int = 8
    gradient_accumulation: int = 4
    max_steps: int = 1_500
    max_minutes_per_model: float = 55.0
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 150
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 25
    log_interval: int = 10
    num_workers: int = 0
    output_dir: str = "artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = ExperimentConfig(
        project=ProjectConfig(**raw.get("project", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
    config.project.validate()
    config.model.validate()
    if config.data.context_length != config.model.context_length:
        raise ValueError("data and model context_length must match")
    if config.train.gradient_accumulation < 1 or config.train.micro_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    return config
