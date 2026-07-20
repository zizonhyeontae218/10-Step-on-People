from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .config import DEFAULT_MODEL_FAMILY, ModelConfig
from .model import LanguageModel


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    model: LanguageModel,
    step: int,
    validation_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> None:
    payload: dict[str, Any] = {
        **model.checkpoint_metadata(),
        "step": step,
        "validation_loss": validation_loss,
        "model_family": model_family,
        "variant": model.model_type,
        "model_state": model.state_dict(),
        "tokenizer_repo": "oz1115/korean-gpt-150m-ko",
        "tokenizer_revision": "669560d7d1de8c3213e43a343e577a80ad6cb7ee",
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    atomic_torch_save(payload, path)


def load_model_checkpoint(path: str | Path, device: torch.device) -> tuple[LanguageModel, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. "
            "Run `python -m onebit_llm.train --model all --config configs/demo.yaml` first."
        )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    payload.setdefault("model_family", DEFAULT_MODEL_FAMILY)
    payload.setdefault("variant", payload.get("model_type", "unknown"))
    model = LanguageModel(ModelConfig(**payload["model_config"]), payload["model_type"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload
