from __future__ import annotations

from pathlib import Path

import pytest
import torch

from onebit_llm.checkpoint import load_model_checkpoint
from onebit_llm.config import DEFAULT_MODEL_FAMILY, ModelConfig
from onebit_llm.generation import generate_token_ids
from onebit_llm.model import LanguageModel


def make_model() -> LanguageModel:
    return LanguageModel(
        ModelConfig(
            vocab_size=16,
            context_length=4,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_size=16,
            dropout=0.0,
        ),
        "baseline",
    ).eval()


def test_generation_is_seeded_and_honors_max_length() -> None:
    torch.manual_seed(1)
    model = make_model()
    first = generate_token_ids(model, [1, 3, 4, 5, 6, 7], max_new_tokens=5, seed=9, eos_token_id=-1)
    second = generate_token_ids(model, [1, 3, 4, 5, 6, 7], max_new_tokens=5, seed=9, eos_token_id=-1)
    assert first == second
    assert len(first) == 5


def test_generation_rejects_empty_or_invalid_controls() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="prompt"):
        generate_token_ids(model, [])
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_token_ids(model, [1], max_new_tokens=65)


def test_missing_checkpoint_has_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="onebit_llm.train"):
        load_model_checkpoint(tmp_path / "missing.pt", torch.device("cpu"))


def test_legacy_checkpoint_gets_family_and_variant_defaults(tmp_path: Path) -> None:
    model = make_model()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            **model.checkpoint_metadata(),
            "step": 10,
            "validation_loss": 1.0,
            "model_state": model.state_dict(),
        },
        path,
    )
    _, metadata = load_model_checkpoint(path, torch.device("cpu"))
    assert metadata["model_family"] == DEFAULT_MODEL_FAMILY
    assert metadata["variant"] == "baseline"
