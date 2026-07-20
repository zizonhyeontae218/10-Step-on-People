from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .checkpoint import load_model_checkpoint
from .model import LanguageModel


@dataclass(frozen=True)
class GenerationResult:
    text: str
    continuation: str
    new_token_count: int
    elapsed_seconds: float
    model_type: str


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> None:
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if penalty == 1.0:
        return
    for token_id in token_ids.unique():
        index = int(token_id.item())
        logits[index] = logits[index] * penalty if logits[index] < 0 else logits[index] / penalty


@torch.inference_mode()
def generate_token_ids(
    model: LanguageModel,
    prompt_ids: list[int],
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    seed: int = 42,
    eos_token_id: int = 2,
) -> list[int]:
    """Sample continuation token IDs from a causal language model."""
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if not 1 <= max_new_tokens <= 64:
        raise ValueError("max_new_tokens must be between 1 and 64")
    if not 0.1 <= temperature <= 1.5:
        raise ValueError("temperature must be between 0.1 and 1.5")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    device = next(model.parameters()).device
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    random_generator = torch.Generator(device=device).manual_seed(seed)
    new_ids: list[int] = []
    model.eval()

    for _ in range(max_new_tokens):
        model_input = generated[:, -model.config.context_length :]
        logits, _ = model(model_input)
        next_logits = logits[0, -1].float() / temperature
        _apply_repetition_penalty(next_logits, generated[0], repetition_penalty)
        selected_values, selected_indices = torch.topk(next_logits, k=min(top_k, next_logits.numel()))
        probabilities = torch.softmax(selected_values, dim=-1)
        selected_position = torch.multinomial(probabilities, 1, generator=random_generator)
        next_id = int(selected_indices[selected_position].item())
        new_ids.append(next_id)
        generated = torch.cat(
            (generated, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1
        )
        if next_id == eos_token_id:
            break
    return new_ids


def load_tokenizer(path: str | Path) -> Tokenizer:
    tokenizer_path = Path(path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer not found: {tokenizer_path}. Run `python -m onebit_llm.prepare` first."
        )
    return Tokenizer.from_file(str(tokenizer_path))


def generate_text(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> GenerationResult:
    if not prompt.strip():
        raise ValueError("문장 앞부분을 입력해 주세요.")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise ValueError("입력에서 token을 만들 수 없습니다.")
    prompt_ids = prompt_ids[-model.config.context_length :]
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    new_ids = generate_token_ids(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
    return GenerationResult(
        text=prompt + continuation,
        continuation=continuation,
        new_token_count=len(new_ids),
        elapsed_seconds=elapsed,
        model_type=model.model_type,
    )


def load_generator(
    model_type: str,
    tokenizer_path: str | Path = "data/tokenizer/tokenizer.json",
    checkpoint_root: str | Path = "artifacts/checkpoints",
    device: torch.device | None = None,
) -> tuple[LanguageModel, Tokenizer, dict[str, object]]:
    if model_type not in {"baseline", "bitnet"}:
        raise ValueError("model_type must be baseline or bitnet")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_root) / model_type / "best.pt"
    model, metadata = load_model_checkpoint(checkpoint_path, device)
    return model, load_tokenizer(tokenizer_path), metadata
