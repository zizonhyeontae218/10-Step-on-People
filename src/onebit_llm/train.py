from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import ExperimentConfig, TrainConfig, load_config
from .data import build_dataloaders
from .model import LanguageModel
from .preflight import run_preflight
from .quantization import ternary_statistics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learning_rate_at_step(step: int, config: TrainConfig, total_steps: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def _autocast(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int,
    use_amp: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            _, loss = model(inputs, targets)
        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    if not losses:
        raise RuntimeError("validation produced no finite losses")
    return sum(losses) / len(losses)


def _cycle(loader: Iterable[tuple[torch.Tensor, torch.Tensor]]):
    while True:
        yield from loader


def train_one(
    model_type: str,
    experiment: ExperimentConfig,
    target_steps: int | None = None,
    max_minutes: float | None = None,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    """Train one model, always saving final and best checkpoints."""
    seed_everything(experiment.train.seed)
    device = device or select_device()
    train_loader, val_loader = build_dataloaders(experiment.data, experiment.train)
    model = LanguageModel(experiment.model, model_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.train.learning_rate,
        betas=(experiment.train.beta1, experiment.train.beta2),
        weight_decay=experiment.train.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = target_steps or experiment.train.max_steps
    time_limit = 60.0 * (max_minutes or experiment.train.max_minutes_per_model)

    output_root = Path(experiment.train.output_dir)
    checkpoint_dir = output_root / "checkpoints" / model_type
    metric_path = output_root / "metrics" / f"{model_type}.jsonl"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text("", encoding="utf-8")

    batches = _cycle(train_loader)
    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()
    interval_start = start_time
    interval_tokens = 0
    best_validation = math.inf
    completed_steps = 0
    latest_validation = math.inf

    initial_validation = evaluate(
        model, val_loader, device, experiment.train.eval_batches, use_amp
    )
    initial_stats = ternary_statistics(model)
    initial_record: dict[str, float | int | str | None] = {
        "model_family": experiment.project.name,
        "model_type": model_type,
        "step": 0,
        "train_loss": None,
        "validation_loss": initial_validation,
        "perplexity": math.exp(min(initial_validation, 20.0)),
        "learning_rate": 0.0,
        "elapsed_seconds": time.perf_counter() - start_time,
        "parameter_count": model.parameter_count,
        **initial_stats,
    }
    with metric_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(initial_record, ensure_ascii=False) + "\n")
    best_validation = initial_validation
    latest_validation = initial_validation
    save_checkpoint(
        checkpoint_dir / "best.pt",
        model,
        0,
        initial_validation,
        model_family=experiment.project.name,
    )
    print(
        f"[{model_type}] step=0 validation_loss={initial_validation:.4f} "
        f"perplexity={initial_record['perplexity']:.2f}"
    )

    for step_index in range(total_steps):
        learning_rate = learning_rate_at_step(step_index, experiment.train, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        accumulated_loss = 0.0
        for _ in range(experiment.train.gradient_accumulation):
            inputs, targets = next(batches)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast(device, use_amp):
                _, loss = model(inputs, targets)
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at step {step_index + 1}")
                scaled_loss = loss / experiment.train.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.item())
            interval_tokens += int((targets != experiment.data.pad_token_id).sum().item())

        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), experiment.train.grad_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step_index + 1}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_steps = step_index + 1

        should_log = completed_steps % experiment.train.log_interval == 0 or completed_steps == 1
        should_evaluate = completed_steps % experiment.train.eval_interval == 0 or completed_steps == total_steps
        elapsed = time.perf_counter() - start_time
        timed_out = elapsed >= time_limit
        if timed_out:
            should_evaluate = True

        if should_log:
            interval_elapsed = max(time.perf_counter() - interval_start, 1e-9)
            print(
                f"[{model_type}] step={completed_steps}/{total_steps} "
                f"loss={accumulated_loss / experiment.train.gradient_accumulation:.4f} "
                f"lr={learning_rate:.2e} tokens/s={interval_tokens / interval_elapsed:.0f}"
            )
            interval_start = time.perf_counter()
            interval_tokens = 0

        if should_evaluate:
            latest_validation = evaluate(
                model, val_loader, device, experiment.train.eval_batches, use_amp
            )
            stats = ternary_statistics(model)
            record: dict[str, float | int | str] = {
                "model_family": experiment.project.name,
                "model_type": model_type,
                "step": completed_steps,
                "train_loss": accumulated_loss / experiment.train.gradient_accumulation,
                "validation_loss": latest_validation,
                "perplexity": math.exp(min(latest_validation, 20.0)),
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - start_time,
                "parameter_count": model.parameter_count,
                **stats,
            }
            with metric_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                completed_steps,
                latest_validation,
                optimizer,
                model_family=experiment.project.name,
            )
            if latest_validation < best_validation:
                best_validation = latest_validation
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    completed_steps,
                    latest_validation,
                    model_family=experiment.project.name,
                )
            print(
                f"[{model_type}] validation_loss={latest_validation:.4f} "
                f"perplexity={record['perplexity']:.2f} zero={record['zero_fraction']:.3f}"
            )

        if timed_out:
            print(f"[{model_type}] stopped at the {max_minutes or experiment.train.max_minutes_per_model:.1f}-minute limit")
            break

    if completed_steps == 0:
        raise RuntimeError("training ended before completing one optimizer step")
    return {
        "model_type": model_type,
        "model_family": experiment.project.name,
        "completed_steps": completed_steps,
        "initial_validation_loss": initial_validation,
        "best_validation_loss": best_validation,
        "last_validation_loss": latest_validation,
        "elapsed_seconds": time.perf_counter() - start_time,
        "parameter_count": model.parameter_count,
    }


def run_training(model_choice: str, experiment: ExperimentConfig) -> list[dict[str, float | int | str]]:
    device = select_device()
    print(f"device={device}")
    experiment, run_profile = run_preflight(experiment, device)
    output_root = Path(experiment.train.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_profile.json").write_text(
        json.dumps(run_profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_profile, ensure_ascii=False, indent=2))
    if model_choice == "all":
        bitnet_result = train_one("bitnet", experiment, device=device)
        fair_steps = int(bitnet_result["completed_steps"])
        baseline_result = train_one(
            "baseline",
            experiment,
            target_steps=fair_steps,
            max_minutes=experiment.train.max_minutes_per_model,
            device=device,
        )
        if int(baseline_result["completed_steps"]) != fair_steps:
            raise RuntimeError("baseline did not reach the BitNet step count within its time budget")
        results = [bitnet_result, baseline_result]
    else:
        results = [train_one(model_choice, experiment, device=device)]

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_family": experiment.project.name,
        "run_profile": run_profile,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Korean baseline and BitNet models")
    parser.add_argument("--model", choices=("baseline", "bitnet", "all"), default="all")
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args(argv)
    results = run_training(args.model, load_config(args.config))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
