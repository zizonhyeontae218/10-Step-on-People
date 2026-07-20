from __future__ import annotations

import gc
from dataclasses import asdict, dataclass, replace
from typing import Callable

import torch

from .config import ExperimentConfig
from .model import LanguageModel


@dataclass(frozen=True)
class BatchSelection:
    micro_batch_size: int
    gradient_accumulation: int
    effective_batch_size: int
    peak_allocated_mb: float
    attempts: list[dict[str, object]]


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def select_batch_configuration(
    requested_micro_batch: int,
    requested_accumulation: int,
    probe: Callable[[int], float],
) -> BatchSelection:
    """Choose the largest fitting micro-batch while preserving effective batch size."""
    effective_batch = requested_micro_batch * requested_accumulation
    candidates = [size for size in (16, 8, 4, 2, 1) if size <= requested_micro_batch]
    if requested_micro_batch not in candidates:
        candidates.insert(0, requested_micro_batch)
    attempts: list[dict[str, object]] = []
    for candidate in candidates:
        if effective_batch % candidate:
            continue
        try:
            peak_mb = float(probe(candidate))
            attempts.append({"micro_batch_size": candidate, "status": "ok", "peak_allocated_mb": peak_mb})
            return BatchSelection(
                micro_batch_size=candidate,
                gradient_accumulation=effective_batch // candidate,
                effective_batch_size=effective_batch,
                peak_allocated_mb=peak_mb,
                attempts=attempts,
            )
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            attempts.append({"micro_batch_size": candidate, "status": "oom"})
    raise RuntimeError(
        "CUDA preflight failed even at micro-batch 1. Close GPU-heavy applications or reduce the model explicitly."
    )


def _cuda_probe(experiment: ExperimentConfig, device: torch.device, micro_batch_size: int) -> float:
    model = optimizer = scaler = inputs = targets = loss = None
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = LanguageModel(experiment.model, "bitnet").to(device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=experiment.train.learning_rate,
            betas=(experiment.train.beta1, experiment.train.beta2),
            weight_decay=experiment.train.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        inputs = torch.randint(
            1,
            experiment.model.vocab_size,
            (micro_batch_size, experiment.model.context_length),
            device=device,
        )
        targets = torch.randint(
            1,
            experiment.model.vocab_size,
            (micro_batch_size, experiment.model.context_length),
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("preflight model did not produce a loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        optimizer.step()
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    finally:
        del model, optimizer, scaler, inputs, targets, loss
        gc.collect()
        torch.cuda.empty_cache()


def run_preflight(
    experiment: ExperimentConfig,
    device: torch.device,
    probe: Callable[[int], float] | None = None,
) -> tuple[ExperimentConfig, dict[str, object]]:
    """Run CUDA memory calibration and return a runtime-adjusted configuration."""
    configured_effective_batch = (
        experiment.train.micro_batch_size * experiment.train.gradient_accumulation
    )
    if device.type != "cuda":
        profile = {
            "model_family": experiment.project.name,
            "device": str(device),
            "cuda_available": False,
            "preflight_status": "skipped",
            "micro_batch_size": experiment.train.micro_batch_size,
            "gradient_accumulation": experiment.train.gradient_accumulation,
            "effective_batch_size": configured_effective_batch,
        }
        return experiment, profile

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    probe_function = probe or (lambda batch: _cuda_probe(experiment, device, batch))
    selection = select_batch_configuration(
        experiment.train.micro_batch_size,
        experiment.train.gradient_accumulation,
        probe_function,
    )
    adjusted_train = replace(
        experiment.train,
        micro_batch_size=selection.micro_batch_size,
        gradient_accumulation=selection.gradient_accumulation,
    )
    adjusted = replace(experiment, train=adjusted_train)
    properties = torch.cuda.get_device_properties(device)
    profile = {
        "model_family": experiment.project.name,
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": True,
        "preflight_status": "passed",
        "free_vram_before_mb": free_bytes / (1024**2),
        "total_vram_mb": total_bytes / (1024**2),
        **asdict(selection),
    }
    return adjusted, profile
