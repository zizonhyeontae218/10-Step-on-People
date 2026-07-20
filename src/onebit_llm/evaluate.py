from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import load_config
from .generation import generate_text, load_generator
from .quantization import ternary_statistics

PROMPTS = ("오늘 학교에서", "인공지능은", "대한민국의", "학생들은")
GENERATION_SETTINGS = {
    "max_new_tokens": 32,
    "temperature": 0.8,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "seed": 42,
}


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_report(results: dict[str, Any]) -> str:
    """Render a compact Korean Markdown report from evaluation results."""
    lines = [
        f"# {results['model_family']} 품질 확인",
        "",
        "> 이름의 `10-step`은 브랜드이며 실제 학습 step 수를 뜻하지 않습니다.",
        "",
        "## 정량 비교",
        "",
        "| 모델 | step | 초기 loss | best loss | 개선 | perplexity | ternary 0 비율 | checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in results["models"]:
        improvement = model["validation_improvement"]
        lines.append(
            f"| {model['variant']} | {model['step']} | {_fmt(model['initial_validation_loss'])} | "
            f"{_fmt(model['best_validation_loss'])} | {_fmt(improvement)} | "
            f"{_fmt(model['perplexity'], 2)} | {_fmt(model['ternary_zero_fraction'], 3)} | "
            f"{model['checkpoint_size_mb']:.1f} MB |"
        )
    lines.extend(["", "## 학습 판정", ""])
    for model in results["models"]:
        if model["improved_from_step_zero"] is True:
            verdict = "step 0보다 validation loss가 개선됐습니다."
        elif model["improved_from_step_zero"] is False:
            verdict = "⚠️ step 0보다 validation loss가 개선되지 않았습니다."
        else:
            verdict = "기존 metric에 step 0 기록이 없어 개선 여부를 계산할 수 없습니다."
        lines.append(f"- **{model['variant']}**: {verdict}")

    lines.extend(["", "## 고정 prompt 생성", ""])
    for prompt in results["prompts"]:
        lines.append(f"### `{prompt}`")
        lines.append("")
        for model in results["models"]:
            sample = next(item for item in model["generations"] if item["prompt"] == prompt)
            lines.extend(
                [
                    f"**{model['variant']}** · {sample['elapsed_seconds']:.2f}초",
                    "",
                    "```text",
                    sample["text"],
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## 해석 주의",
            "",
            "일반 PyTorch fake quantization 결과이며 packed 1.58-bit 저장이나 전용 kernel 속도를 나타내지 않습니다.",
            "짧은 제한 학습이므로 생성문의 사실성·일관성·안전성을 보장하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_checkpoints(
    config_path: str | Path = "configs/demo.yaml",
    checkpoint_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    experiment = load_config(config_path)
    artifact_root = Path(experiment.train.output_dir)
    checkpoint_root = Path(checkpoint_root) if checkpoint_root else artifact_root / "checkpoints"
    evaluation_dir = Path(output_dir) if output_dir else artifact_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[str, Any] = {
        "model_family": experiment.project.name,
        "device": str(device),
        "prompts": list(PROMPTS),
        "generation_settings": GENERATION_SETTINGS,
        "models": [],
    }
    for variant in ("bitnet", "baseline"):
        checkpoint_path = checkpoint_root / variant / "best.pt"
        model, tokenizer, metadata = load_generator(
            variant,
            experiment.data.tokenizer_path,
            checkpoint_root,
            device=device,
        )
        metrics = _read_metrics(artifact_root / "metrics" / f"{variant}.jsonl")
        initial_record = next((record for record in metrics if int(record.get("step", -1)) == 0), None)
        initial_loss = float(initial_record["validation_loss"]) if initial_record else None
        best_loss = float(metadata["validation_loss"])
        improvement = initial_loss - best_loss if initial_loss is not None else None
        stats = ternary_statistics(model)
        generations = []
        for prompt in PROMPTS:
            generated = generate_text(model, tokenizer, prompt, **GENERATION_SETTINGS)
            generations.append(
                {
                    "prompt": prompt,
                    "text": generated.text,
                    "continuation": generated.continuation,
                    "new_token_count": generated.new_token_count,
                    "elapsed_seconds": generated.elapsed_seconds,
                }
            )
        results["models"].append(
            {
                "model_family": metadata.get("model_family", experiment.project.name),
                "variant": metadata.get("variant", variant),
                "step": int(metadata["step"]),
                "parameter_count": int(metadata["parameter_count"]),
                "initial_validation_loss": initial_loss,
                "best_validation_loss": best_loss,
                "validation_improvement": improvement,
                "improved_from_step_zero": (improvement > 0) if improvement is not None else None,
                "perplexity": math.exp(min(best_loss, 20.0)),
                "ternary_zero_fraction": float(stats["zero_fraction"]),
                "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
                "generations": generations,
            }
        )
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_path = evaluation_dir / "results.json"
    report_path = evaluation_dir / "report.md"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(results), encoding="utf-8")
    print(f"wrote {results_path}")
    print(f"wrote {report_path}")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate 10-step-on-people checkpoints")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    evaluate_checkpoints(args.config, args.checkpoint_root, args.output_dir)


if __name__ == "__main__":
    main()
