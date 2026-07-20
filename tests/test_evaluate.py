from __future__ import annotations

from onebit_llm.evaluate import render_report


def test_report_contains_brand_metrics_and_prompts() -> None:
    result = {
        "model_family": "10-step-on-people",
        "prompts": ["오늘 학교에서"],
        "models": [
            {
                "variant": "bitnet",
                "step": 100,
                "initial_validation_loss": 10.0,
                "best_validation_loss": 9.0,
                "validation_improvement": 1.0,
                "improved_from_step_zero": True,
                "perplexity": 8103.0,
                "ternary_zero_fraction": 0.3,
                "checkpoint_size_mb": 72.0,
                "generations": [
                    {"prompt": "오늘 학교에서", "text": "오늘 학교에서 공부했다.", "elapsed_seconds": 0.1}
                ],
            }
        ],
    }
    report = render_report(result)
    assert "# 10-step-on-people 품질 확인" in report
    assert "오늘 학교에서 공부했다." in report
    assert "validation loss가 개선" in report
