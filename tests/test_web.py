from __future__ import annotations

from onebit_llm.generation import GenerationResult
from onebit_llm.web import ModelRegistry, create_app


def test_gradio_handler_without_launching(monkeypatch) -> None:
    registry = ModelRegistry()
    registry._cache["bitnet"] = (
        object(),
        object(),
        {"step": 10, "model_family": "10-step-on-people"},
    )

    def fake_generate(*args, **kwargs) -> GenerationResult:
        return GenerationResult("오늘 학교에서 공부했다.", " 공부했다.", 3, 0.01, "bitnet")

    monkeypatch.setattr("onebit_llm.web.generate_text", fake_generate)
    text, details = registry.continue_text("오늘 학교에서", "bitnet", 32, 0.8, 40)
    assert text == "오늘 학교에서 공부했다."
    assert "step 10" in details
    assert "10-step-on-people" in details


def test_gradio_app_builds_without_launching() -> None:
    app = create_app(ModelRegistry())
    assert app is not None
