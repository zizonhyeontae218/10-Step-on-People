from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_MODEL_FAMILY
from .generation import generate_text, load_generator


class ModelRegistry:
    """Lazily load and cache the two presentation checkpoints."""

    def __init__(
        self,
        tokenizer_path: str | Path = "data/tokenizer/tokenizer.json",
        checkpoint_root: str | Path = "artifacts/checkpoints",
        model_family: str = DEFAULT_MODEL_FAMILY,
    ) -> None:
        self.tokenizer_path = Path(tokenizer_path)
        self.checkpoint_root = Path(checkpoint_root)
        self.model_family = model_family
        self._cache: dict[str, tuple[object, object, dict[str, object]]] = {}

    def get(self, model_type: str):
        if model_type not in self._cache:
            self._cache[model_type] = load_generator(
                model_type, self.tokenizer_path, self.checkpoint_root
            )
        return self._cache[model_type]

    def continue_text(
        self,
        prompt: str,
        model_type: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[str, str]:
        model, tokenizer, metadata = self.get(model_type)
        result = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            repetition_penalty=1.1,
            seed=42,
        )
        details = (
            f"**{metadata.get('model_family', self.model_family)} · {model_type}** · "
            f"checkpoint step {metadata['step']} · "
            f"생성 {result.new_token_count} tokens · {result.elapsed_seconds:.2f}초"
        )
        return result.text, details


def create_app(registry: ModelRegistry | None = None):
    """Build the non-chat Gradio continuation interface."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is missing; install with `pip install -e \".[app]\"`") from error

    registry = registry or ModelRegistry()
    with gr.Blocks(title=f"{registry.model_family} · 한국어 문장 이어쓰기") as demo:
        gr.Markdown(
            f"# {registry.model_family}\n"
            "### 한국어 1.58-bit 문장 이어쓰기\n"
            "문장 앞부분을 입력하면 작은 언어 모델이 다음 내용을 생성합니다. 채팅이나 지식 질답 모델은 아닙니다."
        )
        prompt = gr.Textbox(
            label="문장 앞부분",
            placeholder="오늘 학교에서",
            value="오늘 학교에서",
            lines=2,
            autofocus=True,
        )
        with gr.Row():
            model_type = gr.Radio(
                choices=[
                    (f"{registry.model_family} · BitNet b1.58", "bitnet"),
                    (f"{registry.model_family} · FP baseline", "baseline"),
                ],
                value="bitnet",
                label="모델",
            )
            max_new_tokens = gr.Slider(1, 64, value=32, step=1, label="생성 token 수")
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(1, 100, value=40, step=1, label="Top-k")
        submit = gr.Button("이어쓰기", variant="primary")
        output = gr.Textbox(label="완성된 문장", lines=6, interactive=False)
        details = gr.Markdown()

        inputs = [prompt, model_type, max_new_tokens, temperature, top_k]
        submit.click(registry.continue_text, inputs=inputs, outputs=[output, details])
        prompt.submit(registry.continue_text, inputs=inputs, outputs=[output, details])
        gr.Examples(
            examples=[["오늘 학교에서"], ["인공지능은"], ["대한민국의"], ["학생들은"]],
            inputs=[prompt],
        )
    return demo
