from __future__ import annotations

import argparse
from typing import Sequence

from .generation import generate_text, load_generator


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Continue a Korean text prompt")
    parser.add_argument("--model", choices=("baseline", "bitnet"), default="bitnet")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    model, tokenizer, metadata = load_generator(args.model)
    result = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    print(result.text)
    print(
        f"\n[{metadata['model_type']}, step {metadata['step']}, "
        f"{result.new_token_count} tokens, {result.elapsed_seconds:.2f}s]"
    )


if __name__ == "__main__":
    main()
