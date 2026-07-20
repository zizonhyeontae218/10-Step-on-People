from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .prepare import TOKENIZER_REPO, TOKENIZER_REVISION, make_splits

SOURCE_REPO = "wikimedia/wikipedia"
SOURCE_REVISION = "a634f78b1c435397c07001e175fa74cc4ad5e775"
SOURCE_FILE = "20231101.ko/train/0000.parquet"
ROW_TOKENS = 512
VOCAB_SIZE = 32_000
EOS_TOKEN_ID = 2

STOP_SECTION = re.compile(
    r"^(같이 보기|각주|참고 문헌|참고문헌|외부 링크|외부링크|관련 문서|출처)\s*$"
)
BAD_MARKUP = re.compile(
    r"https?://|www\.|youtu(?:be|\.be)|\[\[|\]\]|\{\{|\}\}|<ref\b|</ref>|"
    r"(?:파일|분류|틀):|\.(?:com|net|org)(?:/|\b)",
    re.IGNORECASE,
)
BLOCKED_TOPIC = re.compile(
    r"비디오\s*게임|온라인\s*게임|모바일\s*게임|게임\s*(?:등장인물|개발사|파일)|"
    r"유튜버|유튜브\s*(?:채널|영상)",
    re.IGNORECASE,
)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
SPACES = re.compile(r"[ \t]+")


def clean_document(title: str, text: str) -> str | None:
    """Return plain Korean prose, dropping link dumps, markup, and game/video pages."""
    title = unicodedata.normalize("NFC", html.unescape(title or "")).strip()
    text = unicodedata.normalize("NFC", html.unescape(text or ""))
    text = CONTROL.sub("", text).replace("\u200b", "")
    if BLOCKED_TOPIC.search(title):
        return None

    kept: list[str] = []
    for raw_line in text.splitlines():
        line = SPACES.sub(" ", raw_line).strip()
        if STOP_SECTION.fullmatch(line):
            break
        if not line or BAD_MARKUP.search(line):
            continue
        if len(line) > 24:
            letters = sum(character.isalpha() for character in line)
            hangul = sum("가" <= character <= "힣" for character in line)
            if letters and hangul / letters < 0.18:
                continue
        kept.append(line)

    cleaned = "\n".join(kept)
    hangul = sum("가" <= character <= "힣" for character in cleaned)
    letters = sum(character.isalpha() for character in cleaned)
    if len(cleaned) < 160 or hangul < 80 or (letters and hangul / letters < 0.30):
        return None
    if BLOCKED_TOPIC.search(cleaned[:800]):
        return None
    return cleaned


def token_rows(tokenizer: Tokenizer, text: str) -> Iterator[np.ndarray]:
    """Tokenize one document into independent padded rows without crossing its boundary."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        return
    ids.append(EOS_TOKEN_ID)
    for start in range(0, len(ids), ROW_TOKENS):
        chunk = ids[start : start + ROW_TOKENS]
        if len(chunk) < 129:
            continue
        if min(chunk) < 0 or max(chunk) >= VOCAB_SIZE:
            raise ValueError("tokenizer produced an out-of-range token ID")
        row = np.zeros(ROW_TOKENS, dtype=np.uint16)
        row[: len(chunk)] = chunk
        yield row


def _selected(document_id: str, modulus: int = 3) -> bool:
    digest = hashlib.blake2b(document_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus == 0


def prepare_clean(root: Path, target_rows: int = 80_000, force: bool = False) -> dict[str, object]:
    """Build a reproducible clean Korean subset from pinned Wikimedia parquet data."""
    import pyarrow.parquet as pq

    base = root / "data" / "9-step-on-people"
    raw_dir = base / "raw"
    processed_dir = base / "processed"
    tokenizer_dir = base / "tokenizer"
    for directory in (raw_dir, processed_dir, tokenizer_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tokens_path = processed_dir / "tokens.npy"
    splits_path = processed_dir / "splits.npz"
    metadata_path = processed_dir / "metadata.json"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not force and all(path.exists() for path in (tokens_path, splits_path, metadata_path, tokenizer_path)):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    parquet_path = Path(
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename=SOURCE_FILE,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            local_dir=raw_dir,
        )
    )
    downloaded_tokenizer = Path(
        hf_hub_download(
            repo_id=TOKENIZER_REPO,
            filename="tokenizer.json",
            revision=TOKENIZER_REVISION,
            local_dir=tokenizer_dir,
        )
    )
    if downloaded_tokenizer.resolve() != tokenizer_path.resolve():
        raise RuntimeError(f"tokenizer was downloaded to an unexpected path: {downloaded_tokenizer}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    temporary = tokens_path.with_suffix(".npy.tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint16, shape=(target_rows, ROW_TOKENS)
    )
    accepted_documents = rejected_documents = scanned_documents = row_count = 0
    parquet = pq.ParquetFile(parquet_path)
    try:
        for batch in parquet.iter_batches(batch_size=256, columns=["id", "title", "text"]):
            columns = batch.to_pydict()
            for document_id, title, source_text in zip(
                columns["id"], columns["title"], columns["text"], strict=True
            ):
                scanned_documents += 1
                if not _selected(str(document_id)):
                    continue
                cleaned = clean_document(str(title or ""), str(source_text or ""))
                if cleaned is None:
                    rejected_documents += 1
                    continue
                document_rows = list(token_rows(tokenizer, cleaned))
                if not document_rows:
                    rejected_documents += 1
                    continue
                accepted_documents += 1
                for row in document_rows:
                    output[row_count] = row
                    row_count += 1
                    if row_count >= target_rows:
                        break
                if row_count >= target_rows:
                    break
            if row_count >= target_rows:
                break
        output.flush()
    finally:
        del output

    if row_count != target_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"source yielded only {row_count:,} rows; expected {target_rows:,}")
    os.replace(temporary, tokens_path)

    train_indices, validation_indices = make_splits(target_rows, train_fraction=0.95, seed=42)
    split_temp = splits_path.with_suffix(".npz.tmp")
    with split_temp.open("wb") as handle:
        np.savez(handle, train=train_indices, validation=validation_indices)
    os.replace(split_temp, splits_path)

    metadata: dict[str, object] = {
        "model_family": "9-step-on-people",
        "dataset_repo": SOURCE_REPO,
        "dataset_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_license": "CC BY-SA 4.0 / GFDL",
        "cleaning": "plain Korean prose; link/markup/game-video pages removed; tail sections removed",
        "shape": [target_rows, ROW_TOKENS],
        "dtype": "uint16",
        "token_capacity": target_rows * ROW_TOKENS,
        "scanned_documents": scanned_documents,
        "accepted_documents": accepted_documents,
        "rejected_documents": rejected_documents,
        "document_sampling_modulus": 3,
        "train_rows": int(train_indices.size),
        "validation_rows": int(validation_indices.size),
        "split_seed": 42,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the cleaned 9-step-on-people corpus")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--target-rows", type=int, default=80_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.target_rows < 2:
        parser.error("--target-rows must be at least 2")
    metadata = prepare_clean(args.root.resolve(), args.target_rows, args.force)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
