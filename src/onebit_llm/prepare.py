from __future__ import annotations

import argparse
import io
import json
import os
import pickle
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np
from huggingface_hub import hf_hub_download

DATASET_REPO = "oz1115/korean-pretraining-corpus-ko"
DATASET_REVISION = "324488e1befab3a9b4eac7571bf1557ef4a4eeec"
TOKENIZER_REPO = "oz1115/korean-gpt-150m-ko"
TOKENIZER_REVISION = "669560d7d1de8c3213e43a343e577a80ad6cb7ee"
EXPECTED_ROWS = 24_396
EXPECTED_COLUMNS = 512
VOCAB_SIZE = 32_000


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that rejects every global/class lookup."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(f"global object is forbidden: {module}.{name}")


def restricted_load(source: BinaryIO | bytes) -> object:
    """Load pickle data containing only primitive Python values and containers."""
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    return RestrictedUnpickler(stream).load()


def validate_token_rows(
    rows: object,
    expected_shape: tuple[int, int] = (EXPECTED_ROWS, EXPECTED_COLUMNS),
    vocab_size: int = VOCAB_SIZE,
) -> np.ndarray:
    """Validate nested token IDs and return a compact uint16 array."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("dataset must be a list or tuple of token rows")
    if len(rows) != expected_shape[0]:
        raise ValueError(f"expected {expected_shape[0]} rows, got {len(rows)}")
    if any(not isinstance(row, (list, tuple)) or len(row) != expected_shape[1] for row in rows):
        raise ValueError(f"every row must contain exactly {expected_shape[1]} token IDs")

    array = np.asarray(rows, dtype=np.int64)
    if array.shape != expected_shape:
        raise ValueError(f"expected shape {expected_shape}, got {array.shape}")
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(f"token IDs must be in [0, {vocab_size - 1}], got [{minimum}, {maximum}]")
    return array.astype(np.uint16)


def make_splits(num_rows: int, train_fraction: float = 0.9, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, disjoint row-level train and validation indices."""
    if num_rows < 2 or not 0.0 < train_fraction < 1.0:
        raise ValueError("need at least two rows and a train fraction between zero and one")
    indices = np.random.default_rng(seed).permutation(num_rows)
    train_count = int(num_rows * train_fraction)
    return np.sort(indices[:train_count]), np.sort(indices[train_count:])


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def prepare(root: Path, force: bool = False) -> dict[str, object]:
    """Download, validate, and convert the pinned educational token subset."""
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    tokenizer_dir = root / "data" / "tokenizer"
    for directory in (raw_dir, processed_dir, tokenizer_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tokens_path = processed_dir / "tokens.npy"
    splits_path = processed_dir / "splits.npz"
    metadata_path = processed_dir / "metadata.json"
    tokenizer_path = tokenizer_dir / "tokenizer.json"

    if not force and all(path.exists() for path in (tokens_path, splits_path, metadata_path, tokenizer_path)):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    pickle_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename="val.pkl",
            repo_type="dataset",
            revision=DATASET_REVISION,
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

    with pickle_path.open("rb") as handle:
        rows = restricted_load(handle)
    tokens = validate_token_rows(rows)
    del rows
    train_indices, val_indices = make_splits(len(tokens), seed=42)

    _atomic_save_npy(tokens_path, tokens)
    _atomic_save_npz(splits_path, train=train_indices, validation=val_indices)
    if downloaded_tokenizer.resolve() != tokenizer_path.resolve():
        raise RuntimeError(f"tokenizer was downloaded to an unexpected path: {downloaded_tokenizer}")

    metadata: dict[str, object] = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "source_file": "val.pkl",
        "shape": list(tokens.shape),
        "dtype": str(tokens.dtype),
        "token_count": int(tokens.size),
        "train_rows": int(train_indices.size),
        "validation_rows": int(val_indices.size),
        "split_seed": 42,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned Korean token subset")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    metadata = prepare(args.root.resolve(), force=args.force)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
