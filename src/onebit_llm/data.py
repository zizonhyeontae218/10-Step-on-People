from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainConfig


class TokenWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Yield causal-LM windows without joining separate source rows."""

    def __init__(
        self,
        tokens: np.ndarray,
        row_indices: np.ndarray,
        context_length: int,
        pad_token_id: int = 0,
        random_offset: bool = True,
    ) -> None:
        if tokens.ndim != 2:
            raise ValueError("tokens must be a rank-2 array")
        self.tokens = tokens
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.random_offset = random_offset

    def __len__(self) -> int:
        return int(self.row_indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = np.asarray(self.tokens[int(self.row_indices[index])], dtype=np.int64)
        pad_positions = np.flatnonzero(row == self.pad_token_id)
        valid_length = int(pad_positions[0]) if pad_positions.size else int(row.size)
        valid_length = max(valid_length, 1)
        window_length = self.context_length + 1
        max_start = max(valid_length - window_length, 0)
        if self.random_offset and max_start:
            start = int(torch.randint(max_start + 1, ()).item())
        else:
            start = max_start // 2
        segment = row[start : min(start + window_length, valid_length)]
        window = np.full(window_length, self.pad_token_id, dtype=np.int64)
        window[: segment.size] = segment
        tensor = torch.from_numpy(window)
        return tensor[:-1], tensor[1:]


def load_token_arrays(data_config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_path = Path(data_config.tokens_path)
    splits_path = Path(data_config.splits_path)
    if not tokens_path.exists() or not splits_path.exists():
        raise FileNotFoundError("prepared data is missing; run `python -m onebit_llm.prepare` first")
    tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
    with np.load(splits_path, allow_pickle=False) as splits:
        train_indices = splits["train"].copy()
        val_indices = splits["validation"].copy()
    return tokens, train_indices, val_indices


def build_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    tokens, train_indices, val_indices = load_token_arrays(data_config)
    train_dataset = TokenWindowDataset(
        tokens, train_indices, data_config.context_length, data_config.pad_token_id, random_offset=True
    )
    val_dataset = TokenWindowDataset(
        tokens, val_indices, data_config.context_length, data_config.pad_token_id, random_offset=False
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    common = {
        "batch_size": train_config.micro_batch_size,
        "num_workers": train_config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader
