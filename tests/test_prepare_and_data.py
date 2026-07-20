from __future__ import annotations

import io
import pickle

import numpy as np
import pytest
import torch

from onebit_llm.data import TokenWindowDataset
from onebit_llm.prepare import make_splits, restricted_load, validate_token_rows


class UnsafePayload:
    pass


def test_restricted_unpickler_accepts_plain_tokens() -> None:
    payload = pickle.dumps([[1, 2, 3], [4, 5, 0]])
    assert restricted_load(payload) == [[1, 2, 3], [4, 5, 0]]


def test_restricted_unpickler_rejects_classes() -> None:
    with pytest.raises(pickle.UnpicklingError):
        restricted_load(io.BytesIO(pickle.dumps(UnsafePayload())))


def test_validate_rows_and_token_range() -> None:
    result = validate_token_rows([[1, 2, 0], [3, 4, 5]], expected_shape=(2, 3), vocab_size=6)
    assert result.dtype == np.uint16
    with pytest.raises(ValueError, match="token IDs"):
        validate_token_rows([[1, 6, 0], [3, 4, 5]], expected_shape=(2, 3), vocab_size=6)


def test_splits_are_deterministic_and_disjoint() -> None:
    train_a, val_a = make_splits(100, seed=42)
    train_b, val_b = make_splits(100, seed=42)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert len(train_a) == 90
    assert not set(train_a).intersection(val_a)


def test_token_window_keeps_row_boundary_and_masks_with_padding() -> None:
    tokens = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]], dtype=np.uint16)
    dataset = TokenWindowDataset(tokens, np.array([1]), context_length=4, random_offset=False)
    inputs, targets = dataset[0]
    assert torch.equal(inputs, torch.tensor([7, 8, 9, 0]))
    assert torch.equal(targets, torch.tensor([8, 9, 0, 0]))
    assert 1 not in inputs
