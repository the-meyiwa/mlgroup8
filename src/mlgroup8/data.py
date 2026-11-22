"""Tokenizer and batching.

Character level on purpose. A BPE tokenizer is a week of debugging for a coursework model;
block_size characters carry the same lesson about sequence modelling and keep the vocabulary
small enough to inspect by hand. Swapping in BPE later means replacing this class only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        if not chars:
            raise ValueError("empty vocabulary")
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # --- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text())["chars"])

    def state_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        return cls(state["chars"])


def split_ids(ids: list[int], val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous train/val split. No shuffling: adjacent tokens are correlated."""
    arr = np.asarray(ids, dtype=np.int64)
    n_val = int(len(arr) * val_frac)
    if n_val < 1:
        raise ValueError("validation split would be empty; add more data or raise val_frac")
    return arr[:-n_val], arr[-n_val:]


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample batch_size windows of block_size tokens, and the same windows shifted by one.

    x[i, t] = data[s+t], y[i, t] = data[s+t+1]: y is x with the next-token target, which is
    the entire training signal for a language model.
    """
    if len(data) <= block_size + 1:
        raise ValueError(
            f"corpus too small ({len(data)} tokens) for block_size {block_size}"
        )
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s : s + block_size] for s in starts])
    y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts])
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )


def load_corpus(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py"
        )
    text = path.read_text(encoding="utf-8")
    if len(text) < 10_000:
        raise ValueError(f"{path} is only {len(text)} chars; need at least 10k")
    return text
