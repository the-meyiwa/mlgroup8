"""Schedule properties and the overfit test.

The overfit test is the one that catches a broken pipeline: a small model must be able to
memorise a tiny corpus. If it cannot, the bug is in the data or the loop, not the
hyperparameters.
"""

import numpy as np
import pytest
import torch

from mlgroup8.config import ModelConfig, TrainConfig
from mlgroup8.data import CharTokenizer, get_batch
from mlgroup8.model import GPT
from mlgroup8.train import lr_at


def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(1e-3 / 10)
    assert lr_at(9, cfg) == pytest.approx(1e-3)
    assert lr_at(110, cfg) == pytest.approx(1e-4)
    mid = lr_at(60, cfg)
    assert 1e-4 < mid < 1e-3


def test_overfits_tiny_corpus():
    text = "the quick brown fox jumps over the lazy dog. " * 40
    tok = CharTokenizer.from_text(text)
    ids = np.asarray(tok.encode(text), dtype=np.int64)

    cfg = ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2,
                      n_head=2, n_embd=64, dropout=0.0)
    model = GPT(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)

    first = last = None
    for step in range(120):
        x, y = get_batch(ids, 16, cfg.block_size, rng)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert first is not None and last is not None
    assert last < 0.5, f"failed to overfit: loss went {first:.3f} -> {last:.3f}"
    assert last < first / 3
