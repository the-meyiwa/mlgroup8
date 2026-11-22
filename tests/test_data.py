import numpy as np
import pytest
import torch

from mlgroup8.data import CharTokenizer, get_batch, split_ids


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("hello world\n!")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
    assert tok.vocab_size == len(tok.chars)


def test_tokenizer_state_dict_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("abcdef")
    again = CharTokenizer.from_state_dict(tok.state_dict())
    assert again.encode("cafe") == tok.encode("cafe")

    tok.save(tmp_path / "tok.json")
    loaded = CharTokenizer.load(tmp_path / "tok.json")
    assert loaded.chars == tok.chars


def test_encode_unknown_char_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.encode("abz")


def test_split_is_contiguous_and_sized():
    ids = list(range(1000))
    tr, va = split_ids(ids, 0.1)
    assert len(va) == 100
    assert len(tr) == 900
    # No overlap and the val split is the tail, so no window straddles the boundary.
    assert tr[-1] == 899 and va[0] == 900


def test_get_batch_shapes_and_shift():
    data = np.arange(500, dtype=np.int64)
    rng = np.random.default_rng(0)
    x, y = get_batch(data, batch_size=4, block_size=16, rng=rng)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    # y is x shifted one token: the whole training signal.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_get_batch_rejects_short_corpus():
    with pytest.raises(ValueError):
        get_batch(np.arange(8, dtype=np.int64), 2, 16, np.random.default_rng(0))
