import pytest
import torch

from mlgroup8.config import ModelConfig
from mlgroup8.model import GPT


def tiny_cfg(**over):
    cfg = ModelConfig(
        vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_forward_shapes():
    model = GPT(tiny_cfg())
    x = torch.randint(0, 32, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 32)
    assert loss.dim() == 0 and loss.item() > 0


def test_causality():
    """Changing future tokens must not change past logits. A broken mask passes shape tests."""
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 32, (1, 16))
    x2 = x.clone()
    x2[0, 9:] = torch.randint(0, 32, (7,))

    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)

    assert torch.allclose(a[0, :9], b[0, :9], atol=1e-5)
    assert not torch.allclose(a[0, 9:], b[0, 9:], atol=1e-5)


def test_rejects_sequence_longer_than_block():
    model = GPT(tiny_cfg())
    with pytest.raises(ValueError):
        model(torch.randint(0, 32, (1, 17)))


def test_weight_tying_shares_tensor():
    tied = GPT(tiny_cfg(tie_weights=True))
    assert tied.lm_head.weight is tied.tok_emb.weight
    untied = GPT(tiny_cfg(tie_weights=False))
    assert untied.lm_head.weight is not untied.tok_emb.weight


def test_param_count_excludes_embeddings():
    model = GPT(tiny_cfg())
    total = sum(p.numel() for p in model.parameters())
    assert model.num_params() == total - model.tok_emb.weight.numel() - model.pos_emb.weight.numel()


def test_generate_length_and_determinism():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)

    a = model.generate(idx, 5, temperature=0.8, top_k=8,
                       generator=torch.Generator().manual_seed(1))
    b = model.generate(idx, 5, temperature=0.8, top_k=8,
                       generator=torch.Generator().manual_seed(1))
    assert a.shape == (1, 6)
    assert torch.equal(a, b)


def test_generate_greedy_is_argmax():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, 3, temperature=0.0)
    with torch.no_grad():
        logits, _ = model(idx)
    assert out[0, 1].item() == logits[0, -1].argmax().item()
