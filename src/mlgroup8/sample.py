"""Generate text from a checkpoint."""

from __future__ import annotations

import argparse

import torch

from .data import CharTokenizer
from .model import GPT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tiny/best.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    ck = torch.load(args.ckpt, weights_only=False)
    from .config import ModelConfig

    model = GPT(ModelConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    tok = CharTokenizer.from_state_dict(ck["tokenizer"])

    gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k,
                         generator=gen)
    print(tok.decode(out[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
