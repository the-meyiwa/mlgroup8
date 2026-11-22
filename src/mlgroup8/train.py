"""The training loop.

Design rule: everything needed to reproduce a run lives in the config plus the checkpoint.
No global state, no notebook cells, no "it worked that one time".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, TrainConfig, load_config
from .data import CharTokenizer, get_batch, load_corpus, split_ids
from .model import GPT


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr.

    Warmup exists because early gradients are large and noisy; cosine decay because a
    constant rate at the end wastes steps bouncing around the minimum instead of settling.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model: GPT, val: np.ndarray, cfg: TrainConfig, rng: np.random.Generator) -> float:
    model.eval()
    losses = torch.zeros(cfg.eval_batches)
    for i in range(cfg.eval_batches):
        x, y = get_batch(val, cfg.batch_size, model.cfg.block_size, rng)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def save_checkpoint(path: Path, model, optim, step, model_cfg, train_cfg, tokenizer, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
            "tokenizer": tokenizer.state_dict(),
            "best_val": best_val,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="train the mlgroup8 language model")
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--out-dir", default=None, help="override train.out_dir")
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--resume", action="store_true", help="continue from ckpt.pt if present")
    args = ap.parse_args(argv)

    model_cfg, train_cfg = load_config(args.config)
    if args.out_dir:
        train_cfg.out_dir = args.out_dir
    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determinism first: same seed, same run. Costs a little speed, buys trustworthy curves.
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    rng = np.random.default_rng(train_cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    text = load_corpus(train_cfg.data_path)
    tokenizer = CharTokenizer.from_text(text)
    model_cfg.vocab_size = tokenizer.vocab_size
    train_ids, val_ids = split_ids(tokenizer.encode(text), train_cfg.val_frac)

    model = GPT(model_cfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )

    start_step, best_val = 0, float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from step {start_step}")

    n_params = model.num_params()
    print(
        f"vocab {model_cfg.vocab_size} | params (non-embedding) {n_params/1e6:.2f}M | "
        f"tokens train {len(train_ids)} val {len(val_ids)} | steps {start_step}->{train_cfg.max_steps}"
    )
    (out_dir / "config.json").write_text(
        json.dumps({"model": model_cfg.__dict__, "train": train_cfg.__dict__}, indent=2)
    )
    metrics_path = out_dir / "metrics.jsonl"

    t0 = time.time()
    tokens_seen = 0
    for step in range(start_step, train_cfg.max_steps):
        lr = lr_at(step, train_cfg)
        for group in optim.param_groups:
            group["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = get_batch(train_ids, train_cfg.batch_size, model_cfg.block_size, rng)
            _, loss = model(x, y)
            (loss / train_cfg.grad_accum).backward()
            loss_accum += loss.item() / train_cfg.grad_accum
            tokens_seen += x.numel()

        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optim.step()

        if step % train_cfg.log_every == 0:
            dt = time.time() - t0
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                f"{tokens_seen/max(dt,1e-9):,.0f} tok/s"
            )

        if (step + 1) % train_cfg.eval_interval == 0 or step + 1 == train_cfg.max_steps:
            val_loss = evaluate(model, val_ids, train_cfg, rng)
            row = {
                "step": step + 1,
                "train_loss": loss_accum,
                "val_loss": val_loss,
                "val_ppl": math.exp(min(val_loss, 20)),
                "lr": lr,
                "tokens_seen": tokens_seen,
                "elapsed_s": round(time.time() - t0, 1),
            }
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  eval {row}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    out_dir / "best.pt", model, optim, step + 1, model_cfg, train_cfg,
                    tokenizer, best_val,
                )
            save_checkpoint(
                ckpt_path, model, optim, step + 1, model_cfg, train_cfg, tokenizer, best_val
            )

    # One sample on the way out: the fastest way to see whether the loss means anything.
    model.eval()
    idx = torch.tensor(
        [tokenizer.encode(train_cfg.sample_prompt)], dtype=torch.long
    )
    gen = model.generate(idx, train_cfg.sample_tokens, temperature=0.8, top_k=40)
    sample_text = tokenizer.decode(gen[0].tolist())
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / f"step_{train_cfg.max_steps}.txt").write_text(sample_text)
    print("--- sample ---")
    print(sample_text)
    print(f"done in {time.time()-t0:.1f}s, best val {best_val:.4f} (ppl {math.exp(min(best_val,20)):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
