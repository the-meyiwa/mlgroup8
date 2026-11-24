# mlgroup8 — language model

Group 8's coursework repo: a small GPT-style language model trained from scratch, with the
plumbing that makes results reportable instead of anecdotal.

Two things live here: this language model, and `concrete-strength/`, a separate
deliverable (compressive strength regression on the UCI 165 dataset) that used to have
its own repository.

Status: foundation. The model, data pipeline, training loop, checkpointing and tests work.
No trained model is committed yet (see "Reproducing" below).

## What this is

A decoder-only transformer (pre-LayerNorm, learned positional embeddings, tied output head)
implemented in plain PyTorch. Nothing is imported from a model zoo and no pretrained weights
are used, so every number in this repo was produced by this repo.

## Layout

    configs/        training configs (yaml). tiny.yaml is the one that runs on a laptop.
    data/           sample corpus lives here; real corpora are downloaded, not committed.
    docs/           the brief, foundations checklist, results writeups.
    scripts/        data download.
    src/mlgroup8/   the package: config, tokenizer, model, training loop.
    tests/          shape, causality, determinism and overfit tests.
    runs/           checkpoints, metrics.jsonl and samples. Gitignored.
    concrete-strength/  separate project: FastAPI service + Next.js form predicting
                    concrete compressive strength. Own README, docs and tests.

## Setup

    python3 -m venv .venv
    .venv/bin/pip install -r requirements-dev.txt
    .venv/bin/pip install -e .

## Reproducing

    .venv/bin/python scripts/download_data.py            # tiny shakespeare -> data/tinyshakespeare.txt
    .venv/bin/python -m mlgroup8.train --config configs/tiny.yaml

Writes runs/tiny/: ckpt.pt (latest, resumable), best.pt, metrics.jsonl (one json object per
eval, so it can be plotted without rerunning), and samples/. Rerun the same command to resume.

Generate:

    .venv/bin/python -m mlgroup8.sample --ckpt runs/tiny/best.pt --prompt "ROMEO: " --tokens 300

## Tests

    .venv/bin/python -m pytest -q

Three tests matter more than the rest:

- test_causality: changing tokens after position t must not change the logits at t. A broken
  mask passes every shape test and produces a model that cheats.
- test_determinism: the same seed and config give the same loss curve, bit for bit.
- test_overfit: the model must memorise a 2KB corpus. If it cannot overfit, the bug is in the
  pipeline, not in the hyperparameters.

## Smoke run

tiny config, 150 steps, 2 CPU cores, 43s: loss 4.215 (uniform guessing is ln 65 = 4.17),
val loss 2.629, val perplexity 13.9, ~15k tokens/sec. Output is word-shaped nonsense, which
is the correct result at 150 steps and proof the pipeline is wired right. Details in
docs/FOUNDATIONS.md.

## Results

Fill in as we go. Per run: steps, params, final val loss, val perplexity, tokens/sec, hardware.
Perplexity is exp(loss); quote both, since a loss of 3.2 means nothing to a reader who does not
know the scale.

## Who does what

| area | owner |
| --- | --- |
| model, training loop | |
| data pipeline, tokenizer | |
| evaluation, writeup | |

Update this table in the same commit that changes ownership. It is the only coordination
mechanism that survives people missing meetings.

## Conventions

- One change per commit, message says what and why.
- No notebooks in main. Experiments are configs, not .ipynb files.
- Never commit data, checkpoints, .venv or API keys. .gitignore already covers them.
- If a number goes in the report, it must be reproducible from a committed config.
