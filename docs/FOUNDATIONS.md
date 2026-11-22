# Foundations

Three layers, in order. Each one is a thing you must be able to do without looking it up,
not a thing you have read once.

## 1. Conceptual

Before touching code, be able to do these on paper:

- [ ] Cross-entropy, and why perplexity is just exp(loss). If someone quotes a loss of 2.6,
      you should be able to say "perplexity about 13, so roughly thirteen-way confusion per
      token" without a calculator.
- [ ] Softmax, logits and temperature. Why temperature 0 is argmax and why top-k truncation
      happens before the renormalisation, not after.
- [ ] Backprop through a residual block: why x + f(x) gives the gradient a clean identity path,
      and what that buys at depth.
- [ ] Self-attention as three projections plus a mask. Be able to write the shapes:
      (B, T, C) -> (B, H, T, C/H) for q, k, v; scores (B, H, T, T); output back to (B, T, C).
      Nine of ten transformer bugs are a missing transpose or a wrong head dimension.
- [ ] Positional information: learned embeddings here, but know what rotary and sinusoidal
      encodings do differently and why length generalisation is the trade.
- [ ] Pre-LN vs post-LN and why this repo uses pre-LN.
- [ ] Why the output head and input embedding can share weights.

## 2. Code

- [ ] Read src/mlgroup8/model.py end to end and be able to redraw the block diagram from memory.
- [ ] Write a training loop from an empty file: forward, loss, zero_grad, backward, clip, step.
      Know why zero_grad comes before backward and what set_to_none changes.
- [ ] Understand the batch sampler. x and y are the same window shifted one token, and that is
      the entire supervision signal.
- [ ] Be able to explain every hyperparameter in configs/tiny.yaml and what happens when you
      halve or double it.
- [ ] Checkpointing: what is in the file and why the optimiser state is in it.

## 3. Rig

This is what separates a pass from a distinction, because it is what makes results citable.

- [ ] Deterministic seeds for python, numpy and torch. Same config, same curve.
- [ ] Validation split, evaluated on a schedule, written to metrics.jsonl rather than printed
      once and forgotten.
- [ ] Checkpoint every eval, keep the best, support resume. A run you cannot interrupt is a run
      you cannot afford.
- [ ] One config file per run. The report cites the config, not a screenshot.
- [ ] Log tokens/sec and hardware, because "trained for 5000 steps" means nothing without them.
- [ ] Quote val loss and val perplexity together, and say which checkpoint.

## 4. What not to do

- Do not fine-tune a pretrained model for this coursework unless the brief explicitly allows
  it. That teaches prompt engineering, not language modelling, and it is obvious in a viva.
- Do not build a BPE tokenizer before the char-level model trains. Order of work matters.
- Do not report a single run. Seed variance across three seeds is the cheapest credibility
  you can buy.

## 5. Smoke numbers

Recorded from this repo on 2 CPU cores, tiny config, 150 steps, tiny shakespeare:

    start loss 4.2152 (ln 65 = 4.17, i.e. uniform guessing)
    step 150: train 2.6135, val 2.6293, val ppl 13.86, ~15k tok/s, 43s
    sample: word-shaped nonsense, as expected at 150 steps

Uniform-guessing loss is ln(vocab). If your step-0 loss is far above that, the init is wrong.
If it is far below, you have a leak.
