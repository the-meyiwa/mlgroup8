"""Config loading.

Two dataclasses, one file, so that a run is fully described by a single yaml file and the
checkpoint records exactly what produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 65          # filled in from the tokenizer at train time
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    tie_weights: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 2000
    eval_interval: int = 100
    eval_batches: int = 20
    log_every: int = 10
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    seed: int = 1337
    val_frac: float = 0.1
    data_path: str = "data/tinyshakespeare.txt"
    out_dir: str = "runs/tiny"
    sample_tokens: int = 200
    sample_prompt: str = "\n"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    """Read a yaml config with optional 'model:' and 'train:' sections."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model_raw = raw.pop("model", {}) or {}
    train_raw = raw.pop("train", {}) or {}
    # Anything at the top level is treated as a train setting.
    train_raw = {**raw, **train_raw}

    known_model = set(ModelConfig.__dataclass_fields__)
    unknown_model = set(model_raw) - known_model
    if unknown_model:
        raise ValueError(f"unknown model config keys: {sorted(unknown_model)}")

    known_train = set(TrainConfig.__dataclass_fields__)
    unknown_train = set(train_raw) - known_train
    extra = {k: train_raw.pop(k) for k in unknown_train}

    model = ModelConfig(**model_raw)
    train = TrainConfig(**train_raw, extra=extra)
    return model, train
