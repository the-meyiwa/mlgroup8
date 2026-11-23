"""Canonical feature schema (DD-06).

The order of the eight features in features.json is the only order that is
allowed to exist. Training and serving both import from this module, so a
column-order mistake cannot be introduced by hand.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FEATURES_PATH = Path(__file__).resolve().parents[2] / "features.json"


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(FEATURES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def features() -> list[Feature]:
    return [
        Feature(
            name=f["name"],
            label=f["label"],
            unit=f["unit"],
            minimum=float(f["min"]),
            maximum=float(f["max"]),
            step=float(f["step"]),
        )
        for f in _load()["features"]
    ]


def feature_names() -> list[str]:
    return [f.name for f in features()]


def target() -> dict:
    return _load()["target"]


def feature_order_hash() -> str:
    """Short hash of the canonical order; both halves of the project report it."""
    joined = ",".join(feature_names())
    return hashlib.sha256(joined.encode()).hexdigest()[:12]
