"""Prediction boundary (DD-02, DD-05).

One object owns the fitted pipeline. It is loaded once when the process starts;
if that fails the process must not start, so a bad deploy never serves numbers
from a missing or corrupt artefact.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from .schema import feature_names, feature_order_hash

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"


class ModelUnavailable(RuntimeError):
    """Raised when the artefact cannot be used to produce a number."""


class PredictionService:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self._model = None
        self.version: str = "unavailable"
        self.meta: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if not self.model_path.is_file():
            raise ModelUnavailable(f"Model artefact not found: {self.model_path}")

        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.version = digest[:9]

        sidecar = self.model_path.with_suffix(".meta.json")
        if sidecar.is_file():
            with open(sidecar, encoding="utf-8") as fh:
                self.meta = json.load(fh)
        else:
            self.meta = {}

        # The artefact must have been trained on the canonical order.
        trained_on = self.meta.get("features")
        if trained_on and trained_on != feature_names():
            raise ModelUnavailable("Artefact feature order does not match the schema")
        order_hash = self.meta.get("feature_order_hash")
        if order_hash and order_hash != feature_order_hash():
            raise ModelUnavailable("Artefact feature order hash does not match the schema")

        with self._lock:
            self._model = joblib.load(self.model_path)

        logger.info(
            "model loaded",
            extra={"model_version": self.version, "algorithm": self.meta.get("algorithm")},
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- prediction --------------------------------------------------------
    def predict(self, vector: Sequence[float]) -> float:
        if self._model is None:
            raise ModelUnavailable("Model artefact is not loaded")
        expected = len(feature_names())
        if len(vector) != expected:
            raise ModelUnavailable(f"Expected {expected} features, got {len(vector)}")
        row = np.asarray([vector], dtype=float)
        try:
            value = float(self._model.predict(row)[0])
        except Exception as exc:  # noqa: BLE001 - surfaced as 503 by the API
            logger.exception("prediction failed")
            raise ModelUnavailable("Prediction failed") from exc
        if not np.isfinite(value):
            raise ModelUnavailable("Prediction produced a non-finite value")
        # Strength cannot be negative; clamp tiny negative overshoot at zero.
        return max(0.0, value)

    # -- reporting ---------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "model_version": self.version,
            "algorithm": self.meta.get("algorithm"),
            "metrics": self.meta.get("metrics", {}),
            "trained_on": {
                "dataset_rows": self.meta.get("dataset_rows"),
                "dataset_hash": self.meta.get("dataset_hash"),
                "trained_at": self.meta.get("trained_at"),
                "metrics_basis": self.meta.get("metrics_basis"),
            },
            "feature_order_hash": feature_order_hash(),
        }
