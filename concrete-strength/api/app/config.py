"""Runtime configuration, read from the environment."""
from __future__ import annotations

import os
from pathlib import Path


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent.parent / "model" / "model.joblib")))


def rate_limit() -> tuple[int, int]:
    """(requests, window_seconds) per client address."""
    return int(os.getenv("RATE_LIMIT_REQUESTS", "30")), int(os.getenv("RATE_LIMIT_WINDOW", "60"))


def max_payload_bytes() -> int:
    return int(os.getenv("MAX_PAYLOAD_BYTES", "8192"))
