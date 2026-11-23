"""FastAPI application (DD-04, DD-08, DD-09, DD-12, DD-13)."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .model import ModelUnavailable, PredictionService
from .schema import features, target

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("csp.api")

DISCLAIMER = (
    "This is an estimate from a machine-learning model trained on laboratory "
    "samples. It is not a substitute for a laboratory compression test."
)

service = PredictionService(config.model_path())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A failure here stops the process (DD-05).
    service.load()
    yield


app = FastAPI(
    title="Concrete Strength Predictor API",
    version="1.0.0",
    description="Predicts 28-day-style compressive strength (MPa) from eight mixture proportions.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)

# -- rate limiting ---------------------------------------------------------
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def guard(request: Request, call_next):
    limit, window = config.rate_limit()
    client = request.client.host if request.client else "unknown"

    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > config.max_payload_bytes():
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"errors": [{"field": "body", "message": "Request body is too large"}]},
        )

    if request.url.path.startswith("/api/"):
        now = time.monotonic()
        bucket = _hits[client]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(window - (now - bucket[0])))
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(retry_after)},
                content={"errors": [{"field": "rate", "message": "Too many requests, slow down"}]},
            )
        bucket.append(now)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    # DD-10: route, status, latency and model version only; never the inputs.
    logger.info(
        "%s %s %s %.1fms model=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        service.version,
    )
    return response


def _error(status_code: int, errors: list[dict[str, str]], headers: dict[str, str] | None = None):
    return JSONResponse(status_code=status_code, content={"errors": errors}, headers=headers)


def validate(payload: Any) -> tuple[list[float] | None, list[dict[str, str]]]:
    """Server-side validation, authoritative (DD-08). Returns (vector, errors)."""
    if not isinstance(payload, dict):
        return None, [{"field": "body", "message": "Request body must be a JSON object"}]

    vector: list[float] = []
    errors: list[dict[str, str]] = []

    for spec in features():
        if spec.name not in payload or payload[spec.name] is None:
            errors.append({"field": spec.name, "message": f"{spec.label} is required"})
            continue
        raw = payload[spec.name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors.append({"field": spec.name, "message": f"{spec.label} must be a number"})
            continue
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            errors.append({"field": spec.name, "message": f"{spec.label} must be a finite number"})
            continue
        if value < spec.minimum or value > spec.maximum:
            errors.append(
                {
                    "field": spec.name,
                    "message": (
                        f"{spec.label} must be between {_fmt(spec.minimum)} and "
                        f"{_fmt(spec.maximum)} {spec.unit}"
                    ),
                }
            )
            continue
        vector.append(value)

    if errors:
        return None, errors
    return vector, []


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


@app.post("/api/v1/predict")
async def predict(request: Request):
    try:
        payload = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error(400, [{"field": "body", "message": "Request body is not valid JSON"}])

    vector, errors = validate(payload)
    if errors:
        return _error(422, errors)

    try:
        value = service.predict(vector or [])
    except ModelUnavailable:
        correlation_id = uuid.uuid4().hex[:12]
        logger.exception("prediction unavailable correlation_id=%s", correlation_id)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "errors": [
                    {
                        "field": "service",
                        "message": "The prediction service is temporarily unavailable",
                    }
                ],
                "correlation_id": correlation_id,
            },
        )

    return {
        "prediction": round(value, 1),
        "unit": target()["unit"],
        "model_version": service.version,
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
async def health():
    return {"status": "ok" if service.loaded else "degraded", "model_loaded": service.loaded}


@app.get("/model-info")
async def model_info():
    return service.info()


@app.get("/api/v1/features")
async def public_features():
    """Lets the interface render the form from the canonical schema (DD-06)."""
    return {
        "target": target(),
        "features": [
            {
                "name": f.name,
                "label": f.label,
                "unit": f.unit,
                "min": f.minimum,
                "max": f.maximum,
                "step": f.step,
            }
            for f in features()
        ],
    }
