"""Contract and validation tests (System Design section 9)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.main import app  # noqa: E402
from app.schema import feature_names, feature_order_hash  # noqa: E402

VALID = {
    "cement": 540,
    "blast_furnace_slag": 0,
    "fly_ash": 0,
    "water": 162,
    "superplasticizer": 2.5,
    "coarse_aggregate": 1040,
    "fine_aggregate": 676,
    "age": 28,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["algorithm"] == "GradientBoostingRegressor"
    assert len(body["model_version"]) >= 7
    assert body["metrics"]["rmse"] > 0
    assert body["feature_order_hash"] == feature_order_hash()


def test_features_endpoint_matches_schema(client):
    names = [f["name"] for f in client.get("/api/v1/features").json()["features"]]
    assert names == feature_names()


def test_prediction_shape(client):
    res = client.post("/api/v1/predict", json=VALID)
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["prediction"], float)
    assert body["unit"] == "MPa"
    assert body["model_version"]
    assert "not a substitute" in body["disclaimer"]


def test_prediction_matches_offline_call(client):
    model = joblib.load(ROOT / "api" / "model" / "model.joblib")
    vector = [[VALID[n] for n in feature_names()]]
    expected = round(float(model.predict(vector)[0]), 1)
    got = client.post("/api/v1/predict", json=VALID).json()["prediction"]
    assert got == expected


def test_missing_field_is_422(client):
    payload = dict(VALID)
    del payload["water"]
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 422
    assert res.json()["errors"][0]["field"] == "water"


def test_text_in_number_field_is_422(client):
    payload = dict(VALID, cement="lots")
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 422
    assert res.json()["errors"][0]["field"] == "cement"
    assert "number" in res.json()["errors"][0]["message"]


def test_out_of_range_is_422(client):
    res = client.post("/api/v1/predict", json=dict(VALID, age=1000))
    assert res.status_code == 422
    assert res.json()["errors"][0]["field"] == "age"
    assert "365" in res.json()["errors"][0]["message"]


def test_boolean_is_rejected(client):
    res = client.post("/api/v1/predict", json=dict(VALID, cement=True))
    assert res.status_code == 422


def test_malformed_json_is_400(client):
    res = client.post(
        "/api/v1/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 400
    assert res.json()["errors"][0]["field"] == "body"


def test_rate_limit(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "60")
    from app import main

    main._hits.clear()
    assert client.post("/api/v1/predict", json=VALID).status_code == 200
    assert client.post("/api/v1/predict", json=VALID).status_code == 200
    res = client.post("/api/v1/predict", json=VALID)
    assert res.status_code == 429
    assert "Retry-After" in res.headers
    main._hits.clear()
