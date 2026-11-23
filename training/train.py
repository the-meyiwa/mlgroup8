"""Train the concrete strength model offline (DD-05, DD-07).

Usage:
    python training/train.py --data data/concrete.csv --out api/model/model.joblib

Fits a linear baseline and a gradient-boosting ensemble on identical k-fold
splits, selects the lower cross-validated RMSE, refits on all data, writes the
artefact plus a metadata sidecar and an evaluation table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.schema import feature_names, feature_order_hash, target  # noqa: E402

SEED = 42
FOLDS = 10


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def candidates() -> dict[str, Pipeline]:
    return {
        "LinearRegression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "GradientBoostingRegressor": Pipeline(
            [
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "RandomForestRegressor": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1
                    ),
                )
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "concrete.csv"))
    parser.add_argument("--out", default=str(ROOT / "api" / "model" / "model.joblib"))
    parser.add_argument("--report", default=str(ROOT / "training" / "evaluation.md"))
    args = parser.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    names = feature_names()
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")

    X = df[names].to_numpy(dtype=float)
    y = df[target()["name"]].to_numpy(dtype=float)

    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    for name, pipeline in candidates().items():
        pred = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=-1)
        predictions[name] = pred
        rows.append({"model": name, **score(y, pred)})

    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    winner = table.iloc[0]["model"]
    print(table.to_string(index=False))

    best = candidates()[winner]
    best.fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out, compress=3)
    version = sha256_file(out)[:9]

    metrics = {
        k: float(table.iloc[0][k]) for k in ("rmse", "mae", "r2")
    }
    meta = {
        "model_version": version,
        "algorithm": winner,
        "metrics": metrics,
        "metrics_basis": f"{FOLDS}-fold cross-validation, shuffle=True, seed={SEED}",
        "features": names,
        "feature_order_hash": feature_order_hash(),
        "dataset_hash": sha256_file(data_path),
        "dataset_rows": int(len(df)),
        "seed": SEED,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        f"# Evaluation report - {winner}",
        "",
        f"Artefact: `{out.name}`  ",
        f"Model version: `{version}`  ",
        f"Dataset: {len(df)} rows, sha256 `{meta['dataset_hash'][:16]}...`  ",
        f"Protocol: {FOLDS}-fold cross-validation, identical folds for all candidates, seed {SEED}",
        "",
        "| Model | CV RMSE (MPa) | CV MAE (MPa) | CV R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, r in table.iterrows():
        lines.append(f"| {r['model']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.4f} |")
    lines += [
        "",
        f"Selected on lowest cross-validated RMSE: **{winner}**.  ",
        "RMSE is the selection metric because a 5 MPa miss matters the same whether it is over or under.",
        "",
        "## Per-split view",
        "",
        "Cross-validated predictions pooled across all folds; the numbers above are computed on the pooled set.",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out} (version {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
