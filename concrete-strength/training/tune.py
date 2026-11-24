"""Hyperparameter sweep for the concrete strength regressor (DD-05).

Same protocol as train.py: identical 10-fold shuffled splits, seed 42, RMSE
used for selection. Adds a log-target variant and a histogram-based booster
so the model choice is argued with numbers, not taste.

Writes training/tuning.md. Does not touch the served artefact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "concrete.csv"
OUT = ROOT / "training" / "tuning.md"
FEATURES = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
]
SEED = 42


def log_gbr(est):
    """GBR fitted on ln(strength) and inverted on predict (DD-05 variant).

    TransformedTargetRegressor, not a hand-rolled wrapper: a bare wrapper class
    is not a scikit-learn estimator and breaks tag introspection inside
    cross_val_predict.
    """
    return TransformedTargetRegressor(
        regressor=est, func=np.log, inverse_func=np.exp
    )


def score(est, X, y, folds):
    pred = cross_val_predict(est, X, y, cv=folds, n_jobs=2)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES].to_numpy(float), df["strength"].to_numpy(float)
    folds = KFold(n_splits=10, shuffle=True, random_state=SEED)

    rows = []

    gbr = GridSearchCV(
        GradientBoostingRegressor(random_state=SEED),
        {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.06],
            "max_depth": [3, 4],
            "min_samples_leaf": [1, 3],
            "subsample": [0.8, 1.0],
        },
        scoring="neg_root_mean_squared_error",
        cv=folds, n_jobs=2,
    ).fit(X, y)
    rows.append(("GBR tuned " + json.dumps(gbr.best_params_, sort_keys=True), score(gbr.best_estimator_, X, y, folds)))

    hgb = GridSearchCV(
        HistGradientBoostingRegressor(random_state=SEED),
        {
            "learning_rate": [0.05, 0.1],
            "max_iter": [500],
            "max_leaf_nodes": [15, 31],
            "min_samples_leaf": [10, 20],
            "l2_regularization": [0.0, 1.0],
        },
        scoring="neg_root_mean_squared_error",
        cv=folds, n_jobs=2,
    ).fit(X, y)
    rows.append(("HGB tuned " + json.dumps(hgb.best_params_, sort_keys=True), score(hgb.best_estimator_, X, y, folds)))

    rows.append((
        "GBR tuned, log target",
        score(log_gbr(GradientBoostingRegressor(random_state=SEED, **gbr.best_params_)), X, y, folds),
    ))

    for name, m in sorted(rows, key=lambda r: r[1]["rmse"]):
        print(name, "-> RMSE %.3f MAE %.3f R2 %.4f" % (m["rmse"], m["mae"], m["r2"]))

    best_name, best = min(rows, key=lambda r: r[1]["rmse"])
    lines = [
        "# Hyperparameter sweep",
        "",
        "Protocol: 10-fold shuffled CV, seed 42, RMSE used for selection.",
        "",
        "| Candidate | CV RMSE (MPa) | CV MAE (MPa) | CV R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, m in sorted(rows, key=lambda r: r[1]["rmse"]):
        lines.append("| %s | %.3f | %.3f | %.4f |" % (name, m["rmse"], m["mae"], m["r2"]))
    lines += ["", "Best: " + best_name, ""]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
