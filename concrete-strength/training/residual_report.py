"""Out-of-fold residual distribution, and how much of the headline is fold luck.

Runs the two model generations (v1 defaults, v2 tuned) through 10 independent
10-fold CV partitions and reports, per seed, the RMSE and the quantiles of
absolute out-of-fold error. Also reports the full-sample quantiles for the
shipped configuration (seed 42) so the numbers quoted in the docs can be
restated as p50 / p99 rather than a single average.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "concrete.csv")
X = df.drop(columns=["strength"]).to_numpy()
y = df["strength"].to_numpy()

V1 = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9, random_state=42)
V2 = dict(n_estimators=600, learning_rate=0.06, max_depth=4, min_samples_leaf=3, subsample=0.8, random_state=42)


def gbr(**kw):
    return Pipeline([("model", GradientBoostingRegressor(**kw))])


def stats(a):
    return {
        "rmse": float(np.sqrt(np.mean(a**2))),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def run(model_factory, seed):
    folds = list(KFold(n_splits=10, shuffle=True, random_state=seed).split(X))
    pred = cross_val_predict(model_factory(), X, y, cv=folds, n_jobs=-1)
    return pred - y


# --- seed variance on the pooled folds
seed_rows = []
for seed in range(10):
    for label, factory in (("v1", lambda: gbr(**V1)), ("v2", lambda: gbr(**V2))):
        err = run(factory, seed)
        s = stats(np.abs(err))
        s.update({"seed": seed, "model": label})
        seed_rows.append(s)
seeds = pd.DataFrame(seed_rows)
seeds.to_csv(ROOT / "training" / "residual_by_seed.csv", index=False)
summary = seeds.groupby("model").agg(
    rmse_mean=("rmse", "mean"), rmse_min=("rmse", "min"), rmse_max=("rmse", "max"),
    p50_mean=("p50", "mean"), p90_mean=("p90", "mean"),
    p99_mean=("p99", "mean"), p99_min=("p99", "min"), p99_max=("p99", "max"),
).round(3)
print("=== 10 independent 10-fold partitions, gradient boosting v1 vs v2")
print(summary.to_string())

# --- model families on the shipped folds (seed 42), with quantiles
families = {
    "linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
    "random_forest": Pipeline([("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1))]),
    "gbr_v1": gbr(**V1),
    "gbr_v2_shipped": gbr(**V2),
}
rows = []
for name, model in families.items():
    err = run(lambda m=model: m, 42)
    a = np.abs(err)
    rows.append({
        "model": name,
        **{k: round(v, 3) for k, v in stats(a).items()},
        "mae": round(float(a.mean()), 3),
        "bias": round(float(err.mean()), 3),
        "within_5": round(float((a <= 5).mean() * 100), 1),
        "worst_rows_over_15": int((a > 15).sum()),
    })
shipped = pd.DataFrame(rows)
shipped.to_csv(ROOT / "training" / "residual_report.csv", index=False)
print()
print("=== absolute out-of-fold error, shipped folds (seed 42), MPa")
print(shipped.to_string(index=False))

# --- who are the p99 rows?
err = run(lambda: gbr(**V2), 42)
a = np.abs(err)
worst = df.assign(abs_err=a, pred=y + err).nlargest(10, "abs_err")
print()
print("=== ten worst out-of-fold predictions")
print(worst[["cement", "water", "age", "strength", "pred", "abs_err"]].round(2).to_string(index=False))
