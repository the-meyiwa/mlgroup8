"""Does engineering the physics terms beat the raw eight columns? (DD-05 follow-up)

Same folds, same seed, same tuned GradientBoostingRegressor settings as
training/train.py, so the comparison is apples to apples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

SEED, FOLDS = 42, 10
ROOT = "/home/sprite/work/csp-app"

BASE = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
]


def build(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    x = df[BASE].copy()
    binder = x["cement"] + x["blast_furnace_slag"] + x["fly_ash"]
    agg = x["coarse_aggregate"] + x["fine_aggregate"]
    total = binder + x["water"] + x["superplasticizer"] + agg

    if mode in ("ratios", "ratios+age"):
        x = x.assign(
            water_cement=x["water"] / x["cement"],
            water_binder=x["water"] / binder,
            binder=binder,
            aggregate_binder=agg / binder,
            paste_fraction=(binder + x["water"] + x["superplasticizer"]) / total,
            sp_percent_binder=100.0 * x["superplasticizer"] / binder,
            scm_fraction=(x["blast_furnace_slag"] + x["fly_ash"]) / binder,
        )
    if mode == "ratios+age":
        x = x.assign(
            log_age=np.log(x["age"]),
            inv_sqrt_age=1.0 / np.sqrt(x["age"]),
            age_x_water_cement=np.log(x["age"]) / (x["water"] / x["cement"]),
        )
    return x


def gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.06, max_depth=4,
        min_samples_leaf=3, subsample=0.8, random_state=SEED,
    )


def main() -> None:
    df = pd.read_csv(f"{ROOT}/data/concrete.csv")
    y = df["strength"].to_numpy(float)
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    lines = ["| feature set | cols | RMSE | MAE | R2 |", "| --- | --- | --- | --- | --- |"]
    results = {}
    for mode in ("base", "ratios", "ratios+age"):
        X = build(df, mode).to_numpy(float)
        pred = cross_val_predict(gbr(), X, y, cv=kf)
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        results[mode] = (rmse, pred, build(df, mode).columns.tolist())
        lines.append(
            f"| {mode} | {X.shape[1]} | {rmse:.3f} | "
            f"{mean_absolute_error(y, pred):.3f} | {r2_score(y, pred):.4f} |"
        )
        print(lines[-1])

    best = min(results, key=lambda m: results[m][0])
    print(f"\nbest: {best} ({results[best][0]:.3f})")
    print("columns:", results[best][2])

    # paired comparison against the current production feature set
    base_rmse, base_pred, _ = results["base"]
    if best != "base":
        d = np.abs(results[best][1] - y) - np.abs(base_pred - y)
        print(f"mean MAE delta (best - base): {d.mean():+.3f} "
              f"| wins on {int((d < 0).sum())}/{len(d)} samples")

    with open(f"{ROOT}/training/features_experiment.md", "w") as fh:
        fh.write("# Derived-feature experiment\n\n")
        fh.write("Identical 10-fold split, seed 42, production GBR settings.\n\n")
        fh.write("\n".join(lines) + "\n")


main()
