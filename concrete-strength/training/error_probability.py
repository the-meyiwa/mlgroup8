"""How often does a prediction miss by 4 MPa or more?

Out-of-fold predictions for the shipped configuration (seed 42, 10-fold),
then the empirical exceedance probabilities requested by the reviewer:
P(|error| > 4), P(|error| > 5), P(|error| > 8), and how those split between
rows whose mix appears once and rows whose mix appears several times with
different measured strengths.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "concrete.csv")
FEATS = [c for c in df.columns if c != "strength"]
X = df[FEATS].to_numpy()
y = df["strength"].to_numpy()

SHIPPED = dict(n_estimators=600, learning_rate=0.06, max_depth=4,
               min_samples_leaf=3, subsample=0.8, random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
pred = cross_val_predict(GradientBoostingRegressor(**SHIPPED), X, y, cv=kf)
err = np.abs(y - pred)
signed = pred - y

def pct(mask):
    return float(mask.mean() * 100)

# duplicate-mix flag: identical feature rows with more than one target value
dup = df.groupby(FEATS)["strength"].transform("nunique") > 1

out = {
    "model": "gradient_boosting_tuned",
    "cv": "10-fold out-of-fold, seed 42",
    "n": int(len(y)),
    "rmse": float(np.sqrt(np.mean(err**2))),
    "median_abs_error": float(np.median(err)),
    "exceedance_pct": {
        ">2": pct(err > 2), ">3": pct(err > 3), ">4": pct(err > 4),
        ">5": pct(err > 5), ">8": pct(err > 8), ">10": pct(err > 10),
    },
    "within_pm4_pct": pct(err <= 4),
    "within_pm5_pct": pct(err <= 5),
    # tail concentration
    "share_of_squared_error_from_top_5pct": float(
        np.sort(err**2)[::-1][: max(1, int(0.05 * len(err)))].sum() / (err**2).sum() * 100),
    # bias check
    "mean_signed_error": float(signed.mean()),
    "prob_overpredict_by_4": pct(signed > 4),
    "prob_underpredict_by_4": pct(signed < -4),
}

# split by duplicate-mix rows
for label, mask in (("unique_mix_rows", ~dup.to_numpy()), ("duplicate_mix_rows", dup.to_numpy())):
    e = err[mask]
    out[label] = {
        "n": int(mask.sum()),
        "median_abs_error": float(np.median(e)),
        "pct_over_4": float((e > 4).mean() * 100),
        "pct_over_8": float((e > 8).mean() * 100),
    }

# by strength band
bands = [(0, 25), (25, 40), (40, 55), (55, 90)]
out["by_strength_band"] = {}
for lo, hi in bands:
    m = (y >= lo) & (y < hi)
    e = err[m]
    out["by_strength_band"][f"{lo}-{hi}"] = {
        "n": int(m.sum()),
        "median_abs_error": float(np.median(e)),
        "pct_over_4": float((e > 4).mean() * 100),
    }

(ROOT / "training" / "error_probability.json").write_text(json.dumps(out, indent=2) + "\n")

lines = ["# Probability of a 4 MPa error", "",
         "Out-of-fold predictions, shipped model, 10-fold CV seed 42, n=%d." % out["n"], "",
         "| threshold | share of predictions |", "|---|---|"]
for k, v in out["exceedance_pct"].items():
    lines.append(f"| error > {k[1:]} MPa | {v:.1f}% |")
lines += ["", f"Within +/-4 MPa: {out['within_pm4_pct']:.1f}%. Within +/-5 MPa: {out['within_pm5_pct']:.1f}%.",
          f"Median absolute error {out['median_abs_error']:.2f} MPa. RMSE {out['rmse']:.2f}.",
          f"Mean signed error {out['mean_signed_error']:+.2f} MPa (positive = overpredicts). "
          f"Overpredicts by >4 MPa {out['prob_overpredict_by_4']:.1f}%, underpredicts by >4 MPa {out['prob_underpredict_by_4']:.1f}%.",
          f"Top 5% of errors carry {out['share_of_squared_error_from_top_5pct']:.0f}% of the squared error.", "",
          "| rows | n | median abs err | %>4 MPa | %>8 MPa |", "|---|---|---|---|---|"]
for label in ("unique_mix_rows", "duplicate_mix_rows"):
    d = out[label]
    lines.append(f"| {label.replace('_', ' ')} | {d['n']} | {d['median_abs_error']:.2f} | {d['pct_over_4']:.1f}% | {d['pct_over_8']:.1f}% |")
lines += ["", "| true strength | n | median abs err | %>4 MPa |", "|---|---|---|---|"]
for k, d in out["by_strength_band"].items():
    lines.append(f"| {k} MPa | {d['n']} | {d['median_abs_error']:.2f} | {d['pct_over_4']:.1f}% |")

(ROOT / "training" / "error_probability.md").write_text("\n".join(lines) + "\n")
print(json.dumps(out, indent=2))
