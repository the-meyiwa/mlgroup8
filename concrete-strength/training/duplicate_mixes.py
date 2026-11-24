"""How much of the error tail is label noise on repeated mixes?"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "concrete.csv")
feat = [c for c in df.columns if c != "strength"]

g = df.groupby(feat, dropna=False)["strength"]
dup = g.agg(["count", "min", "max", "mean"]).reset_index()
repeated = dup[dup["count"] > 1].copy()
repeated["spread"] = repeated["max"] - repeated["min"]

print(f"rows: {len(df)}")
print(f"unique mixes: {len(dup)}")
print(f"mixes appearing more than once: {len(repeated)} covering {int(repeated['count'].sum())} rows")
print(f"median within-mix spread: {repeated['spread'].median():.2f} MPa")
print(f"p90 within-mix spread: {repeated['spread'].quantile(0.9):.2f} MPa")
print(f"max within-mix spread: {repeated['spread'].max():.2f} MPa")
print(f"repeated-mix rows with spread over 15 MPa: {int((repeated['spread'] > 15).sum())}")

print()
print("worst disagreements between identical mixes:")
print(
    repeated.nlargest(5, "spread")[feat[:3] + ["count", "min", "max", "spread"]]
    .round(2)
    .to_string(index=False)
)
