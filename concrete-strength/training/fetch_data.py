"""Download the UCI dataset once and produce data/concrete.csv.

The raw archive is downloaded, hashed and left untouched; column renaming and
whitespace stripping happen here in code so the diff is visible.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "strength",
]


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "cc.zip"

    if not archive.is_file():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    raw_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"archive sha256 {raw_hash}")

    with zipfile.ZipFile(archive) as zf:
        xls = [n for n in zf.namelist() if n.lower().endswith(".xls")]
        if not xls:
            raise SystemExit("No .xls file in the archive")
        (DATA / "Concrete_Data.xls").write_bytes(zf.read(xls[0]))

    frame = pd.read_excel(DATA / "Concrete_Data.xls", header=0)
    if frame.shape[1] != len(COLUMNS):
        raise SystemExit(f"Expected {len(COLUMNS)} columns, found {frame.shape[1]}")
    frame.columns = COLUMNS
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column].astype(str).str.strip(), errors="raise")

    out = DATA / "concrete.csv"
    frame.to_csv(out, index=False, float_format="%.2f")
    print(f"wrote {out} rows={len(frame)} sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
