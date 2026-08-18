#!/usr/bin/env python
"""T2.4 — held-out predictive validation of the mechanistic->alignment claim.

Leave-one-family-out: fit a linear model predicting per-step brain alignment from
the pico-analyze mechanistic metrics on N-1 families, predict the held-out family,
report out-of-sample R². This converts the in-sample R2 correlation into a
falsifiable, cross-model predictive claim (what ICLR reviewers ask for).

Usage:
  python scripts/heldout_predictor.py --families A B C --grid-dir ... --out ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["per", "hoyer", "cka_to_prev", "condition_number", "norm"]


def _load(grid, fam):
    a = Path(grid) / f"alignment_{fam}.csv"
    m = Path(grid) / f"mechanistic_{fam}.csv"
    if not (a.exists() and m.exists()):
        return None
    al = pd.read_csv(a).groupby("step", as_index=False)["rsa"].mean()
    me = pd.read_csv(m)
    d = al.merge(me, on="step", how="inner").dropna(subset=["rsa"])
    d["family"] = fam
    return d


def _r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--grid-dir", default="data/processed/language_models/devai_grid")
    ap.add_argument("--out", default="data/processed/language_models/devai")
    args = ap.parse_args()

    frames = [d for f in args.families if (d := _load(args.grid_dir, f)) is not None]
    if len(frames) < 2:
        print("Need >=2 families with alignment+mechanistic CSVs; skipping held-out CV.")
        return
    data = pd.concat(frames, ignore_index=True)
    cols = [c for c in METRICS if c in data and not data[c].isna().all()]
    data = data.dropna(subset=cols + ["rsa"])

    rows = []
    for held in data["family"].unique():
        tr = data[data["family"] != held]
        te = data[data["family"] == held]
        if len(tr) < len(cols) + 2 or len(te) < 2:
            continue
        Xtr = np.c_[np.ones(len(tr)), tr[cols].to_numpy()]
        Xte = np.c_[np.ones(len(te)), te[cols].to_numpy()]
        coef, *_ = np.linalg.lstsq(Xtr, tr["rsa"].to_numpy(), rcond=None)
        r2 = _r2(te["rsa"].to_numpy(), Xte @ coef)
        rows.append({"held_out_family": held, "r2_heldout": r2, "n_test": len(te)})

    res = pd.DataFrame(rows)
    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    res.to_csv(outp / "heldout_predictor.csv", index=False)
    if len(res):
        print(res.to_string(index=False))
        print(f"\nMean held-out R² = {res['r2_heldout'].mean():.3f}")
    print(f"Saved {outp/'heldout_predictor.csv'}")


if __name__ == "__main__":
    main()
