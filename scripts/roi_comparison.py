#!/usr/bin/env python
"""Masked vs unmasked: does anatomical masking change the answer?

This is the comparison the whole ROI rebuild exists to make. The published
datasets are whole-brain; if masking to language- or phonology-responsive cortex
lifts alignment above the untrained baseline, then the near-zero result was a
property of averaging over the whole acquired volume rather than a property of
language models, and every headline in both published packages needs restating.

Pre-declared, so the answer cannot be chosen after the fact:

  A masked variant RESCUES the result iff, on >=3 independent cells, its best
  family mean exceeds the SAME-VARIANT untrained (step-0) band by 2 SD, with the
  below-band rate at chance. A one-sided excess matched by an equal below-band
  rate is a variance artifact, not alignment -- the same rule already applied to
  the PARC reference.

  Masking is NEUTRAL iff the paired per-cell change in best-family alignment has
  a 95% CI containing zero.

Reports per cell and per variant. Writes nothing if no masked sweep exists yet.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RES = ROOT / "results"
KEY = ["dataset", "task", "session"]


def load_grid(griddir: Path) -> pd.DataFrame | None:
    fs = glob.glob(str(griddir / "*" / "alignment_*.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def summarise(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Best trained-family mean per cell, plus that cell's own untrained band."""
    trained = df[df.step > 0] if "step" in df else df
    init = df[df.step == 0] if "step" in df else df.iloc[0:0]
    fam = trained.groupby(KEY + ["family"]).rsa.mean().reset_index()
    best = fam.groupby(KEY).rsa.max().rename("best_family_rsa").reset_index()
    if len(init):
        band = (init.groupby(KEY + ["family"]).rsa.mean().reset_index()
                .groupby(KEY).rsa.agg(["mean", "std", "count"])
                .rename(columns={"mean": "init_mean", "std": "init_sd",
                                 "count": "init_n"}).reset_index())
        best = best.merge(band, on=KEY, how="left")
        best["z_vs_untrained"] = (best.best_family_rsa - best.init_mean) / best.init_sd
        best["beats_untrained_2sd"] = best.z_vs_untrained > 2.0
        best["below_untrained_2sd"] = best.z_vs_untrained < -2.0
    best.insert(0, "variant", variant)
    return best


def main() -> int:
    variants = {"within-run-normalised": ROOT / "grid"}
    for d in sorted(ROOT.glob("grid_roi-*")):
        variants[d.name.replace("grid_", "")] = d

    frames = []
    for v, d in variants.items():
        g = load_grid(d)
        if g is None:
            continue
        frames.append(summarise(g, v))
    if len(frames) < 2:
        print("no masked sweep on disk yet -- nothing to compare "
              f"(found: {[f.variant.iloc[0] for f in frames]})")
        return 0

    allv = pd.concat(frames, ignore_index=True)
    allv.round(6).to_csv(RES / "roi_by_cell.csv", index=False)

    base = allv[allv.variant == "within-run-normalised"].set_index(KEY)
    rows, verdicts = [], {}
    for v in [x for x in allv.variant.unique() if x != "within-run-normalised"]:
        m = allv[allv.variant == v].set_index(KEY)
        common = base.index.intersection(m.index)
        if not len(common):
            continue
        d = (m.loc[common, "best_family_rsa"] - base.loc[common, "best_family_rsa"])
        rng = np.random.default_rng(0)
        bs = rng.integers(0, len(d), size=(10000, len(d)))
        means = d.to_numpy()[bs].mean(axis=1)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        beats = int(m.loc[common, "beats_untrained_2sd"].sum()) if "beats_untrained_2sd" in m else 0
        below = int(m.loc[common, "below_untrained_2sd"].sum()) if "below_untrained_2sd" in m else 0
        rescued = bool(beats >= 3 and beats > below)
        neutral = bool(lo <= 0.0 <= hi)
        verdicts[v] = dict(
            n_cells=int(len(common)), mean_change=round(float(d.mean()), 5),
            ci_lo=round(lo, 5), ci_hi=round(hi, 5),
            cells_beating_untrained_2sd=beats, cells_below_2sd=below,
            verdict=("MASKING RESCUES THE RESULT -- restate both published packages"
                     if rescued else
                     "masking is NEUTRAL: paired change CI contains zero" if neutral else
                     "masking changes alignment but does not clear the untrained band"))
        rows.append(dict(variant=v, **verdicts[v]))
    pd.DataFrame(rows).to_csv(RES / "roi_comparison.csv", index=False)
    (RES / "roi_verdict.json").write_text(json.dumps(verdicts, indent=2))
    print(json.dumps(verdicts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
