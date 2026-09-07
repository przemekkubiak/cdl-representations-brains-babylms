#!/usr/bin/env python
"""The developmental trajectory, which is better powered than the endpoint.

`power_analysis.py` shows the single-cell minimum detectable effect for this
estimator is rsa ~ 0.061, against a median best-observed model score of 0.038.
So no per-cell LEVEL comparison in this collection is adequately powered. A
TRAJECTORY is different: across checkpoints of one family in one cell, the brain
RDM, the stimuli, the tokeniser and the architecture are all held fixed, so the
permutation noise that dominates the level is shared and cancels. The quantity
that survives is the within-cell trend, and this repo has 11 checkpoints per
family --- the asset the published analysis does not use.

Computed per (dataset, family, task, session):
  * trend_rho   Spearman(training step, rsa) across that family's checkpoints
  * delta       final minus first checkpoint

Then aggregated three ways:
  1. per family over cells: is the trend consistently signed? (Wilcoxon)
  2. per PARC architecture over seeds: do architectures differ in trajectory
     where they do not differ in level? (Kruskal over per-cell means, seeds
     averaged first --- the package convention)
  3. trajectory agreement between families in the same cell: if the rise and
     fall were signal, independent families should trace it together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RES = ROOT / "results"
MIN_CKPT = 5


def main() -> int:
    a = pd.read_csv(RES / "alignment_rows.csv")
    a = a.dropna(subset=["rsa", "step"])

    rows = []
    for (ds, fam, task, ses), g in a.groupby(["dataset", "family", "task", "session"]):
        g = g.sort_values("step")
        if g.step.nunique() < MIN_CKPT:
            continue
        rho = stats.spearmanr(g.step, g.rsa)
        rows.append(dict(dataset=ds, family=fam, task=task, session=ses,
                         n_ckpt=int(g.step.nunique()),
                         trend_rho=round(float(rho.statistic), 4),
                         trend_p=round(float(rho.pvalue), 4),
                         rsa_first=round(float(g.rsa.iloc[0]), 4),
                         rsa_final=round(float(g.rsa.iloc[-1]), 4),
                         delta=round(float(g.rsa.iloc[-1] - g.rsa.iloc[0]), 4),
                         rsa_mean=round(float(g.rsa.mean()), 4)))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "trajectory_by_cell.csv", index=False)

    # --- 1. per family: is the trend consistently signed across its cells? ---
    fam_rows = []
    for (ds, fam), g in t.groupby(["dataset", "family"]):
        if len(g) < 5:
            continue
        w = stats.wilcoxon(g.trend_rho) if g.trend_rho.abs().sum() > 0 else None
        fam_rows.append(dict(dataset=ds, family=fam, n_cells=len(g),
                             median_trend_rho=round(float(g.trend_rho.median()), 4),
                             frac_positive=round(float((g.trend_rho > 0).mean()), 3),
                             mean_delta=round(float(g.delta.mean()), 5),
                             wilcoxon_trend_p=round(float(w.pvalue), 4) if w else None))
    fam = pd.DataFrame(fam_rows).sort_values(["dataset", "median_trend_rho"])
    fam.to_csv(RES / "trajectory_by_family.csv", index=False)

    # --- 2. PARC architectures: trajectory where the level does not separate --
    p = t[t.family.str.startswith("parc-")].copy()
    arch_rows = []
    if len(p):
        p["arch"] = p.family.str.split("-").str[1]
        p["seed"] = p.family.str.split("-").str[2]
        # package convention: mean within seed first, then spread across seeds
        per_seed = (p.groupby(["dataset", "arch", "seed", "task", "session"])
                    .trend_rho.mean().reset_index())
        for ds, g in per_seed.groupby("dataset"):
            cellmeans = g.groupby(["arch", "seed"]).trend_rho.mean().reset_index()
            groups = [v.trend_rho.values for _, v in cellmeans.groupby("arch")]
            kw = stats.kruskal(*groups) if len(groups) > 1 else None
            for arch, gg in cellmeans.groupby("arch"):
                arch_rows.append(dict(
                    dataset=ds, arch=arch, n_seeds=len(gg),
                    mean_trend_rho=round(float(gg.trend_rho.mean()), 4),
                    sd_across_seeds=round(float(gg.trend_rho.std()), 4),
                    kruskal_arch_p=round(float(kw.pvalue), 4) if kw else None))
    arch = pd.DataFrame(arch_rows)
    arch.to_csv(RES / "trajectory_parc_arch.csv", index=False)

    # --- 3. do independent families trace the same trajectory in a cell? -----
    agree = []
    for (ds, task, ses), g in t.groupby(["dataset", "task", "session"]):
        if g.family.nunique() < 4:
            continue
        v = g.trend_rho.values
        # intraclass-style: is the between-family spread smaller than chance?
        agree.append(dict(dataset=ds, task=task, session=ses, n_families=len(v),
                          mean_trend_rho=round(float(v.mean()), 4),
                          sd_trend_rho=round(float(v.std()), 4),
                          frac_positive=round(float((v > 0).mean()), 3),
                          binom_p=round(float(stats.binomtest(
                              int((v > 0).sum()), len(v), 0.5).pvalue), 4)))
    ag = pd.DataFrame(agree)
    ag.to_csv(RES / "trajectory_cell_agreement.csv", index=False)

    print("=== per family (trend across checkpoints, over that family's cells) ===")
    print(fam.to_string(index=False))
    print()
    print("=== PARC architectures: trajectory ===")
    print(arch.to_string(index=False))
    print()
    print("=== per cell: do independent families agree on the trend direction? ===")
    print(ag.to_string(index=False))
    print()
    print("cells where families agree at p<0.05:",
          int((ag.binom_p < 0.05).sum()), "of", len(ag))
    print("expected by chance:", round(0.05 * len(ag), 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
