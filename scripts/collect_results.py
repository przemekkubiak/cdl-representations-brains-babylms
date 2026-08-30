#!/usr/bin/env python
"""Flatten the per-family grid CSVs into one tidy table + a scaling summary.

Outputs (brainalign-evals/results/):
  alignment_rows.csv    one row per (dataset, family, checkpoint, task, session)
                        with the RSA metrics, the cell's noise ceiling and the
                        ceiling-normalised value.
  scaling_curve.csv     alignment vs parameter count, per dataset x cell, which
                        is the headline: does brain alignment grow with scale?
"""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/local/scratch/sas245/brainalign-evals")
GRID = ROOT / "grid"
RDM = ROOT / "data/ds003604-session-rdms"
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)


def ceilings() -> pd.DataFrame:
    """(dataset, task, session) -> inter-subject noise ceiling, off the RDM files."""
    rows = []
    for f in glob.glob(str(RDM / "*/within-run-normalised/*/session_rdm_*.npz")):
        p = Path(f)
        ds = p.parts[-4]
        task = p.parts[-2]
        ses = re.sub(r"session_rdm_(.*)\.npz", r"\1", p.name)
        d = np.load(f, allow_pickle=True)
        rows.append(dict(
            dataset=ds, task=task, session=ses,
            n_stim=int(d["rdm"].shape[0]),
            ceiling_lower=float(d["noise_ceiling_lower"]) if "noise_ceiling_lower" in d.files else np.nan,
            ceiling_upper=float(d["noise_ceiling_upper"]) if "noise_ceiling_upper" in d.files else np.nan,
            ceiling_n=int(d["noise_ceiling_n"]) if "noise_ceiling_n" in d.files else -1,
            n_subjects=int(d["n_subjects"]) if "n_subjects" in d.files else -1,
        ))
    return pd.DataFrame(rows)


def params_map() -> dict:
    """family -> parameter count, from the pipeline's cached table where possible."""
    m = {}
    known = ROOT / "data/cdl-devai-results/corrected-sweep/grid/model_params.csv"
    repo_params = {}
    if known.exists():
        d = pd.read_csv(known)
        repo_params = dict(zip(d["repo"], d["params"]))
    zoo_path = ROOT / "configs/model_zoo_extended.yaml"
    if not zoo_path.exists():
        zoo_path = ROOT / "pipeline/configs/model_zoo.yaml"
    zoo = yaml.safe_load(zoo_path.read_text())["families"]
    for fam, cfg in zoo.items():
        repo = cfg.get("hf_repo")
        if repo in repo_params:
            m[fam] = int(repo_params[repo])
    # anything not in the cached table is filled from the Hub by fill_params.py
    extra = RES / "params_extra.json"
    if extra.exists():
        m.update({k: int(v) for k, v in json.load(open(extra)).items()})
    return m


def main() -> None:
    files = sorted(glob.glob(str(GRID / "*/alignment_*.csv")))
    if not files:
        print("no alignment CSVs yet under", GRID)
        return
    frames = []
    for f in files:
        p = Path(f)
        d = pd.read_csv(f)
        d["dataset"] = d.get("dataset", p.parent.name)
        d["family"] = d.get("family", p.name[len("alignment_"):-len(".csv")])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    c = ceilings()
    df = df.merge(c, on=["dataset", "task", "session"], how="left",
                  suffixes=("", "_ceil"))
    df["frac_of_ceiling"] = df["rsa"] / df["ceiling_lower"]

    pm = params_map()
    df["params"] = df["family"].map(pm)
    df["cell"] = df["dataset"] + "/" + df["task"] + "/" + df["session"]

    df.to_csv(RES / "alignment_rows.csv", index=False)
    print(f"alignment_rows.csv: {len(df)} rows, "
          f"{df['family'].nunique()} families, {df['cell'].nunique()} cells")

    # --- scaling curve: mean over checkpoints, per family x cell -------------
    g = (df.groupby(["dataset", "task", "session", "family", "params"],
                    dropna=False)
           .agg(n_checkpoints=("rsa", "size"),
                rsa_mean=("rsa", "mean"),
                rsa_sd=("rsa", "std"),
                rsa_max=("rsa", "max"),
                frac_of_ceiling_mean=("frac_of_ceiling", "mean"),
                ceiling_lower=("ceiling_lower", "first"))
           .reset_index())
    g.to_csv(RES / "scaling_curve.csv", index=False)
    print(f"scaling_curve.csv: {len(g)} family x cell rows")

    # --- headline: alignment vs log-params, per dataset ---------------------
    lines = []
    from scipy.stats import spearmanr
    for ds, sub in g.dropna(subset=["params"]).groupby("dataset"):
        per_fam = sub.groupby(["family", "params"])["rsa_mean"].mean().reset_index()
        if len(per_fam) >= 3:
            rho, p = spearmanr(per_fam["params"], per_fam["rsa_mean"])
            lines.append(dict(dataset=ds, n_families=len(per_fam),
                              scale_trend_rho=rho, scale_trend_p=p,
                              best_family=per_fam.loc[per_fam["rsa_mean"].idxmax(), "family"],
                              best_rsa_mean=per_fam["rsa_mean"].max()))
    if lines:
        pd.DataFrame(lines).to_csv(RES / "scale_trend.csv", index=False)
        print("scale_trend.csv:")
        print(pd.DataFrame(lines).to_string(index=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Null-referenced reporting.
#
# The instrument, not the models, is the object of study here: the pipeline's own
# positive control fails on all three datasets (0/108, 0/6, 0/8 stimulus controls
# significant) and per-cell means from real families correlate with pure-noise
# PARC seeds at r = +0.987. So an alignment number compared against ZERO is
# meaningless -- every cell has its own idiosyncratic bias that any model, or any
# noise seed, will reproduce. The only honest reference is the same statistic
# measured on models that differ from each other by nothing but random seed.
#
# PARC supplies that: 18 runs (3 architectures x 6 seeds) that share data and
# order and differ only by initialisation. This section expresses every family's
# per-cell alignment as a z-score against the PARC spread AT THAT SAME CELL,
# which is the like-for-like comparison.
# ---------------------------------------------------------------------------
def null_reference() -> None:
    rows = RES / "alignment_rows.csv"
    if not rows.exists():
        return
    df = pd.read_csv(rows)
    is_null = df["family"].str.startswith("parc-")
    if not is_null.any():
        print("\n[null] no PARC rows yet -- skipping null-referenced table. "
              "Run the PARC families to populate it.")
        return

    # Null distribution per cell: mean over checkpoints within a seed, then the
    # spread ACROSS seeds. Pooling raw checkpoint rows would count one
    # trajectory many times and shrink the sd artificially.
    per_seed = (df[is_null]
                .groupby(["dataset", "task", "session", "family"])["rsa"]
                .mean().reset_index())
    null = (per_seed.groupby(["dataset", "task", "session"])["rsa"]
            .agg(null_mean="mean", null_sd="std", null_n="size",
                 null_min="min", null_max="max").reset_index())

    real = (df[~is_null]
            .groupby(["dataset", "task", "session", "family", "params"],
                     dropna=False)["rsa"].mean().reset_index()
            .rename(columns={"rsa": "rsa_mean"}))
    m = real.merge(null, on=["dataset", "task", "session"], how="left")
    m["z_vs_null"] = (m["rsa_mean"] - m["null_mean"]) / m["null_sd"]
    m["exceeds_null_range"] = m["rsa_mean"] > m["null_max"]
    m["cell"] = m["dataset"] + "/" + m["task"] + "/" + m["session"]
    m.to_csv(RES / "null_referenced.csv", index=False)
    print(f"\nnull_referenced.csv: {len(m)} family x cell rows against "
          f"{per_seed['family'].nunique()} noise seeds")

    # The headline test: is any family distinguishable from a random seed?
    agg = (m.groupby("family")
             .agg(n_cells=("z_vs_null", "size"),
                  mean_z=("z_vs_null", "mean"),
                  max_z=("z_vs_null", "max"),
                  cells_beating_null=("exceeds_null_range", "sum"))
             .reset_index().sort_values("mean_z", ascending=False))
    agg.to_csv(RES / "null_summary.csv", index=False)
    print(agg.to_string(index=False))

    # And the r=+0.987 check: do per-cell means track the noise runs?
    from scipy.stats import pearsonr
    cellmean = m.groupby(["dataset", "task", "session"]).agg(
        real=("rsa_mean", "mean"), noise=("null_mean", "first")).dropna()
    if len(cellmean) >= 3:
        r, p = pearsonr(cellmean["real"], cellmean["noise"])
        print(f"\n[null] per-cell means, real families vs noise seeds: "
              f"r = {r:+.4f} (p = {p:.3g}, n = {len(cellmean)} cells)")
        print("       (published ds003604 value was r = +0.987 -- if this "
              "reproduces, the structure belongs to the cell, not the model)")


if __name__ == "__main__":
    null_reference()
