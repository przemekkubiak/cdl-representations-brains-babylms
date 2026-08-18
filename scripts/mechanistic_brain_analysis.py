#!/usr/bin/env python
"""Join brain alignment, LM isolation, and mechanistic metrics; test the DevAI claims.

Consumes the three CSVs from run_devai_grid.py (+ optional brain specialization) and
computes:

  R1  Alignment trajectory      : does brain-LM RSA rise over training? (slope vs step)
  R2  Mechanistic correlation   : alignment ~ each pico-analyze metric across checkpoints,
                                  Spearman + partial correlation controlling for step.
  R3  Isolation comparison      : LM isolation (Gini) vs brain isolation per phenomenon,
                                  plus developmental-order match (Kendall tau: LM onset
                                  step vs brain onset age).
  R5  Mechanism of isolation    : LM isolation ~ mechanistic metrics across checkpoints.

Outputs devai_summary_<family>.csv (+ figures) under --output-dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

METRICS = ["norm", "gini", "hoyer", "per", "condition_number", "cka_to_prev"]


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank residuals)."""
    x, y, z = map(lambda v: rankdata(np.asarray(v, float)), (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    rx, ry = resid(x, z), resid(y, z)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan"), float("nan")
    r, p = spearmanr(rx, ry)
    return float(r), float(p)


def _onset_step(steps, values, frac=0.5):
    """First step at which `values` reaches frac of its max (development onset)."""
    steps = np.asarray(steps, float)
    v = np.asarray(values, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return np.nan
    thr = frac * np.nanmax(v)
    hit = np.where(v >= thr)[0]
    return float(steps[hit[0]]) if hit.size else float(steps[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--grid-dir", default="data/processed/language_models/devai_grid")
    ap.add_argument("--brain-specialization", default=None,
                    help="brain_specialization.csv [phenomenon, brain_localization, onset_age?]")
    ap.add_argument("--output-dir", default="data/processed/language_models/devai")
    args = ap.parse_args()

    g = Path(args.grid_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fam = args.family

    def _read(name):
        f = g / f"{name}_{fam}.csv"
        return pd.read_csv(f) if f.exists() else None

    align = _read("alignment")
    iso = _read("isolation")
    mech = _read("mechanistic")
    behav = _read("behaviour")
    abl_b = _read("ablation_behaviour")
    abl_a = _read("ablation_alignment")
    summary = []

    # ---- R1: alignment trajectory (slope of RSA vs log-step) ---------------
    if align is not None and len(align):
        a = align.dropna(subset=["rsa"]).copy()
        a["logstep"] = np.log1p(a["step"].clip(lower=0))
        per_step = a.groupby("step", as_index=False)["rsa"].mean()
        if len(per_step) >= 3:
            r, p = spearmanr(per_step["step"], per_step["rsa"])
            summary.append({"claim": "R1_alignment_rises", "stat": "spearman(step,rsa)",
                            "value": r, "p": p, "n": len(per_step)})

    # ---- R2: mechanistic correlation with alignment ------------------------
    if align is not None and mech is not None and len(align) and len(mech):
        al = align.groupby("step", as_index=False)["rsa"].mean()
        merged = al.merge(mech, on="step", how="inner")
        for metric in METRICS:
            if metric not in merged or merged[metric].isna().all():
                continue
            sub = merged.dropna(subset=["rsa", metric])
            if len(sub) < 4:
                continue
            r, p = spearmanr(sub["rsa"], sub[metric])
            pr, pp = partial_spearman(sub["rsa"], sub[metric], sub["step"])
            summary.append({"claim": "R2_alignment_vs_mechanistic", "stat": f"spearman(rsa,{metric})",
                            "value": r, "p": p, "n": len(sub)})
            summary.append({"claim": "R2_partial_control_step", "stat": f"partial(rsa,{metric}|step)",
                            "value": pr, "p": pp, "n": len(sub)})

    # ---- R3: isolation comparison (LM vs brain) ----------------------------
    if iso is not None and len(iso):
        # LM isolation per phenomenon: mean Gini over the late half of training
        iso2 = iso.dropna(subset=["gini"]).copy()
        late = iso2[iso2["step"] >= iso2["step"].median()]
        lm_iso = late.groupby("phenomenon", as_index=False)["gini"].mean() \
            .rename(columns={"gini": "lm_isolation"})
        # LM onset step per phenomenon
        onsets = []
        for ph, sub in iso2.sort_values("step").groupby("phenomenon"):
            onsets.append({"phenomenon": ph, "lm_onset_step": _onset_step(sub["step"], sub["gini"])})
        lm_onset = pd.DataFrame(onsets)

        if args.brain_specialization and Path(args.brain_specialization).exists():
            bs = pd.read_csv(args.brain_specialization)
            bs = bs.groupby("phenomenon", as_index=False).agg(
                {c: "mean" for c in ["brain_localization", "onset_age"] if c in bs.columns})
            cmp = lm_iso.merge(bs, on="phenomenon", how="inner").merge(lm_onset, on="phenomenon")
            cmp.to_csv(out / f"isolation_comparison_{fam}.csv", index=False)
            if "brain_localization" in cmp and len(cmp) >= 3:
                r, p = spearmanr(cmp["lm_isolation"], cmp["brain_localization"])
                summary.append({"claim": "R3_isolation_LM_vs_brain", "stat": "spearman(lm_iso,brain_iso)",
                                "value": r, "p": p, "n": len(cmp)})
            if "onset_age" in cmp and len(cmp) >= 3:
                tau, p = kendalltau(cmp["lm_onset_step"], cmp["onset_age"])
                summary.append({"claim": "R3_developmental_order", "stat": "kendall(lm_onset,brain_age)",
                                "value": tau, "p": p, "n": len(cmp)})
        else:
            lm_iso.merge(lm_onset, on="phenomenon").to_csv(
                out / f"isolation_comparison_{fam}.csv", index=False)

    # ---- R5: mechanism of isolation ---------------------------------------
    if iso is not None and mech is not None and len(iso) and len(mech):
        iso_step = iso.dropna(subset=["gini"]).groupby("step", as_index=False)["gini"].mean() \
            .rename(columns={"gini": "lm_isolation"})
        m2 = iso_step.merge(mech, on="step", how="inner")
        for metric in METRICS:
            if metric not in m2 or m2[metric].isna().all():
                continue
            sub = m2.dropna(subset=["lm_isolation", metric])
            if len(sub) < 4:
                continue
            r, p = spearmanr(sub["lm_isolation"], sub[metric])
            summary.append({"claim": "R5_isolation_vs_mechanistic", "stat": f"spearman(iso,{metric})",
                            "value": r, "p": p, "n": len(sub)})

    # ---- R6: CAUSAL ablation (T2.1) — circuit vs random, paired -----------
    if abl_b is not None and len(abl_b):
        for col, label in [("drop_localized", "behav_drop_circuit"),
                           ("drop_random", "behav_drop_random"),
                           ("causal_selectivity", "behav_causal_selectivity")]:
            if col in abl_b:
                summary.append({"claim": "R6_causal_behaviour", "stat": label,
                                "value": float(abl_b[col].mean()), "p": np.nan, "n": len(abl_b)})
        # paired test: does the localized circuit hurt more than random?
        if {"drop_localized", "drop_random"}.issubset(abl_b.columns):
            from scipy.stats import wilcoxon
            d = abl_b.dropna(subset=["drop_localized", "drop_random"])
            if len(d) >= 5 and (d["drop_localized"] - d["drop_random"]).abs().sum() > 0:
                try:
                    _, p = wilcoxon(d["drop_localized"], d["drop_random"])
                    summary.append({"claim": "R6_causal_behaviour", "stat": "wilcoxon(circuit>random)",
                                    "value": float((d["drop_localized"] > d["drop_random"]).mean()),
                                    "p": float(p), "n": len(d)})
                except Exception:
                    pass
    if abl_a is not None and len(abl_a):
        for c in ["rsa_intact", "rsa_circuit_ablated", "rsa_random_ablated"]:
            if c in abl_a:
                summary.append({"claim": "R6_causal_alignment", "stat": f"mean_{c}",
                                "value": float(abl_a[c].mean()), "p": np.nan, "n": len(abl_a)})

    # ---- R2b: behaviour ~ mechanistic and behaviour ~ alignment (T2.2) ----
    if behav is not None and len(behav):
        bstep = behav.groupby("step", as_index=False)["mp_accuracy"].mean()
        if mech is not None and len(mech):
            bm = bstep.merge(mech, on="step", how="inner")
            for metric in METRICS:
                if metric in bm and not bm[metric].isna().all() and len(bm) >= 4:
                    r, p = spearmanr(bm["mp_accuracy"], bm[metric])
                    summary.append({"claim": "R2b_behaviour_vs_mechanistic",
                                    "stat": f"spearman(acc,{metric})", "value": r, "p": p, "n": len(bm)})
        if align is not None and len(align):
            astep = align.groupby("step", as_index=False)["rsa"].mean()
            ba = bstep.merge(astep, on="step", how="inner")
            if len(ba) >= 4:
                r, p = spearmanr(ba["mp_accuracy"], ba["rsa"])
                summary.append({"claim": "R2b_behaviour_vs_alignment", "stat": "spearman(acc,rsa)",
                                "value": r, "p": p, "n": len(ba)})

    # ---- write summary -----------------------------------------------------
    sdf = pd.DataFrame(summary)
    spath = out / f"devai_summary_{fam}.csv"
    sdf.to_csv(spath, index=False)
    print(f"Saved {spath}")
    if len(sdf):
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(sdf.to_string(index=False))
    else:
        print("No claims computed (missing inputs?).")


if __name__ == "__main__":
    main()
