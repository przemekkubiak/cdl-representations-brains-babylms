#!/usr/bin/env python
"""CoDLA: Co-Developmental Localization-Alignment — join the three axes and test.

Ties together, along the shared developmental axis (LM checkpoint step ~ child age):
  L_LM(P,t)   LM circuit localization        <- localization_trajectory_<family>.csv
  A(P,t)      brain-LM RSA alignment         <- checkpoint_alignment_trajectory*.csv
  L_brain(P)  brain ROI specialization       <- optional --brain-specialization CSV

and evaluates the four claims (see PRIVATE_NOTES.md §6):
  C1  LM develops localization:      Spearman(step, gini) > 0 per phenomenon
  C2  localization tracks alignment: Spearman(gini, RSA) > 0, and the partial
                                     correlation controlling for step stays > 0
                                     (specialization, not just "trained longer")
  C3  developmental correspondence:  rank order of per-P "time-to-specialize"
                                     vs brain onset order (if brain data given)
  C4  differentiation:               Spearman(step, cross-phenomenon overlap) < 0

Inputs
  --localization  localization_trajectory_<family>.csv  (from run_circuit_localization.py)
  --alignment     repeatable PHEN=path.csv  (per-phenomenon checkpoint alignment CSV;
                  each produced by checkpoint_alignment_trajectory.py --task PHEN)
  --brain-specialization  optional CSV: columns [phenomenon, brain_localization]
                          (+ optional onset_age for C3 ordering)

Outputs
  codla_summary_<family>.csv   per-phenomenon C1-C4 statistics + verdicts
  fig_codla_<family>.png       gini↑, alignment~gini, overlap↓ panels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-residualized)."""
    x, y, z = map(lambda a: rankdata(np.asarray(a, float)), (x, y, z))

    def resid(a, b):
        A = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef

    rx, ry = resid(x, z), resid(y, z)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(rx, ry)
    return float(r), float(p)


def time_to_specialize(sub: pd.DataFrame, metric: str = "gini", frac: float = 0.5) -> float:
    """First step at which `metric` reaches min + frac*(max-min) — a developmental
    'onset' of specialization. NaN if it never rises."""
    sub = sub.sort_values("step")
    v = pd.to_numeric(sub[metric], errors="coerce").to_numpy()
    s = sub["step"].to_numpy(dtype=float)
    if not np.isfinite(v).any() or np.nanmax(v) == np.nanmin(v):
        return float("nan")
    thr = np.nanmin(v) + frac * (np.nanmax(v) - np.nanmin(v))
    hit = np.where(v >= thr)[0]
    return float(s[hit[0]]) if hit.size else float("nan")


def load_alignment(specs: list[str]) -> pd.DataFrame:
    """Parse --alignment PHEN=path.csv args into long df [phenomenon, step, RSA]."""
    rows = []
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--alignment expects PHEN=path.csv, got '{spec}'")
        phen, path = spec.split("=", 1)
        df = pd.read_csv(path)
        # average RSA across brain sessions at each step (per-session kept in raw file)
        g = (df.groupby("step", as_index=False)["correlation"].mean()
               .rename(columns={"correlation": "RSA"}))
        g["phenomenon"] = phen
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--localization", required=True, help="localization_trajectory_<family>.csv")
    ap.add_argument("--alignment", nargs="*", default=[], help="PHEN=path.csv (repeatable)")
    ap.add_argument("--brain-specialization", default=None, help="CSV [phenomenon, brain_localization, onset_age?]")
    ap.add_argument("--family", default="model")
    ap.add_argument("--output-dir", default="data/processed/language_models/codla")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loc = pd.read_csv(args.localization)
    loc["step"] = pd.to_numeric(loc["step"], errors="coerce")
    align = load_alignment(args.alignment)
    brain = pd.read_csv(args.brain_specialization) if args.brain_specialization else None

    summary = []
    for phen, sub in loc.groupby("phenomenon"):
        sub = sub.sort_values("step")
        row = {"phenomenon": phen, "n_checkpoints": len(sub)}

        # C1: localization develops
        rho1, p1 = spearmanr(sub["step"], sub["gini"])
        row.update(C1_rho=rho1, C1_p=p1, C1_pass=bool(rho1 > 0 and p1 < 0.05))
        row["onset_step_gini"] = time_to_specialize(sub, "gini")

        # C4: differentiation (overlap falls)
        if "mean_overlap_with_others" in sub:
            rho4, p4 = spearmanr(sub["step"], sub["mean_overlap_with_others"])
            row.update(C4_rho=rho4, C4_p=p4, C4_pass=bool(rho4 < 0 and p4 < 0.05))

        # C2: localization tracks alignment (needs alignment for this phenomenon)
        if not align.empty and phen in set(align["phenomenon"]):
            a = align[align["phenomenon"] == phen][["step", "RSA"]]
            m = sub.merge(a, on="step", how="inner")
            if len(m) >= 4:
                rho2, p2 = spearmanr(m["gini"], m["RSA"])
                prho, pp = partial_spearman(m["gini"], m["RSA"], m["step"])
                row.update(
                    C2_rho=rho2, C2_p=p2, C2_partial_rho=prho, C2_partial_p=pp,
                    C2_pass=bool(rho2 > 0 and prho > 0),
                    n_aligned=len(m),
                )
        summary.append(row)

    summ = pd.DataFrame(summary)

    # C3: developmental correspondence (LM onset order vs brain onset order)
    c3_note = "no brain onset provided"
    if brain is not None and "onset_age" in brain.columns:
        merged = summ.merge(brain[["phenomenon", "onset_age"]], on="phenomenon", how="inner")
        merged = merged.dropna(subset=["onset_step_gini", "onset_age"])
        if len(merged) >= 3:
            rho3, p3 = spearmanr(merged["onset_step_gini"], merged["onset_age"])
            c3_note = f"Spearman(LM onset step, brain onset age) rho={rho3:.3f} p={p3:.3f} (n={len(merged)})"

    csv_path = out / f"codla_summary_{args.family}.csv"
    summ.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print("\n=== CoDLA verdicts ===")
    show = [c for c in ["phenomenon", "C1_rho", "C1_pass", "C2_rho", "C2_partial_rho",
                        "C2_pass", "C4_rho", "C4_pass", "onset_step_gini"] if c in summ]
    print(summ[show].to_string(index=False))
    print(f"\nC3 (developmental correspondence): {c3_note}")

    _plot(loc, align, out / f"fig_codla_{args.family}.png", args.family)


def _plot(loc: pd.DataFrame, align: pd.DataFrame, path: Path, family: str) -> None:
    n = 3 if not align.empty else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    axes = np.atleast_1d(axes)
    for phen, sub in loc.groupby("phenomenon"):
        sub = sub.sort_values("step")
        axes[0].plot(sub["step"], sub["gini"], marker="o", label=phen)
        if "mean_overlap_with_others" in sub:
            axes[1].plot(sub["step"], sub["mean_overlap_with_others"], marker="o", label=phen)
    axes[0].set(xlabel="step", ylabel="Gini (localization)", title="C1: specialization ↑")
    axes[1].set(xlabel="step", ylabel="mean cross-phenomenon overlap", title="C4: differentiation ↓")
    if not align.empty:
        for phen, sub in loc.groupby("phenomenon"):
            a = align[align["phenomenon"] == phen][["step", "RSA"]]
            m = sub.merge(a, on="step", how="inner")
            if len(m):
                axes[2].scatter(m["gini"], m["RSA"], label=phen)
        axes[2].set(xlabel="Gini (localization)", ylabel="brain-LM RSA", title="C2: localization ↔ alignment")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"CoDLA — {family}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
