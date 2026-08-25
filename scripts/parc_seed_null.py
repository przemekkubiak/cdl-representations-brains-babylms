#!/usr/bin/env python
"""Seed-null analysis of the PARC suite: is any alignment bigger than the noise
that random initialisation alone produces?

THE ARGUMENT. Our result is a null, and a null needs a yardstick. Six seeds of
the same architecture, trained on the same data for the same number of steps,
differ ONLY by initialisation -- so the spread of their alignments is what "no
effect" looks like on this measurement. An alignment is only real if it clears
that spread. This is a far stronger claim than a p-value against zero, and it is
the equivalence test TODO.md section 2 asks for.

Three things are computed per (architecture, task, session, step):

  1. SEED SPREAD -- mean and sd of alignment across the six seeds, and a
     one-sample t-test of the seed mean against zero. With n=6 this is honest
     about its own power; the sd is the number that matters.
  2. EQUIVALENCE (TOST) -- is the effect statistically smaller than a
     pre-specified smallest effect of interest? Rejecting both one-sided tests
     lets you assert absence rather than merely fail to find presence.
  3. ARCHITECTURE CONTRAST -- transformer vs state-space vs RNN-like, at matched
     data, scale and step. Whether brain-LM correspondence resembles recurrence
     or signal propagation, measured rather than asserted.

Where a noise ceiling is available, everything is also expressed as a fraction of
it, because an unnormalised RSA is not interpretable.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAM_RE = re.compile(r"parc-(?P<arch>pythia|mamba|rwkv)-seed(?P<seed>\d+)")
ARCH_LABEL = {
    "pythia": "transformer (Pythia 160M)",
    "mamba": "state-space (Mamba 130M)",
    "rwkv": "RNN-like (RWKV 169M)",
}


def load_grid(grid_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(grid_dir.glob("alignment_parc-*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  ! {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    meta = df["family"].str.extract(FAM_RE)
    df["arch"] = meta["arch"]
    df["seed"] = pd.to_numeric(meta["seed"], errors="coerce")
    return df.dropna(subset=["arch", "seed"])


def load_ceilings(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    c = pd.read_csv(path)
    if "ceiling_lower" not in c.columns:
        return {}
    return {(r["task"], r["session"]): r["ceiling_lower"]
            for _, r in c.iterrows() if pd.notna(r.get("ceiling_lower"))}


def tost(vals: np.ndarray, bound: float) -> float:
    """Two one-sided tests. Returns the larger (governing) p-value.

    p < alpha means the effect is statistically WITHIN +/- bound, i.e. absence is
    demonstrated rather than merely un-refuted.
    """
    n = len(vals)
    if n < 3:
        return np.nan
    m, se = vals.mean(), stats.sem(vals)
    if se == 0:
        return 0.0 if abs(m) < bound else 1.0
    dfree = n - 1
    p_lo = stats.t.sf((m - (-bound)) / se, dfree)     # H0: mean <= -bound
    p_hi = stats.t.cdf((m - bound) / se, dfree)       # H0: mean >= +bound
    return float(max(p_lo, p_hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--ceilings", default="paper_results/ceiling/ceilings_ds003604.csv")
    ap.add_argument("--out", default="paper_results/parc")
    ap.add_argument("--sesoi", type=float, default=0.05,
                    help="smallest effect of interest for the equivalence test")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_grid(Path(a.grid_dir))
    if df.empty:
        print(f"no PARC alignment rows under {a.grid_dir}")
        return
    print(f"{len(df)} alignment rows | architectures {sorted(df['arch'].unique())} "
          f"| seeds {sorted(df['seed'].unique().astype(int))}")

    ceil = load_ceilings(Path(a.ceilings))
    if ceil:
        df["ceiling"] = [ceil.get((t, s), np.nan) for t, s in zip(df["task"], df["session"])]
        df["frac_of_ceiling"] = df["rsa"] / df["ceiling"]
    else:
        df["ceiling"] = np.nan
        df["frac_of_ceiling"] = np.nan
        print("  (no ceilings found -- fractions unavailable)")

    # ---- 1 & 2: seed spread, t vs zero, equivalence ----------------------
    keys = ["arch", "task", "session", "step"]
    rows = []
    for k, g in df.groupby(keys):
        v = g["rsa"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            continue
        t, p = stats.ttest_1samp(v, 0.0)
        rows.append({
            **dict(zip(keys, k)),
            "n_seeds": len(v),
            "rsa_mean": v.mean(), "rsa_sd": v.std(ddof=1),
            "rsa_min": v.min(), "rsa_max": v.max(),
            "t_vs_zero": t, "p_vs_zero": p,
            "p_equivalence_tost": tost(v, a.sesoi),
            "ceiling": g["ceiling"].iloc[0],
            "frac_of_ceiling_mean": g["frac_of_ceiling"].mean(),
        })
    seed_df = pd.DataFrame(rows)
    if seed_df.empty:
        print("no cell had >= 3 seeds")
        return
    seed_df.to_csv(out / "parc_seed_spread.csv", index=False)

    print("\n--- SEED SPREAD: what 'no effect' looks like on this measurement ---")
    for arch, g in seed_df.groupby("arch"):
        print(f"  {ARCH_LABEL[arch]:32s} "
              f"mean rsa {g['rsa_mean'].mean():+.4f}  "
              f"typical across-seed sd {g['rsa_sd'].median():.4f}  "
              f"|mean| > 2sd in {100*(g['rsa_mean'].abs() > 2*g['rsa_sd']).mean():.1f}% of cells")

    print(f"\n--- EQUIVALENCE (TOST, SESOI = +/-{a.sesoi}) ---")
    eq = seed_df["p_equivalence_tost"] < 0.05
    print(f"  {int(eq.sum())}/{len(seed_df)} cells ({100*eq.mean():.1f}%) are statistically")
    print(f"  EQUIVALENT TO ZERO within +/-{a.sesoi} -- absence demonstrated, not just unfound.")

    # ---- 3: architecture contrast ----------------------------------------
    print("\n--- ARCHITECTURE CONTRAST (matched data, scale, steps) ---")
    arch_rows = []
    for (task, session, step), g in seed_df.groupby(["task", "session", "step"]):
        if g["arch"].nunique() < 2:
            continue
        groups = [df[(df["arch"] == ar) & (df["task"] == task) &
                     (df["session"] == session) & (df["step"] == step)]["rsa"].dropna()
                  for ar in sorted(g["arch"].unique())]
        groups = [x for x in groups if len(x) >= 3]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        arch_rows.append({"task": task, "session": session, "step": step,
                          "F": F, "p": p, "n_arch": len(groups)})
    arch_df = pd.DataFrame(arch_rows)
    if not arch_df.empty:
        arch_df.to_csv(out / "parc_architecture_contrast.csv", index=False)
        sig = (arch_df["p"] < 0.05).mean()
        print(f"  one-way ANOVA across architectures, per cell: "
              f"{100*sig:.1f}% of {len(arch_df)} cells reach p<0.05")
        print("  (at alpha=.05, ~5% is what chance looks like)")
        for arch, g in seed_df.groupby("arch"):
            print(f"    {ARCH_LABEL[arch]:32s} grand mean rsa {g['rsa_mean'].mean():+.4f}")

    summary = seed_df.groupby("arch").agg(
        rsa_mean=("rsa_mean", "mean"), seed_sd=("rsa_sd", "median"),
        frac_of_ceiling=("frac_of_ceiling_mean", "mean"), n_cells=("rsa_mean", "size"),
    ).reset_index()
    summary.to_csv(out / "parc_summary_by_architecture.csv", index=False)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
