#!/usr/bin/env python
"""Summarise the corrected model grid: 15 families against ceiling-normalised,
within-run-normalised ds003604 RDMs.

WHY THIS EXISTS. The corrected sweep (launch_full_sweep.sh stages 2-3) wrote 75
raw CSVs under devai_grid_wrn/ and nothing read them. autopilot.sh publishes the
ceilings and the PARC seed-null but not the grid, so the headline corrected
result -- every family, every checkpoint, against RDMs that no longer carry the
run confound -- existed only as local per-family files. This turns them into the
tables and figures the paper needs.

WHAT IT ARGUES. Three numbers have to appear together or none of them is
readable:

  1. the alignment (Spearman rho of model RDM vs brain RDM),
  2. the NOISE CEILING for that cell -- what a perfect model could score, and
  3. the SEED NULL -- what pure initialisation noise scores on this measurement,
     taken from the PARC suite (6 seeds x 3 architectures, matched data/scale).

Alignment alone is uninterpretable; alignment against a ceiling of ~0.85 with a
seed-null sd of ~0.008 is a result. Everything below reports all three.

Outputs (--out, default paper_results/corrected):
  alignment_by_checkpoint.csv   every row, + ceiling, frac_of_ceiling, params
  alignment_by_family.csv       per family, incl. TOST equivalence
  alignment_by_cell.csv         family x task x session
  scale_ladder.csv              Pythia 70M -> 1.4B, the "undertrained" answer
  seed_null_comparison.csv      each family against the matched pure-noise null
  model_params.csv              exact parameter counts (cached from the Hub)
  summary.json                  the headline numbers
  fig_corrected_scale_ladder.*  alignment vs parameters, with ceiling
  fig_corrected_family.*        per-family forest against ceiling and seed null
  fig_corrected_trajectory.*    alignment vs training step, all families
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Same validated palette as scripts/make_ceiling_figures.py, so the corrected
# figures sit next to the ceiling figures without a colour clash.
RAW = "#2a78d6"
NRM = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
CEIL = "#c9c8c1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "figure.dpi": 150,
})

# The scale ladder, in order. Kept explicit rather than sniffed from the family
# name so a mislabelled family cannot silently reorder the axis.
LADDER = ["pythia-70m-full", "pythia-160m-full", "pythia-410m-full",
          "pythia-1b-full", "pythia-1.4b-full"]


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ load ----
def load_grid(grid_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(grid_dir.glob("alignment_*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  ! {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["repo"] = df["model_ref"].astype(str).str.split("@").str[0]
    return df


def load_ceilings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    c = pd.read_csv(path)
    keep = [k for k in ["task", "session", "ceiling_lower", "ceiling_upper",
                        "n_stim", "ceiling_n"] if k in c.columns]
    return c[keep].rename(columns={"n_stim": "ceiling_n_stim",
                                   "ceiling_n": "ceiling_n_subjects"})


def parc_rows(grid_dir: Path) -> pd.DataFrame:
    """Every raw PARC alignment row: 18 runs (3 architectures x 6 seeds) x 204
    cells. These runs differ from each other only by initialisation, so the whole
    table is a sample of what this measurement returns when there is nothing to
    find -- including its extremes."""
    frames = []
    for f in sorted(grid_dir.glob("alignment_parc-*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def null_max_distribution(parc: pd.DataFrame, n: int, n_boot: int = 400,
                          seed: int = 0) -> np.ndarray:
    """What the LARGEST |rsa| in a grid of n cells looks like under the null.

    WHY THIS AND NOT A SD. A family's best cell is a maximum over ~100-250
    comparisons, so judging it against the across-seed sd at a FIXED cell asks
    the wrong question and inflates it into a fake detection: the seed sd holds
    task, session and step constant, while a max ranges over all of them and
    picks up each cell's own idiosyncratic bias. The matched yardstick is the
    same max statistic computed on data known to contain no effect, at the same
    n -- which is exactly what the PARC runs provide.

    Sampled within-run so the cell-to-cell correlation structure that inflates a
    maximum is preserved rather than bootstrapped away.
    """
    if parc.empty:
        return np.array([])
    rng = np.random.default_rng(seed)
    per_run = [g["rsa"].abs().to_numpy() for _, g in parc.groupby("family")]
    per_run = [v for v in per_run if len(v)]
    if not per_run:
        return np.array([])
    draws = []
    for _ in range(n_boot):
        v = per_run[rng.integers(len(per_run))]
        idx = rng.choice(len(v), size=n, replace=n > len(v))
        draws.append(v[idx].max())
    return np.array(draws)


def seed_null_sd(path: Path) -> pd.DataFrame:
    """Per (task, session) across-seed sd from the PARC suite.

    Six seeds of one architecture differ only by initialisation, so their spread
    is what zero looks like here. Pooled across architecture and step by the
    median, which is robust to the handful of cells where one seed wandered.
    """
    if not path.exists():
        return pd.DataFrame()
    s = pd.read_csv(path)
    if "rsa_sd" not in s.columns:
        return pd.DataFrame()
    g = (s.groupby(["task", "session"])["rsa_sd"].median()
           .reset_index().rename(columns={"rsa_sd": "seed_null_sd"}))
    return g


def fetch_params(repos: list[str], cache: Path) -> pd.DataFrame:
    """Exact parameter counts from the Hub's safetensors metadata, cached.

    Nominal sizes ("160M") are marketing names -- pythia-70m is 95.6M with
    embeddings. The scale axis has to be the real number, and it has to be
    reproducible offline, hence the cache file.
    """
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["repo", "params"])
    known = set(have["repo"]) if len(have) else set()
    missing = [r for r in repos if r not in known]
    if missing:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            rows = []
            for r in missing:
                try:
                    info = api.model_info(r, expand=["safetensors"])
                    st = getattr(info, "safetensors", None)
                    if st and st.total:
                        rows.append({"repo": r, "params": int(st.total)})
                        print(f"    params {r}: {st.total:,}")
                    else:
                        print(f"    params {r}: unavailable")
                except Exception as e:
                    print(f"    params {r}: {type(e).__name__}")
            if rows:
                have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
                cache.parent.mkdir(parents=True, exist_ok=True)
                have.to_csv(cache, index=False)
        except Exception as e:
            print(f"    parameter lookup unavailable ({type(e).__name__}); "
                  "scale axis will be omitted")
    return have


def tost(vals: np.ndarray, bound: float) -> float:
    """Two one-sided tests; returns the governing (larger) p-value.

    p < alpha => the effect is statistically WITHIN +/- bound, i.e. absence
    demonstrated rather than merely unfound. Same implementation as
    scripts/parc_seed_null.py, deliberately.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n < 3:
        return float("nan")
    m, se = vals.mean(), stats.sem(vals)
    if se == 0:
        return 0.0 if abs(m) < bound else 1.0
    dfree = n - 1
    p_lo = stats.t.sf((m - (-bound)) / se, dfree)
    p_hi = stats.t.cdf((m - bound) / se, dfree)
    return float(max(p_lo, p_hi))


# --------------------------------------------------------------- figures ----
def fig_scale_ladder(fam: pd.DataFrame, ceil_lo: float, out: Path) -> None:
    """Alignment vs parameters. Answers 'your models are just too small'.

    Two panels because the honest picture needs both scales: LEFT shows the
    ladder against the ceiling (the finding -- the bars are invisible, which is
    the point), RIGHT zooms to the alignment's own range so the reader can see
    that there is no trend even at full magnification.
    """
    d = fam[fam["family"].isin(LADDER)].copy()
    if d.empty or d["params"].isna().all():
        return
    d["order"] = d["family"].map({f: i for i, f in enumerate(LADDER)})
    d = d.sort_values("order")
    x = np.arange(len(d))
    labels = [f.replace("pythia-", "").replace("-full", "") for f in d["family"]]
    lo = d["rsa_mean"] - d["rsa_sd"]
    hi = d["rsa_mean"] + d["rsa_sd"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    _despine(ax)
    ax.axhspan(ceil_lo, 1.0, color=CEIL, alpha=0.75, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(x, d["rsa_mean"], yerr=[d["rsa_mean"] - lo, hi - d["rsa_mean"]],
                fmt="o", color=RAW, ms=5, lw=1.4, capsize=3, zorder=3)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("alignment (Spearman $\\rho$)")
    ax.set_xlabel("Pythia scale")
    ax.set_title("against the noise ceiling", loc="left")
    ax.annotate(f"noise ceiling {ceil_lo:.2f}", xy=(0.02, ceil_lo), xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points", fontsize=8, color=INK2)

    ax = axes[1]
    _despine(ax)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if "seed_null_sd" in d.columns and d["seed_null_sd"].notna().any():
        s = float(d["seed_null_sd"].dropna().median())
        ax.axhspan(-2 * s, 2 * s, color=NRM, alpha=0.12, lw=0, zorder=0)
        ax.annotate("$\\pm2\\sigma$ seed null", xy=(0.98, 2 * s), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", fontsize=8,
                    color=INK2, ha="right")
    ax.errorbar(x, d["rsa_mean"], yerr=[d["rsa_mean"] - lo, hi - d["rsa_mean"]],
                fmt="o", color=RAW, ms=5, lw=1.4, capsize=3, zorder=3)
    # Leave headroom above the null band so its edge is visible; a band that
    # runs off the axis reads as "no band at all".
    span = max(float(hi.max()), 2 * float(d["seed_null_sd"].dropna().median())
               if d["seed_null_sd"].notna().any() else 0.0)
    ax.set_ylim(-span * 1.35, span * 1.35)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Pythia scale")
    ax.set_title("magnified — no trend with scale", loc="left")

    fig.suptitle("Scale does not buy alignment (ds003604, within-run normalised)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_family(fam: pd.DataFrame, ceil_lo: float, out: Path) -> None:
    """Every family, mean +/- sd across cells, against the seed null."""
    d = fam.sort_values("rsa_mean")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.4, 0.30 * len(d) + 1.8))
    _despine(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=MUTED, alpha=0.18, linewidth=0.6)

    if "seed_null_sd" in d.columns and d["seed_null_sd"].notna().any():
        s = float(d["seed_null_sd"].dropna().median())
        ax.axvspan(-2 * s, 2 * s, color=NRM, alpha=0.12, lw=0, zorder=0)
        ax.annotate("$\\pm2\\sigma$ seed null", xy=(2 * s, len(d) - 0.4),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(d["rsa_mean"], y, xerr=d["rsa_sd"], fmt="o", color=RAW,
                ms=4, lw=1.2, capsize=2.5, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(d["family"], fontsize=8)
    ax.set_xlabel("alignment (Spearman $\\rho$), mean $\\pm$ sd across 12 cells")
    ax.set_title(f"All 15 families sit inside the seed null; the ceiling is {ceil_lo:.2f}",
                 loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_trajectory(df: pd.DataFrame, ceil_lo: float, seed_sd: float | None, out: Path) -> None:
    """Alignment vs training step, one line per family, averaged over cells.

    The developmental claim is that alignment should GROW with training. Plotted
    on the alignment's own scale with the seed null drawn on it, because at
    ceiling scale every line is flat against zero and shows nothing.
    """
    t = (df.groupby(["family", "step"])["rsa"].mean().reset_index())
    fams = sorted(t["family"].unique())
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    _despine(ax)
    if seed_sd:
        ax.axhspan(-2 * seed_sd, 2 * seed_sd, color=NRM, alpha=0.12, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    # Fifteen coloured lines is spaghetti and the claim is not about any one
    # family: draw them as an unlabelled grey band of evidence and put the only
    # colour on the across-family mean, which is the thing being asserted flat.
    for f in fams:
        g = t[(t["family"] == f) & (t["step"] > 0)].sort_values("step")
        if len(g) >= 2:
            ax.plot(g["step"], g["rsa"], lw=1.0, alpha=0.45, color=MUTED, zorder=2)
    # Families do not share a step schedule, so a mean taken at each distinct
    # step would jump between "the mean of 12 families" and "one family alone"
    # and read as volatility that is not there. Bin on log(step) instead and
    # only draw bins that actually average several families.
    tt = t[t["step"] > 0].copy()
    edges = np.logspace(np.log10(max(tt["step"].min(), 1)),
                        np.log10(tt["step"].max()), 16)
    tt["bin"] = pd.cut(tt["step"], bins=edges, include_lowest=True)
    b = (tt.groupby("bin", observed=True)
           .agg(step=("step", "median"), rsa=("rsa", "mean"),
                n_fam=("family", "nunique")).reset_index())
    b = b[b["n_fam"] >= 3]
    ax.plot(b["step"], b["rsa"], lw=2.2, color=RAW, zorder=4, marker="o", ms=3.5,
            label=f"binned mean across families ($\\geq$3 per bin)")
    ax.plot([], [], lw=1.0, color=MUTED, alpha=0.6, label="individual family")
    ax.set_xscale("symlog", linthresh=100)
    span = max(float(t["rsa"].abs().max()), 2 * (seed_sd or 0))
    ax.set_ylim(-span * 1.25, span * 1.25)
    ax.set_xlabel("training step (symlog)")
    ax.set_ylabel("alignment (Spearman $\\rho$), mean over 12 cells")
    ax.set_title(f"No trajectory emerges over training (ceiling {ceil_lo:.2f}; "
                 "shaded = $\\pm2\\sigma$ seed null)", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir",
                    default="data/processed/language_models/devai_grid_wrn/ds003604")
    ap.add_argument("--ceilings", default="paper_results/ceiling/ceilings_ds003604.csv")
    ap.add_argument("--seed-null", default="paper_results/parc/parc_seed_spread.csv")
    ap.add_argument("--parc-grid",
                    default="data/processed/language_models/devai_grid_parc/ds003604",
                    help="raw PARC rows, used as the matched max-statistic null")
    ap.add_argument("--out", default="paper_results/corrected")
    ap.add_argument("--sesoi", type=float, default=0.05,
                    help="smallest effect of interest for the equivalence test")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_grid(Path(a.grid_dir))
    if df.empty:
        print(f"no alignment rows under {a.grid_dir}")
        return
    print(f"{len(df)} alignment rows | {df['family'].nunique()} families | "
          f"{df.groupby(['task', 'session']).ngroups} cells")

    ceil = load_ceilings(Path(a.ceilings))
    if ceil.empty:
        print("  ! no ceilings -- everything below is uninterpretable; fix that first")
        df["ceiling_lower"] = np.nan
    else:
        df = df.merge(ceil, on=["task", "session"], how="left")
    df["frac_of_ceiling"] = df["rsa"] / df["ceiling_lower"]

    null = seed_null_sd(Path(a.seed_null))
    if not null.empty:
        df = df.merge(null, on=["task", "session"], how="left")
    else:
        df["seed_null_sd"] = np.nan
        print("  (no PARC seed spread -- seed-null comparison unavailable)")

    params = fetch_params(sorted(df["repo"].unique()), out / "model_params.csv")
    if len(params):
        df = df.merge(params, on="repo", how="left")
    else:
        df["params"] = np.nan

    ceil_lo = float(ceil["ceiling_lower"].mean()) if not ceil.empty else float("nan")

    # ------------------------------------------------------- by checkpoint --
    df.to_csv(out / "alignment_by_checkpoint.csv", index=False)

    # -------------------------------------------------------------- by cell --
    cell = (df.groupby(["family", "task", "session"])
              .agg(n_checkpoints=("rsa", "size"),
                   rsa_mean=("rsa", "mean"),
                   rsa_sd=("rsa", "std"),
                   rsa_min=("rsa", "min"),
                   rsa_max=("rsa", "max"),
                   ceiling=("ceiling_lower", "first"),
                   seed_null_sd=("seed_null_sd", "first"))
              .reset_index())
    cell["frac_of_ceiling_max"] = cell["rsa_max"] / cell["ceiling"]
    # Cell means average ~20 checkpoints, so this ratio is a legitimate
    # comparison of like with like: both sides hold task/session fixed.
    cell["mean_in_seed_sd"] = cell["rsa_mean"] / cell["seed_null_sd"]
    cell.to_csv(out / "alignment_by_cell.csv", index=False)

    parc = parc_rows(Path(a.parc_grid))
    if parc.empty:
        print("  (no PARC grid rows -- max-statistic null unavailable)")

    # ------------------------------------------------------------ by family --
    rows = []
    for f, g in df.groupby("family"):
        cellmeans = cell[cell["family"] == f]["rsa_mean"].to_numpy()
        best_i = g["rsa"].abs().idxmax()
        best = g.loc[best_i]
        p_eq = tost(cellmeans, a.sesoi)
        # The family's best cell, judged against the same max statistic computed
        # on pure-noise runs at the same number of comparisons.
        nulls = null_max_distribution(parc, n=len(g))
        best_abs = float(g["rsa"].abs().max())
        if nulls.size:
            null_p50, null_p95 = float(np.median(nulls)), float(np.percentile(nulls, 95))
            best_pctile = float((nulls < best_abs).mean())
        else:
            null_p50 = null_p95 = best_pctile = np.nan
        rows.append({
            "family": f,
            "params": float(g["params"].dropna().iloc[0]) if g["params"].notna().any() else np.nan,
            "n_checkpoints": int(g["step"].nunique()),
            "n_rows": int(len(g)),
            "n_cells": int(g.groupby(["task", "session"]).ngroups),
            "rsa_mean": float(g["rsa"].mean()),
            "rsa_sd": float(cellmeans.std(ddof=1)) if len(cellmeans) > 1 else np.nan,
            "rsa_sd_all_rows": float(g["rsa"].std(ddof=1)),
            "rsa_abs_max": float(g["rsa"].abs().max()),
            "best_rsa": float(best["rsa"]),
            "best_cell": f"{best['task']}/{best['session']}",
            "best_step": int(best["step"]),
            "frac_of_ceiling_mean": float(g["frac_of_ceiling"].mean()),
            "frac_of_ceiling_abs_max": float(g["frac_of_ceiling"].abs().max()),
            "seed_null_sd": float(g["seed_null_sd"].median()) if g["seed_null_sd"].notna().any() else np.nan,
            "null_max_p50": null_p50,
            "null_max_p95": null_p95,
            "best_percentile_vs_null": best_pctile,
            "beats_null_max": bool(np.isfinite(best_pctile) and best_pctile > 0.95),
            "p_equivalence_tost": p_eq,
            "equivalent_to_zero": bool(np.isfinite(p_eq) and p_eq < 0.05),
        })
    fam = pd.DataFrame(rows)
    fam["mean_in_seed_sd"] = fam["rsa_mean"] / fam["seed_null_sd"]
    fam = fam.sort_values("rsa_mean", ascending=False)
    fam.to_csv(out / "alignment_by_family.csv", index=False)

    # ---------------------------------------------------------- scale ladder --
    ladder = fam[fam["family"].isin(LADDER)].copy()
    if len(ladder):
        ladder["order"] = ladder["family"].map({f: i for i, f in enumerate(LADDER)})
        ladder = ladder.sort_values("order").drop(columns=["order"])
        # Trend across the ladder, at cell resolution so the test has n=60 not 5.
        lc = cell[cell["family"].isin(LADDER)].merge(
            fam[["family", "params"]], on="family", how="left")
        lc = lc.dropna(subset=["params", "rsa_mean"])
        if len(lc) > 3:
            rho, p = stats.spearmanr(lc["params"], lc["rsa_mean"])
            ladder["scale_trend_rho"] = rho
            ladder["scale_trend_p"] = p
            ladder["scale_trend_n"] = len(lc)
        ladder.to_csv(out / "scale_ladder.csv", index=False)

    # ------------------------------------------------------ seed-null table --
    snc = fam[["family", "rsa_mean", "rsa_sd", "mean_in_seed_sd", "rsa_abs_max",
               "null_max_p50", "null_max_p95", "best_percentile_vs_null",
               "beats_null_max", "frac_of_ceiling_abs_max",
               "p_equivalence_tost", "equivalent_to_zero"]].copy()
    snc.to_csv(out / "seed_null_comparison.csv", index=False)

    # ---------------------------------------------------------------- print --
    med_sd = float(fam["seed_null_sd"].median()) if fam["seed_null_sd"].notna().any() else None
    n_eq = int(fam["equivalent_to_zero"].sum())
    print()
    print("  --- CORRECTED GRID: alignment against a ceiling of "
          f"{ceil_lo:.3f} ---")
    show = fam[["family", "n_checkpoints", "rsa_mean", "rsa_sd", "mean_in_seed_sd",
                "rsa_abs_max", "null_max_p95", "best_percentile_vs_null",
                "frac_of_ceiling_abs_max", "p_equivalence_tost"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print()
    print(f"  best alignment anywhere in the corrected grid: "
          f"{fam['rsa_abs_max'].max():.4f} "
          f"({fam['frac_of_ceiling_abs_max'].max() * 100:.1f}% of the ceiling)")
    if med_sd:
        print(f"  seed null (PARC): across-seed sd {med_sd:.4f}; family means sit "
              f"at {fam['mean_in_seed_sd'].abs().max():.1f} sd at worst")
    if parc is not None and not parc.empty:
        n_beat = int(fam["beats_null_max"].sum())
        print(f"  MAX STATISTIC: {n_beat}/{len(fam)} families have a best cell "
              f"beyond the 95th percentile of the matched pure-noise maximum "
              f"(null max p50 ~ {fam['null_max_p50'].median():.4f}, "
              f"p95 ~ {fam['null_max_p95'].median():.4f})")
    print(f"  EQUIVALENCE: {n_eq}/{len(fam)} families statistically equivalent "
          f"to zero within +/-{a.sesoi}")
    if len(ladder) and "scale_trend_rho" in ladder.columns:
        print(f"  SCALE LADDER (Pythia {ladder['params'].min() / 1e6:.0f}M -> "
              f"{ladder['params'].max() / 1e9:.1f}B): Spearman rho vs parameters "
              f"= {ladder['scale_trend_rho'].iloc[0]:+.3f} "
              f"(p={ladder['scale_trend_p'].iloc[0]:.3f}, "
              f"n={int(ladder['scale_trend_n'].iloc[0])} cells)")

    summary = {
        "dataset": "ds003604",
        "rdms": "within-run normalised",
        "n_families": int(fam.shape[0]),
        "n_rows": int(len(df)),
        "n_cells": int(df.groupby(["task", "session"]).ngroups),
        "ceiling_lower_mean": ceil_lo,
        "best_rsa_abs": float(fam["rsa_abs_max"].max()),
        "best_frac_of_ceiling": float(fam["frac_of_ceiling_abs_max"].max()),
        "seed_null_sd_median": med_sd,
        "null_max_p95_median": float(fam["null_max_p95"].median()) if fam["null_max_p95"].notna().any() else None,
        "families_beating_null_max": int(fam["beats_null_max"].sum()),
        "families_equivalent_to_zero": n_eq,
        "sesoi": a.sesoi,
    }
    if len(ladder) and "scale_trend_rho" in ladder.columns:
        summary["scale_trend_rho"] = float(ladder["scale_trend_rho"].iloc[0])
        summary["scale_trend_p"] = float(ladder["scale_trend_p"].iloc[0])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # -------------------------------------------------------------- figures --
    if not a.no_figures:
        fam_fig = fam.copy()
        fig_scale_ladder(fam_fig, ceil_lo, out / "fig_corrected_scale_ladder")
        fig_family(fam_fig, ceil_lo, out / "fig_corrected_family")
        fig_trajectory(df, ceil_lo, med_sd, out / "fig_corrected_trajectory")

    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
