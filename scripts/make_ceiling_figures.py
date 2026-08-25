#!/usr/bin/env python
"""Figures for the noise-ceiling / within-run-correction result.

Reads what scripts/ceiling_report.py wrote and produces presentation figures.

The central claim these have to carry: the raw RDM is mostly scanner run;
correcting it removes that; and once corrected, model alignment is a negligible
fraction of the noise ceiling. Every panel therefore shows the ceiling, because
an alignment number without one is unreadable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated 2-slot categorical palette (dataviz reference instance, light mode):
# worst adjacent CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface.
RAW = "#2a78d6"       # slot 1, blue
NRM = "#eb6834"       # slot 2, orange
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
CEIL = "#c9c8c1"      # neutral band -- not a series colour
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


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_layerwise(df: pd.DataFrame, summ: dict, out: Path) -> None:
    """The money figure.

    Two panels per model, on separate axes rather than one axis with two scales:
    LEFT shows the true vertical distance from the ceiling (the finding), RIGHT
    zooms to the alignment's own range so the layer profile is actually legible.
    The zoom band is drawn on the left panel so the reader can see what was
    magnified.
    """
    fams = sorted(df["family"].unique())
    lo_n = summ["ceiling_normalised"]["lower"]
    up_n = summ["ceiling_normalised"]["upper"]

    fig, axes = plt.subplots(
        len(fams), 2, figsize=(8.6, 3.3 * len(fams)), squeeze=False,
        gridspec_kw={"width_ratios": [1, 1.15]})

    for row, fam in enumerate(fams):
        g = df[df["family"] == fam].sort_values("layer")
        best = g.loc[g["rsa_normalised"].idxmax()]
        lo_y = min(g[["rsa_raw", "rsa_normalised"]].min().min(), 0) - 0.012
        hi_y = max(g[["rsa_raw", "rsa_normalised"]].max().max(), 0) + 0.012

        # ---------------- left: full scale, ceiling to zero ----------------
        ax = axes[row][0]
        _despine(ax)
        ax.axhspan(lo_n, up_n, color=CEIL, alpha=0.75, lw=0, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        # the magnified band
        ax.axhspan(lo_y, hi_y, color=NRM, alpha=0.10, lw=0, zorder=1)
        ax.plot(g["layer"], g["rsa_raw"], color=RAW, lw=2, zorder=3)
        ax.plot(g["layer"], g["rsa_normalised"], color=NRM, lw=2, zorder=4)
        ax.set_ylim(lo_y - 0.03, up_n + 0.06)
        ax.set_xlabel("layer")
        ax.set_ylabel("alignment (Spearman $\\rho$)")
        ax.set_title(f"{fam} — distance to ceiling", loc="left")
        ax.annotate(f"noise ceiling {lo_n:.2f}", xy=(0.5, up_n), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
        ax.annotate(f"best model layer\n{best['rsa_normalised']:+.3f}  ({100*best['frac_of_ceiling_norm']:.1f}% of ceiling)",
                    xy=(0.5, hi_y), xycoords=("axes fraction", "data"),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)

        # ---------------- right: zoomed layer profile ----------------
        ax = axes[row][1]
        _despine(ax)
        ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.plot(g["layer"], g["rsa_raw"], color=RAW, lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.2, label="raw RDM", zorder=3)
        ax.plot(g["layer"], g["rsa_normalised"], color=NRM, lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.2, label="within-run normalised", zorder=4)
        ax.set_ylim(lo_y, hi_y)
        ax.set_xlabel("layer")
        ax.set_ylabel("alignment (Spearman $\\rho$)")
        ax.set_title("zoomed — layer profile", loc="left")
        ax.margins(x=0.04)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 1.02))

    fig.suptitle(
        f"Model alignment is a negligible fraction of the noise ceiling  "
        f"({summ['task']} / {summ['session']}, n={summ['n_subjects']} subjects, "
        f"{summ['n_stim']} stimuli)",
        x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(fams) == 1 else 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_ceiling_layerwise.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_confound(summ: dict, out: Path) -> None:
    """Run confound before/after, and the ceiling before/after, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # -- panel 1: how much of the RDM is scanner run
    ax = axes[0]
    _despine(ax)
    vals = [summ["run_confound_raw"], summ["run_confound_normalised"]]
    bars = ax.bar([0, 1], vals, width=0.5, color=[RAW, NRM], zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks([0, 1], ["raw", "within-run\nnormalised"])
    ax.set_ylabel('"different run" predicts dissimilarity ($\\rho$)')
    ax.set_title("Scanner-run structure in the brain RDM", loc="left")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v >= 0 else "top",
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                    fontsize=9, color=INK)
    pad = max(0.08, abs(max(vals, key=abs)) * 0.25)
    ax.set_ylim(min(0, min(vals)) - pad, max(0, max(vals)) + pad)

    # -- panel 2: the ceiling itself
    ax = axes[1]
    _despine(ax)
    lo = [summ["ceiling_raw"]["lower"], summ["ceiling_normalised"]["lower"]]
    up = [summ["ceiling_raw"]["upper"], summ["ceiling_normalised"]["upper"]]
    for i, (l, u, c) in enumerate(zip(lo, up, (RAW, NRM))):
        ax.vlines(i, l, u, color=c, lw=8, alpha=0.35)
        ax.plot([i], [l], marker="o", ms=8, color=c, mec=SURFACE, mew=1.4, zorder=3)
        ax.annotate(f"{l:.3f}", (i, l), xytext=(10, -2), textcoords="offset points",
                    fontsize=9, color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks([0, 1], ["raw", "within-run\nnormalised"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("inter-subject reliability ($\\rho$)")
    ax.set_title("Noise ceiling (dot = leave-one-out lower bound)", loc="left")

    fig.suptitle(
        f"Within-run normalisation removes the run confound  "
        f"({summ['task']} / {summ['session']})",
        x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_ceiling_confound.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper_results/ceiling")
    ap.add_argument("--task", default="Phon")
    ap.add_argument("--session", default="ses-5")
    a = ap.parse_args()

    d = Path(a.dir)
    summ = json.load(open(d / f"summary_{a.task}_{a.session}.json"))
    fig_confound(summ, d)

    align = d / f"alignment_vs_ceiling_{a.task}_{a.session}.csv"
    if align.exists():
        df = pd.read_csv(align)
        if not df.empty:
            fig_layerwise(df, summ, d)

    print(f"figures -> {d}")
    for f in sorted(d.glob("fig_*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
