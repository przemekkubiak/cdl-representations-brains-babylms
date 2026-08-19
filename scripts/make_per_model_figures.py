#!/usr/bin/env python
"""One figure per model: the three result axes against training step, on a shared x-axis.

The point is to make a model's directory readable at a glance -- does its brain alignment
move over training, does its representation sparsify, does its localisation sharpen. The
brain panel is drawn with an explicit warning band because those numbers are confounded by
scanner run (see the repo README); it is plotted so the reader can see it is flat and near
zero, not so it can be quoted.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "hf_results_staging"

PANELS = [
    ("brain_rsa_mean", "brain: mean RSA vs fMRI\n(CONFOUNDED - see README)", "#c0392b"),
    ("interp_per", "interp: participation ratio", "#2c7fb8"),
    ("interp_gini", "interp: gini (sparsity)", "#2c7fb8"),
    ("loc_selectivity", "localisation: selectivity index", "#31a354"),
    ("loc_n_active_layers", "localisation: active layers", "#31a354"),
    ("behav_mp_accuracy", "behaviour: minimal-pair acc", "#756bb1"),
]


def main() -> None:
    for fam_dir in sorted((STAGE / "by-model").iterdir()):
        f = fam_dir / "checkpoints.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).sort_values("step")
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
        for ax, (col, title, colour) in zip(axes.ravel(), PANELS):
            if col not in d or d[col].notna().sum() == 0:
                ax.text(.5, .5, "not available", ha="center", va="center",
                        transform=ax.transAxes, color="#888", fontsize=9)
                ax.set_title(title, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.plot(d["step"], d[col], "o-", ms=3, lw=1.2, color=colour)
            if col == "brain_rsa_mean":
                ax.axhline(0, color="#444", lw=.8, ls="--")
                ax.axhspan(-0.02, 0.02, color="#c0392b", alpha=.08)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("training step", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.25, lw=.5)
        fig.suptitle(f"{fam_dir.name} — {d['step'].nunique()} checkpoints", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .96))
        out = fam_dir / "figures" / f"{fam_dir.name}_overview.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out.relative_to(STAGE))


if __name__ == "__main__":
    main()
