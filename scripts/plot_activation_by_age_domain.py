#!/usr/bin/env python
"""Figures: brain specialization by age group, per domain, across datasets.

WHAT THIS ACTUALLY SHOWS -- READ BEFORE INTERPRETING THE FIGURES.
These are NOT anatomical brain maps (no picture of a brain with colored
activation on it). `src/preprocessing/fmri_preprocessing.py` saves each
stimulus's pattern as a flat masked voxel vector (`masker.transform(...).
ravel()`); the mask's shape/affine needed to put that vector back into 3D
space is not saved alongside it, so a spatial map cannot be reconstructed
from what's on disk after the fact. What CAN be shown, correctly, is the
scalar specialization profile `src/rsa/brain_localization.py` already
computes per (dataset, phenomenon, age group): how concentrated the
condition>control response is (Gini), how selectively that region prefers
this phenomenon over the others (selectivity index), and how much its
top-selective voxels overlap with other phenomena's (lower = more
differentiated). That is real signal about brain organization by domain and
age -- just not a picture of the brain itself. Adding real spatial maps would
mean saving mask affine/shape alongside patterns, which is a preprocessing
change, not a plotting one -- flag it if that's what's actually wanted.

INPUT. Reads brain_localization_by_session.csv from every dataset's
localization output dir (produced by scripts/run_brain_localization.py,
called from prepare_brain_rdms.sh -- see that script's module docstring for
why it now runs per-session with --append rather than once at the end).
Session labels are already age-group bins for every dataset except ds003604,
where the BIDS session and the age-group bin happen to share a name (ses-5 ->
bin "5") -- so `session` and "age group" can be treated as the same axis
uniformly across all four without extra mapping.

OUTPUT (under --output-dir):
  activation_by_age_domain.png   one panel per phenomenon, x=age group,
                                  y=selectivity index (or --metric), one
                                  series per dataset that has data at that bin
  activation_by_age_domain.csv   the combined long table the figure is built from

Example:
    python scripts/plot_activation_by_age_domain.py \\
        --dataset ds003604 data/processed/fmri/ds003604/localization \\
        --dataset ds001894 data/processed/fmri/ds001894/localization \\
        --dataset ds006239 data/processed/fmri/ds006239/localization \\
        --dataset ds002236 data/processed/fmri/ds002236/localization \\
        --output-dir paper_results/activation_by_age
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Age-group bin order for the x-axis -- matches configs/age_groups.yaml.
BIN_ORDER = ["5", "7", "9", "11", "11+"]

# Fixed per-dataset marker/color so the same dataset always reads the same
# way across figures -- assigned by hand (not cycled) per the categorical-
# color rule: identity should never depend on which datasets happen to be
# present in a given run.
DATASET_STYLE = {
    "ds003604": {"color": "#2a78d6", "marker": "o", "label": "ds003604 (Wang 2022)"},
    "ds001894": {"color": "#eb6834", "marker": "s", "label": "ds001894 (Lytle 2019)"},
    "ds006239": {"color": "#1baf7a", "marker": "^", "label": "ds006239 (Wang 2025)"},
    "ds002236": {"color": "#4a3aa7", "marker": "D", "label": "ds002236 (Lytle 2020)"},
}


def load_combined(dataset_dirs: dict) -> pd.DataFrame:
    frames = []
    for dataset, loc_dir in dataset_dirs.items():
        f = Path(loc_dir) / "brain_localization_by_session.csv"
        if not f.exists():
            print(f"  ! {dataset}: no {f} -- skipping (run prepare_brain_rdms.sh for "
                  f"it first, or check whether every --append call failed)")
            continue
        df = pd.read_csv(f)
        df["dataset"] = dataset
        df["age_group"] = df["session"].str.replace("^ses-", "", regex=True)
        frames.append(df)
    if not frames:
        raise SystemExit("No brain_localization_by_session.csv found for any --dataset given.")
    return pd.concat(frames, ignore_index=True)


def plot(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    phenomena = sorted(df["phenomenon"].unique())
    n = len(phenomena)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=True, squeeze=False)
    axes = axes[0]

    metric_label = {
        "selectivity_index": "selectivity index (higher = more phenomenon-specific)",
        "brain_localization": "localization (Gini; higher = more concentrated)",
        "mean_overlap_with_others": "overlap with other phenomena (lower = more differentiated)",
        "entropy": "normalized entropy (lower = more concentrated)",
    }.get(metric, metric)

    for ax, phen in zip(axes, phenomena):
        sub = df[df["phenomenon"] == phen]
        for dataset, style in DATASET_STYLE.items():
            d = sub[sub["dataset"] == dataset]
            if d.empty:
                continue
            d = d.set_index("age_group").reindex(
                [b for b in BIN_ORDER if b in d["age_group"].values]
            ).reset_index()
            x = [BIN_ORDER.index(b) for b in d["age_group"]]
            ax.plot(x, d[metric], marker=style["marker"], color=style["color"],
                     label=style["label"], linewidth=1.6, markersize=7)
            # n_subjects annotated at each point -- specialization estimates
            # from n=3 and n=97 are not the same kind of evidence, and this
            # keeps that visible rather than implied only by a footnote.
            for xi, (yi, ni) in zip(x, zip(d[metric], d["n_subjects"])):
                ax.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points",
                            xytext=(0, 7), fontsize=7, color=style["color"], ha="center")
        ax.set_xticks(range(len(BIN_ORDER)))
        ax.set_xticklabels(BIN_ORDER)
        ax.set_xlabel("age group")
        ax.set_title(phen)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(metric_label)
    # Collect legend entries across ALL panels, not just the first: which
    # datasets appear in a given phenomenon's panel varies (Gram/Plaus are
    # ds003604-only; Orth/SemLocal only appear for the other three), so no
    # single panel is guaranteed to have every dataset that's actually
    # plotted somewhere in the figure.
    by_label = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            by_label.setdefault(li, hi)
    handles, labels = list(by_label.values()), list(by_label.keys())
    fig.legend(handles, labels, loc="lower center", ncol=len(DATASET_STYLE),
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Brain specialization by age group and domain, across datasets\n"
                 "(scalar specialization metrics, not a spatial brain map -- see module docstring)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs=2, action="append", metavar=("KEY", "LOCALIZATION_DIR"),
                     required=True, dest="datasets",
                     help="Registry key and its localization output dir "
                          "(<pattern-root>/localization from prepare_brain_rdms.sh). "
                          "Repeat for each dataset to include.")
    ap.add_argument("--metric", default="selectivity_index",
                     choices=["selectivity_index", "brain_localization", "mean_overlap_with_others", "entropy"])
    ap.add_argument("--output-dir", default="paper_results/activation_by_age")
    args = ap.parse_args()

    dataset_dirs = {k: v for k, v in args.datasets}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_combined(dataset_dirs)
    csv_path = out / "activation_by_age_domain.csv"
    df.sort_values(["phenomenon", "age_group", "dataset"]).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    png_path = out / "activation_by_age_domain.png"
    plot(df, args.metric, png_path)
    print(f"Saved: {png_path}")

    print("\ncoverage (dataset x age group x phenomenon):")
    cov = df.pivot_table(index=["phenomenon", "age_group"], columns="dataset",
                          values="n_subjects", aggfunc="sum", fill_value=0)
    print(cov.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
