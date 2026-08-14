#!/usr/bin/env python
"""Compute brain-side ROI/voxel specialization -> L_brain(P, age).

Mirrors scripts/run_circuit_localization.py on the brain: per-voxel
`condition > control` t-map per phenomenon per session, then the same
specialization metrics (Gini/entropy/selectivity/overlap). Emits both a
per-session table and a collapsed per-phenomenon table with `onset_age` for
CoDLA claim C3.

Outputs (under --output-dir):
  brain_localization_by_session.csv   [phenomenon, session, brain_localization, ...]
  brain_specialization.csv            [phenomenon, brain_localization, onset_age]
                                       <-- feed to codla_compare.py --brain-specialization
  fig_brain_localization.png

Example:
  python scripts/run_brain_localization.py \
      --pattern-dir data/processed/fmri \
      --characteristics-dir data/brain/ds003604/stimuli/Stimulus_Characteristics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rsa.brain_localization import (
    SESSION_TO_AGE,
    brain_specialization,
    collapse_onsets,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern-dir", default="data/processed/fmri",
                    help="Dir with sub-*_ses-*_*patterns.npz")
    ap.add_argument("--characteristics-dir",
                    default="data/brain/ds003604/stimuli/Stimulus_Characteristics")
    ap.add_argument("--sessions", nargs="+", default=["ses-5", "ses-7", "ses-9"])
    ap.add_argument("--percentage", type=float, default=5.0,
                    help="Top-%% selective voxels forming the ROI (brain circuit)")
    ap.add_argument("--use-perceptual-control", action="store_true",
                    help="Contrast against perceptual (*_C) control instead of linguistic")
    ap.add_argument("--output-dir", default="data/processed/fmri/localization")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session_df = brain_specialization(
        pattern_dir=args.pattern_dir,
        characteristics_dir=args.characteristics_dir,
        sessions=args.sessions,
        percentage=args.percentage,
        use_perceptual_control=args.use_perceptual_control,
    )
    if session_df.empty:
        raise SystemExit(
            f"No brain patterns found under {args.pattern_dir}. Run preprocessing "
            "first (run_analysis.py) so sub-*_ses-*_*patterns.npz exist."
        )

    by_session = out / "brain_localization_by_session.csv"
    session_df.sort_values(["phenomenon", "session"]).to_csv(by_session, index=False)
    print(f"Saved: {by_session}")

    onsets = collapse_onsets(session_df)
    spec_csv = out / "brain_specialization.csv"
    onsets.to_csv(spec_csv, index=False)
    print(f"Saved: {spec_csv}")
    print("\n=== brain specialization (onset age per phenomenon) ===")
    print(onsets.to_string(index=False))

    _plot(session_df, out / "fig_brain_localization.png")


def _plot(session_df: pd.DataFrame, path: Path) -> None:
    df = session_df.copy()
    df["age"] = df["session"].map(SESSION_TO_AGE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for phen, sub in df.groupby("phenomenon"):
        sub = sub.sort_values("age")
        ax1.plot(sub["age"], sub["brain_localization"], marker="o", label=phen)
        ax2.plot(sub["age"], sub["mean_overlap_with_others"], marker="o", label=phen)
    ax1.set(xlabel="child age (years)", ylabel="brain localization (Gini)",
            title="Cortical specialization ↑ with age")
    ax2.set(xlabel="child age (years)", ylabel="cross-phenomenon voxel overlap",
            title="Cortical differentiation ↓ with age")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
