#!/usr/bin/env python
"""Compute brain-side ROI/voxel specialization -> L_brain(P, age).

Mirrors scripts/run_circuit_localization.py on the brain: per-voxel
`condition > control` t-map per phenomenon per session, then the same
specialization metrics (Gini/entropy/selectivity/overlap). Emits both a
per-session table and a collapsed per-phenomenon table with `onset_age` for
CoDLA claim C3.

WHY --append / --finalize-only EXIST -- READ BEFORE REMOVING THEM.
prepare_brain_rdms.sh reclaims (deletes) a task's pattern files immediately
after that task's session RDM is built, to stay under the disk floor
(PICKUP.md) -- and it always has. This script's ORIGINAL single-shot design
(scan --pattern-dir for every session, compute everything, write one table)
therefore had no patterns left to read by the time it ran once at the very
end of a full sweep: every task's patterns were already gone. There is no
trace of a real brain_localization_by_session.csv or brain_specialization.csv
anywhere in paper_results/ or hf_results_staging/ -- this was silently never
producing real output, for ds003604 or anyone, not a dataset-specific gap.

The fix is call-site, not algorithmic: call this script PER (task, session),
with --append, right before that session's patterns are reclaimed (so the
scan actually finds them), and call it ONCE MORE with --finalize-only at the
very end to collapse the now-complete accumulated table into onsets + a plot.
See the age-group / per-session blocks in prepare_brain_rdms.sh for exactly
where these two calls sit in the loop.

Outputs (under --output-dir):
  brain_localization_by_session.csv   [phenomenon, session, brain_localization, ...]
                                       accumulated across --append calls, deduped
                                       on (phenomenon, session) -- a rerun of the
                                       same session overwrites its own old row.
  brain_specialization.csv            [phenomenon, brain_localization, onset_age]
                                       <-- feed to codla_compare.py --brain-specialization
                                       (written only by --finalize-only)
  fig_brain_localization.png          (written only by --finalize-only)

Example (single dataset, everything already on disk -- the old use case,
still supported when patterns haven't been reclaimed):
  python scripts/run_brain_localization.py \
      --pattern-dir data/processed/fmri \
      --characteristics-dir data/brain/ds003604/stimuli/Stimulus_Characteristics

Example (the streaming use case prepare_brain_rdms.sh actually needs):
  python scripts/run_brain_localization.py --dataset ds002236 \
      --pattern-dir "$OUT" --sessions ses-9 --append --output-dir "$RDM_ROOT/localization"
  # ... (repeat per session, then reclaim patterns) ...
  python scripts/run_brain_localization.py --finalize-only --output-dir "$RDM_ROOT/localization"

Optional: real anatomical brain maps (not just the scalar Gini/entropy table).
Requires --save-native-maps to have been passed at preprocessing time (writes
each subject's whole-brain mask + triggers registration while the BOLD file
still exists). Add --mask-cache-dir/--mni-maps-dir to any non-finalize call
above and each subject/session/phenomenon t-map is additionally reconstructed
in 3D and warped to MNI space:
  python scripts/run_brain_localization.py --dataset ds002236 \
      --pattern-dir "$OUT" --sessions ses-9 --append --output-dir "$RDM_ROOT/localization" \
      --mask-cache-dir data/processed/masks/ds002236 --mni-maps-dir data/processed/mni_maps
Then aggregate + render with scripts/render_brain_atlas_figures.py.
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
                    help="Dir with sub-*_ses-*_*patterns.npz. Not read at all in --finalize-only mode.")
    ap.add_argument("--dataset", default="ds003604",
                    help="Registry key from configs/neuro_datasets.yaml -- picks the "
                         "stimulus-lookup path (build_stim_lookup_for_dataset) and, for "
                         "ds003604, the default --characteristics-dir.")
    ap.add_argument("--characteristics-dir", default=None,
                    help="Only consulted when --dataset=ds003604 (default: "
                         "data/brain/ds003604/stimuli/Stimulus_Characteristics). Ignored otherwise.")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="Sessions/age-group bins to process (default: everything found "
                         "under --pattern-dir). Dataset-specific values -- see DATASETS.md.")
    ap.add_argument("--percentage", type=float, default=5.0,
                    help="Top-%% selective voxels forming the ROI (brain circuit)")
    ap.add_argument("--use-perceptual-control", action="store_true",
                    help="Contrast against perceptual (*_C) control instead of linguistic. "
                         "ds003604 only -- see build_stim_lookup_for_dataset's NotImplementedError.")
    ap.add_argument("--output-dir", default="data/processed/fmri/localization")
    ap.add_argument("--mask-cache-dir", default=None,
                    help="Dir with per-subject whole-brain masks + registration cache, written "
                         "during preprocessing by --save-native-maps (see batch_preprocessing.py). "
                         "Together with --mni-maps-dir, turns on exporting real per-subject "
                         "condition t-maps warped to MNI space alongside the scalar metrics. "
                         "Omit (default) to skip spatial export entirely -- the scalar table is "
                         "unaffected either way.")
    ap.add_argument("--mni-maps-dir", default=None,
                    help="Output dir for exported MNI-space t-maps "
                         "(<dataset>/<subject>_<session>_<phenomenon>_tmap_mni.nii.gz). Must be "
                         "given together with --mask-cache-dir. Feed the result to "
                         "scripts/render_brain_atlas_figures.py for group-level atlas figures.")
    ap.add_argument("--data-dir", default=None,
                    help="Dataset root containing sub-*/anat/*_T1w.nii.gz, needed to look up each "
                         "subject's T1 for registration. Default: the registry's data_dir() for "
                         "--dataset. Only consulted when --mask-cache-dir/--mni-maps-dir are set.")
    ap.add_argument("--append", action="store_true",
                    help="Merge newly-computed rows into an existing "
                         "brain_localization_by_session.csv (deduped on phenomenon+session, "
                         "new rows win) instead of overwriting it. Use this for every "
                         "per-session call in a streaming sweep -- see the module docstring.")
    ap.add_argument("--finalize-only", action="store_true",
                    help="Skip scanning --pattern-dir entirely; read the already-accumulated "
                         "brain_localization_by_session.csv in --output-dir and just write "
                         "brain_specialization.csv + the figure. Call this ONCE, after every "
                         "--append call for a sweep, not per session.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_session_path = out / "brain_localization_by_session.csv"

    if bool(args.mask_cache_dir) != bool(args.mni_maps_dir):
        raise SystemExit("--mask-cache-dir and --mni-maps-dir must be given together (or neither).")

    if args.finalize_only:
        if not by_session_path.exists():
            raise SystemExit(
                f"--finalize-only but {by_session_path} does not exist -- run the "
                "per-session --append calls first."
            )
        session_df = pd.read_csv(by_session_path)
    else:
        session_df = brain_specialization(
            pattern_dir=args.pattern_dir,
            characteristics_dir=args.characteristics_dir,
            sessions=args.sessions,
            percentage=args.percentage,
            use_perceptual_control=args.use_perceptual_control,
            dataset=args.dataset,
            mask_cache_dir=args.mask_cache_dir,
            mni_maps_dir=args.mni_maps_dir,
            data_dir=args.data_dir,
        )
        if session_df.empty:
            raise SystemExit(
                f"No brain patterns found under {args.pattern_dir} for dataset "
                f"'{args.dataset}'. If this ran after preprocessing, check whether "
                "patterns were already reclaimed -- this call needs to happen BEFORE "
                "that, see the module docstring."
            )

        if args.append and by_session_path.exists():
            existing = pd.read_csv(by_session_path)
            combined = pd.concat([existing, session_df], ignore_index=True)
            # New rows win: a rerun of a session that already has a row replaces it
            # rather than duplicating it.
            combined = combined.drop_duplicates(subset=["phenomenon", "session"], keep="last")
            session_df = combined

        session_df.sort_values(["phenomenon", "session"]).to_csv(by_session_path, index=False)
        print(f"Saved: {by_session_path} ({len(session_df)} phenomenon x session rows)")
        if not args.finalize_only and args.append:
            return  # the finalize step (onsets + plot) runs separately, once

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
