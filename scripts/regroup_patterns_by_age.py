#!/usr/bin/env python
"""Relabel per-subject pattern files by real age-group bin, not on-disk session.

WHY THIS IS A SEPARATE STEP, NOT A CHANGE TO session_based_rsa.py.
ds003604's three BIDS sessions already correspond to three developmental
timepoints almost exactly (configs/age_groups.yaml `verified_against`), so
its RDM-aggregation code groups subjects by literal session label
(sub-X_ses-Y_run-Z_patterns.npz -> session "ses-Y") and that has always been
correct. The three datasets added 2026-08-26 are cross-sectional: a single
BIDS session (or none at all) can span multiple age-group bins -- ds001894's
ses-T1 alone runs 7.36-14.54 years. Pooling "ses-T1" as one RDM would average
together children years apart in development, which is exactly the kind of
silent, plausible-looking mistake this project's whole audit trail exists to
catch.

The fix is NOT to teach session_based_rsa.py about ages -- that file is the
core engine behind every already-published ds003604 number, and the safer
change is zero changes to it. Instead: after batch_preprocessing.py produces
per-subject pattern files (labeled with whatever session they were scanned
in, or fmri_preprocessing.SESSIONLESS_LABEL if the dataset has none), this
script makes a RELABELED COPY of each subject's pattern file under the
age-group bin their real age falls into, so session_based_rsa.py can group by
"session" exactly as it always has -- the session strings it sees just now
mean age-group bins for these three datasets, not BIDS sessions.

ds003604 is a no-op here by design (skipped entirely, see main()) --
its existing behaviour is untouched.

Usage:
    python scripts/regroup_patterns_by_age.py --dataset ds002236 \\
        --pattern-dir data/processed/fmri/ds002236/Phon
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.age_groups import per_subject_ages, bin_of, SINGLE_SESSION

# ds003604's sessions are already developmental timepoints 1:1 -- regrouping
# them would be a no-op at best and a silent bug at worst if age_groups.yaml
# ever drifted from ds003604's own definitions. Never touch it here.
SKIP_DATASETS = {"ds003604"}


def regroup(dataset: str, pattern_dir: Path, mode: str = "symlink") -> int:
    if dataset in SKIP_DATASETS:
        print(f"{dataset}: skipped by design (its sessions are already age groups)")
        return 0

    ages = per_subject_ages(dataset)
    pattern_dir = Path(pattern_dir)
    files = sorted(pattern_dir.glob("sub-*_ses-*_run-*_patterns.npz"))
    if not files:
        print(f"no pattern files found under {pattern_dir}")
        return 0

    n_relabeled, n_no_age, n_already = 0, 0, 0
    no_age_subjects = set()
    for f in files:
        # sub-<ID>_ses-<LABEL>_run-<N>_patterns.npz -- LABEL may itself
        # contain no further underscores (ds003604-style) or be the
        # SESSIONLESS_LABEL "all" (from ses-all).
        parts = f.stem.split("_")  # ['sub-01', 'ses-all', 'run-01', 'patterns']
        subject = parts[0]
        on_disk_session = parts[1]  # e.g. "ses-all", "ses-1", "ses-T1"
        run = parts[2]

        subj_ages = ages.get(subject)
        if not subj_ages:
            n_no_age += 1
            no_age_subjects.add(subject)
            continue
        # The on-disk session label maps to an age key: either the literal
        # session (ds001894's ses-T1/ses-T2) or SINGLE_SESSION (ds006239/
        # ds002236, which have at most one age per subject regardless of
        # which BIDS session/lack thereof the file came from).
        age_key = on_disk_session if on_disk_session in subj_ages else SINGLE_SESSION
        age = subj_ages.get(age_key)
        if age is None:
            n_no_age += 1
            no_age_subjects.add(subject)
            continue

        bin_name = bin_of(age)
        new_session = f"ses-{bin_name}"
        target = pattern_dir / f"{subject}_{new_session}_{run}_patterns.npz"
        if target.exists():
            n_already += 1
            continue
        if mode == "symlink":
            target.symlink_to(f.name)
        else:
            shutil.copy2(f, target)
        n_relabeled += 1

    print(f"{dataset}: {len(files)} pattern files, {n_relabeled} relabeled "
          f"({mode}), {n_already} already present, {n_no_age} skipped (no age "
          f"on record for {len(no_age_subjects)} subjects)")
    if no_age_subjects:
        print(f"  subjects with no age: {sorted(no_age_subjects)[:10]}"
              f"{' ...' if len(no_age_subjects) > 10 else ''}")
    return n_relabeled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pattern-dir", required=True, type=Path,
                     help="Directory of sub-*_ses-*_run-*_patterns.npz files (one task's worth)")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                     help="symlink (default, no extra disk) or copy (if the "
                          "pattern dir will be moved/archived independently)")
    args = ap.parse_args()
    regroup(args.dataset, args.pattern_dir, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
