#!/usr/bin/env python
"""Estimate session-wise RSA noise ceilings from subject-level RDMs.

Lower ceiling: leave-one-subject-out group mean vs held-out subject.
Upper ceiling: full-group mean vs each subject.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import sys

# Allow project imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rsa import compare_rdms
from src.rsa.session_based_rsa import SessionBasedRSA


def _common_stimuli_for_session(rsa: SessionBasedRSA, session: str) -> List[str]:
    """Find stimuli shared across all subjects that have this session."""
    subject_session_stimuli = []
    for subject_data in rsa.patterns_by_subject.values():
        if session not in subject_data:
            continue

        run_sets = [
            set(run_data.keys())
            for run_data in subject_data[session].values()
            if run_data
        ]
        if not run_sets:
            continue

        subject_union = set.union(*run_sets)
        if subject_union:
            subject_session_stimuli.append(subject_union)

    if not subject_session_stimuli:
        return []

    return sorted(set.intersection(*subject_session_stimuli))


def estimate_noise_ceiling(
    pattern_dir: str,
    output_dir: str,
    sessions: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None,
    metric: str = "correlation",
    method: str = "spearman",
) -> pd.DataFrame:
    rsa = SessionBasedRSA(pattern_dir=pattern_dir)
    rsa.load_all_patterns(subjects=subjects, sessions=sessions)

    available_sessions = sorted(
        {
            sess
            for subj_data in rsa.patterns_by_subject.values()
            for sess in subj_data.keys()
        }
    )

    if sessions:
        available_sessions = [s for s in available_sessions if s in sessions]

    if not available_sessions:
        raise ValueError("No sessions available for noise ceiling estimation")

    rows = []

    for session in available_sessions:
        common_stimuli = _common_stimuli_for_session(rsa, session)

        if not common_stimuli:
            rows.append(
                {
                    "session": session,
                    "n_subjects": 0,
                    "n_stimuli": 0,
                    "lower_ceiling": np.nan,
                    "upper_ceiling": np.nan,
                    "lower_std": np.nan,
                    "upper_std": np.nan,
                    "method": method,
                    "metric": metric,
                }
            )
            continue

        subject_ids = []
        subject_rdms = []

        for subject_id in sorted(rsa.patterns_by_subject.keys()):
            rdm = rsa.compute_subject_rdm(
                subject_id=subject_id,
                session=session,
                common_stimuli=common_stimuli,
                metric=metric,
            )
            if rdm is not None:
                subject_ids.append(subject_id)
                subject_rdms.append(rdm)

        n_subjects = len(subject_rdms)

        if n_subjects < 2:
            rows.append(
                {
                    "session": session,
                    "n_subjects": n_subjects,
                    "n_stimuli": len(common_stimuli),
                    "lower_ceiling": np.nan,
                    "upper_ceiling": np.nan,
                    "lower_std": np.nan,
                    "upper_std": np.nan,
                    "method": method,
                    "metric": metric,
                }
            )
            continue

        rdms = np.array(subject_rdms)
        group_mean = np.mean(rdms, axis=0)

        upper_scores = []
        lower_scores = []

        for i in range(n_subjects):
            subject_rdm = rdms[i]

            upper_r, _ = compare_rdms(subject_rdm, group_mean, method=method)
            upper_scores.append(float(upper_r))

            loo_mean = np.mean(np.delete(rdms, i, axis=0), axis=0)
            lower_r, _ = compare_rdms(subject_rdm, loo_mean, method=method)
            lower_scores.append(float(lower_r))

        rows.append(
            {
                "session": session,
                "n_subjects": n_subjects,
                "n_stimuli": len(common_stimuli),
                "lower_ceiling": float(np.mean(lower_scores)),
                "upper_ceiling": float(np.mean(upper_scores)),
                "lower_std": float(np.std(lower_scores)),
                "upper_std": float(np.std(upper_scores)),
                "method": method,
                "metric": metric,
            }
        )

    df = pd.DataFrame(rows).sort_values("session")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "noise_ceiling_by_session.csv"
    df.to_csv(out_file, index=False)

    print("\nNoise ceiling estimation complete")
    print(f"Saved: {out_file}")
    print(df.to_string(index=False))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate session-level RSA noise ceilings")
    parser.add_argument("--pattern-dir", default="data/processed/fmri")
    parser.add_argument("--output-dir", default="data/processed/fmri")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", choices=["ses-5", "ses-7", "ses-9"], default=None)
    parser.add_argument("--metric", default="correlation", choices=["correlation", "euclidean", "cosine"])
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    args = parser.parse_args()

    estimate_noise_ceiling(
        pattern_dir=args.pattern_dir,
        output_dir=args.output_dir,
        sessions=args.sessions,
        subjects=args.subjects,
        metric=args.metric,
        method=args.method,
    )


if __name__ == "__main__":
    main()
