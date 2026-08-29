#!/usr/bin/env python
"""Backfill real trial_types/semantic_categories into session RDMs for the
three stim_pair_filename datasets (ds001894, ds002236, ds006239).

WHY THIS EXISTS. Every one of these RDMs was written with trial_types and
semantic_categories hardcoded to the placeholder "unknown" for every
stimulus -- src/rsa/semantic_metadata.py had no dataset-aware way to recover
the real positive/negative label at the time (no per-task characteristics
table exists for these datasets the way it does for ds003604). That silently
degenerated scripts/positive_control.py's `condition` control: it builds a
design-contrast RDM that is 0 within condition / 1 between, and with one
label ("unknown") on every stimulus there is no "between" to test -- the
control could never run, and never showed up as tested at all (it isn't in
paper_results/*/control/control_summary.csv's control list), which is why
ds002236's gate result was really "1 of ~9 possible controls had any data",
not "9 tested, all failed".

Fixed at the source (src/rsa/semantic_metadata.py's
conditions_from_stim_lookup, session_based_rsa.py now threads --dataset
through to it) -- but the RDMs already published before that fix still carry
"unknown". Rather than rebuild hours of fMRI preprocessing, this backfills
them in place from the SAME classify_trials()-derived lookup the fix uses
(src/rsa/brain_localization.py's build_stim_lookup_for_dataset, which reads
real events.tsv), so a label here can never disagree with what a stimulus
was actually treated as when its pattern was extracted. The RDM matrix
itself is never touched, only the metadata arrays alongside it.

Idempotent: an RDM whose trial_types already has no "unknown" is left alone.

    python scripts/backfill_rdm_conditions.py --roots data/processed/fmri/ds002236 --dataset ds002236
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rsa.semantic_metadata import load_semantic_metadata  # noqa: E402


def backfill(path: Path, dataset: str, characteristics_dir: str) -> str:
    task = path.parent.name
    d = np.load(path, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    stimuli = [str(s) for s in data.get("stimuli", [])]
    if not stimuli:
        return "skip (no stimuli)"

    existing = data.get("trial_types")
    if existing is not None and len(existing) == len(stimuli):
        if "unknown" not in {str(t) for t in existing}:
            return f"ok already ({len(stimuli)} stimuli)"

    meta = load_semantic_metadata(
        stimuli, task=task, dataset=dataset, characteristics_dir=characteristics_dir,
    )
    trial_types = meta.get("trial_types")
    if trial_types is None:
        return "FAILED (no metadata returned -- unregistered task/dataset?)"
    trial_types = [str(t) for t in trial_types]
    n_unknown = sum(1 for t in trial_types if t == "unknown")
    if n_unknown == len(trial_types):
        return "FAILED (still all 'unknown' -- no events.tsv for this dataset locally?)"

    data["trial_types"] = np.asarray(trial_types, dtype=object)
    data["semantic_categories"] = np.asarray(
        [str(t) for t in meta.get("semantic_categories", trial_types)], dtype=object
    )
    # np.savez_compressed appends ".npz" unless the name already ends with it.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)
    n_pos = sum(1 for t in trial_types if t == "positive")
    n_neg = sum(1 for t in trial_types if t == "negative")
    extra = f", {n_unknown} unmatched" if n_unknown else ""
    return f"backfilled {len(trial_types)} labels ({n_pos} positive, {n_neg} negative{extra})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="RDM root(s) to walk, e.g. data/processed/fmri/ds002236 "
                         "(scans <root>/<task>/session_rdm_ses-*.npz).")
    ap.add_argument("--dataset", required=True,
                    help="Registry key -- needed to rebuild the stim_id -> condition "
                         "lookup from this dataset's own events.tsv (see "
                         "src.rsa.semantic_metadata.conditions_from_stim_lookup). "
                         "ds003604 never needs this script -- its RDMs get real "
                         "trial_types from its own characteristics table already.")
    ap.add_argument("--characteristics-dir", default=None,
                    help="Default: data/brain/<dataset>/stimuli/Stimulus_Characteristics "
                         "(which won't exist for these datasets -- that absence is what "
                         "routes load_semantic_metadata to the events.tsv-based lookup).")
    args = ap.parse_args()

    char_dir = args.characteristics_dir or f"data/brain/{args.dataset}/stimuli/Stimulus_Characteristics"

    rc = 0
    for root in args.roots:
        for p in sorted(Path(root).glob("*/session_rdm_ses-*.npz")):
            msg = backfill(p, args.dataset, char_dir)
            print(f"{p.parent.name:10s} {p.name:26s} {msg}")
            if msg.startswith("FAILED"):
                rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
