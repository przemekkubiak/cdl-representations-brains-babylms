#!/usr/bin/env python
"""Fail loudly if a tree of session RDMs is not what it claims to be.

WHY THIS EXISTS. On 2026-08-25 a raw, uncorrected RDM was found sitting in
data/processed/fmri_wrn/ (the within-run-normalised tree), left behind by an
earlier pilot. prepare_brain_rdms.sh skips any cell whose RDM already exists, so
that one file would have been silently adopted as the corrected Phon/ses-5 cell
and a confounded cell would have been baked into the corrected results with no
outward sign. It was caught only because every RDM now records
`within_run_normalized`.

Run this before trusting any tree, and before any grid scores models against it.

Exit codes: 0 all consistent · 1 mismatch found · 2 nothing to check.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdm-root", required=True)
    ap.add_argument("--expect-within-run-normalized", dest="expect",
                    choices=["true", "false"], default="true",
                    help="what every RDM in this tree must be (default: true)")
    ap.add_argument("--require-cells", type=int, default=0,
                    help="fail if fewer than this many RDMs are present (0 = no minimum)")
    ap.add_argument("--require-ceilings", action="store_true",
                    help="also require every RDM to carry per-subject RDMs")
    a = ap.parse_args()

    expect = a.expect == "true"
    files = sorted(Path(a.rdm_root).rglob("session_rdm_ses-*.npz"))
    if not files:
        print(f"nothing to check: no session RDMs under {a.rdm_root}")
        return 2

    bad, missing_flag, missing_ceil = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "within_run_normalized" not in d.files:
            missing_flag.append(f)
            continue
        if bool(d["within_run_normalized"]) != expect:
            bad.append((f, bool(d["within_run_normalized"])))
        if a.require_ceilings and (
            "subject_rdms" not in d.files or not len(d["subject_rdms"])
        ):
            missing_ceil.append(f)

    print(f"checked {len(files)} session RDMs under {a.rdm_root}")
    print(f"  expected within_run_normalized = {expect}")

    ok = True
    if bad:
        ok = False
        print(f"  MISMATCH -- {len(bad)} file(s) carry the WRONG correction state:")
        for f, got in bad:
            print(f"    {f}  (within_run_normalized={got})")
        print("  These must be deleted and rebuilt. Leaving them in place means the")
        print("  next run SKIPS those cells and adopts them as if they were correct.")
    if missing_flag:
        ok = False
        print(f"  UNKNOWN PROVENANCE -- {len(missing_flag)} file(s) predate the flag:")
        for f in missing_flag:
            print(f"    {f}")
        print("  Built before provenance was recorded; they cannot be verified. Rebuild.")
    if missing_ceil:
        ok = False
        print(f"  NO SUBJECT RDMs -- {len(missing_ceil)} file(s) cannot yield a ceiling:")
        for f in missing_ceil:
            print(f"    {f}")
    if a.require_cells and len(files) < a.require_cells:
        ok = False
        print(f"  INCOMPLETE -- {len(files)} cells present, {a.require_cells} required.")

    print("  OK -- tree is consistent" if ok else "  FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
