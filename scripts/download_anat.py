#!/usr/bin/env python
"""Download T1w anatomical scans (git-annex symlinks -> real files).

WHY THIS EXISTS. src/preprocessing/spatial_normalization.py's per-subject
EPI->T1->MNI152 registration -- what ROI_SET (src/preprocessing/roi_atlas.py,
MASKING.md) needs to restrict a subject's mask to a named region -- requires
a real T1w file via find_t1w(). Nothing in this pipeline has ever downloaded
one: scripts/batch_download_bold.py only resolves func/*bold.nii.gz, and
scripts/download_stimuli.py only resolves stimuli/. Confirmed 2026-08-29:
every T1w path under a metadata-only checkout is a dangling git-annex
symlink. Since build_subject_roi_mask falls back to the whole-brain mask
(logged, not raised) whenever registration fails for any reason -- including
"T1w file not found" -- an ROI_SET run against undownloaded T1w scans would
silently produce whole-brain-equivalent output labeled as ROI-restricted,
for every subject, with no error. This script is the fix: run it before any
--roi-set run, same as download_stimuli.py before a positive-control run
that wants the acoustic control.

Reuses batch_download_bold.py's URL-resolution/download primitives via
download_stimuli.py's fetch() -- both are already dataset- and path-agnostic
(candidate_urls works off the relative path + annex target, not a
BOLD-specific assumption), so no third copy of that logic.

Small, like download_stimuli.py: one T1w per subject-session, a few MB each.

    python scripts/download_anat.py --dataset ds002236
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_stimuli import fetch  # noqa: E402
from src.datasets import get_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ds003604")
    ap.add_argument("--data-root", default="data/brain")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Restrict to these subject IDs (default: every subject with an anat/ dir).")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    a = ap.parse_args()

    spec = get_dataset(a.dataset)
    data_dir = Path(a.data_root) / spec.require_downloadable()

    # Both layouts in the registry: sub-XX/anat/ (ds002236, ds006239 --
    # single cross-sectional visit) and sub-XX/ses-YY/anat/ (ds003604,
    # ds001894 -- session-nested, matching find_t1w's own fallback order).
    files = sorted(data_dir.glob("sub-*/anat/*T1w.nii.gz"))
    if not files:
        files = sorted(data_dir.glob("sub-*/ses-*/anat/*T1w.nii.gz"))
    if a.subjects:
        wanted = set(a.subjects)
        files = [f for f in files if f.relative_to(data_dir).parts[0] in wanted]
    if not files:
        print(f"no T1w files found under {data_dir} (sub-*/anat/ or sub-*/ses-*/anat/)")
        sys.exit(1)
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} T1w files under {data_dir}")

    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(fetch, f, data_dir, spec): f for f in files}
        for fut in as_completed(futs):
            path, ok, msg = fut.result()
            if not ok:
                failed += 1
                print(f"  FAIL {path.name}: {msg}")
            elif msg.startswith("present"):
                skipped += 1
            else:
                done += 1

    total_mb = sum(f.stat().st_size for f in files if f.exists()) / 1e6
    print(f"downloaded {done}, already present {skipped}, failed {failed} "
          f"-- {total_mb:.0f} MB on disk")
    sys.exit(1 if failed and not done else 0)


if __name__ == "__main__":
    main()
