#!/usr/bin/env python
"""Emit an explicit dataset x ROI coverage matrix for the RSA results.

WHY THIS EXISTS. Twice in this project a job that did nothing has looked like a
job that found nothing: a sweep that skipped models with existing CSVs reported
a 3-cell comparison as if it were the full grid, and an ROI run whose
registration failed silently fell back to whole-brain and was labelled
roi-phonology. Both were caught by accident. The defence is to state coverage
explicitly rather than infer it from the presence of a directory -- an empty
`paper_results/ds002236/roi-phonology/` currently makes that dataset look as
though it has phonology results when it has none.

Writes results/coverage_matrix.csv and prints the same table.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path("/local/scratch/sas245/brainalign-evals")
ROIS = ["within-run-normalised", "roi-phonology", "roi-auditory", "roi-motor",
        "roi-language", "roi-all"]
# A whole-brain mask is hundreds of thousands of voxels; the ROI masks built so
# far are 13k-16k. Anything above this in a file labelled as an ROI is the
# whole-brain fallback wearing an ROI name.
WHOLE_BRAIN_VOXEL_FLOOR = 60000


def cells() -> pd.DataFrame:
    f = ROOT / "results" / "roi_by_cell.csv"
    if not f.exists():
        return pd.DataFrame(columns=["dataset", "variant"])
    return pd.read_csv(f)


def mask_health(ds: str) -> dict:
    """Did the ROI masks actually restrict to an ROI, or fall back to whole brain?"""
    f = ROOT / "pipeline/data/processed/fmri" / ds / "_masks/roi_mask_status.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    out = {}
    for roi, g in d.groupby("roi_set"):
        ok = int((g.status == "ok").sum())
        big = int((g.get("n_roi_voxels", pd.Series(dtype=float))
                   > WHOLE_BRAIN_VOXEL_FLOOR).sum())
        out[str(roi)] = dict(ok=ok, failed=int(len(g) - ok), suspect_wholebrain=big,
                             median_voxels=float(g.n_roi_voxels.median())
                             if "n_roi_voxels" in g else float("nan"))
    return out


def main() -> int:
    d = cells()
    datasets = sorted(set(d.dataset.dropna().unique()) | {"ds001894", "ds006239"})
    rows = []
    for ds in datasets:
        health = mask_health(ds)
        for roi in ROIS:
            n = int(((d.dataset == ds) & (d.variant == roi)).sum()) if len(d) else 0
            key = roi.replace("roi-", "")
            h = health.get(key, {})
            # A directory can exist with nothing in it; say so rather than let
            # its presence imply a result.
            dirp = ROOT / "pipeline/paper_results" / ds / roi
            empty_dir = dirp.is_dir() and not any(dirp.iterdir())
            rows.append(dict(
                dataset=ds, variant=roi, cells=n,
                status=("results" if n else
                        "EMPTY DIR (no results)" if empty_dir else "not run"),
                masks_ok=h.get("ok", ""), masks_failed=h.get("failed", ""),
                suspect_wholebrain=h.get("suspect_wholebrain", ""),
                median_roi_voxels=h.get("median_voxels", "")))
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results/coverage_matrix.csv", index=False)
    print(out.to_string(index=False))
    bad = out[out.suspect_wholebrain.apply(lambda v: bool(v) and v != 0)]
    if len(bad):
        print("\nWARNING: ROI masks larger than a plausible ROI -- possible "
              "whole-brain fallback:\n" + bad.to_string(index=False))
    print(f"\nwrote {ROOT/'results/coverage_matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
