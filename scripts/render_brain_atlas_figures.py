#!/usr/bin/env python
"""Render REAL anatomical brain figures: group-level MNI152 statistical maps,
one per (dataset, session/age-group, phenomenon), from the per-subject t-maps
`scripts/run_brain_localization.py --mask-cache-dir/--mni-maps-dir` exports.

WHY THIS EXISTS, AND WHY IT'S SEPARATE FROM plot_activation_by_age_domain.py.
That script plots scalar summaries (Gini/entropy/selectivity) as bar/line
charts -- useful, but not an actual picture of the brain: no atlas, no
anatomical space, no "where" answered visually. This script is the "where"
half: it takes the real, spatially-normalized per-subject condition>control
t-maps (unmask()'d back to 3D and warped to MNI152 by
`src.rsa.brain_localization.export_native_tmap_to_mni`, which requires
preprocessing to have run with `--save-native-maps`), combines subjects at
each voxel with a one-sample t-test, and renders the result on the real
MNI152 template with `nilearn.plotting.plot_stat_map` / `plot_glass_brain`.
Both scripts are worth keeping: the scalar one answers "how specialized",
this one answers "specialized where".

INPUT LAYOUT (what run_brain_localization.py's exporter writes):
  <mni_maps_dir>/<dataset>/<subject>_<session>_<phenomenon>_tmap_mni.nii.gz
`session` is a real BIDS session for ds003604 (ses-5/7/9) and an age-group
bin for the three cross-sectional datasets (ses-5/7/9/11/11+ -- see
configs/age_groups.yaml); either way it is exactly the grouping key you want
a figure per.

GROUPING: one group-level map per (dataset, session, phenomenon) triple --
never pooled across datasets, since they differ in scanner, task design, and
population, and pooling silently would be exactly the kind of confound this
project's audit trail exists to catch. Use --datasets to render a subset;
there's no --pool flag on purpose.

STATISTIC: at each voxel, a one-sample t-test of that group's per-subject
t-values against zero (scipy.stats.ttest_1samp), restricted to voxels inside
the standard MNI152 grey-matter-inclusive brain mask (nilearn's bundled
`load_mni152_brain_mask`) and to voxels where every subject in the group has
a non-zero (i.e. in-native-mask) value -- a voxel only one subject's native
mask covered would silently understate the group's variance otherwise.
Groups with a single subject skip the t-test and render that subject's own
t-map directly (labeled as such in the figure title -- not a group result).

Usage:
  python scripts/render_brain_atlas_figures.py \\
      --mni-maps-dir data/processed/fmri/ds002236/_mni_maps \\
      --output-dir figures/brain_atlas
  # restrict to specific datasets/phenomena/sessions:
  python scripts/render_brain_atlas_figures.py --mni-maps-dir ... --output-dir ... \\
      --datasets ds002236 ds001894 --phenomena Sem Phon --min-subjects 3
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FNAME_RE = re.compile(r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_(?P<phenomenon>[^_]+)_tmap_mni\.nii\.gz$")


def _discover(mni_maps_dir: Path) -> pd.DataFrame:
    """Scan <mni_maps_dir>/<dataset>/*_tmap_mni.nii.gz into one row per file."""
    rows = []
    if not mni_maps_dir.is_dir():
        return pd.DataFrame(columns=["dataset", "subject", "session", "phenomenon", "path"])
    for dataset_dir in sorted(p for p in mni_maps_dir.iterdir() if p.is_dir()):
        for f in sorted(dataset_dir.glob("*_tmap_mni.nii.gz")):
            m = FNAME_RE.match(f.name)
            if not m:
                continue  # not one of our files -- ignore rather than guess
            rows.append({
                "dataset": dataset_dir.name, "subject": m["subject"],
                "session": m["session"], "phenomenon": m["phenomenon"], "path": f,
            })
    return pd.DataFrame(rows)


def _group_tmap(paths: List[Path], brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Stack subjects' MNI maps and one-sample t-test at each voxel.

    Returns (group_t, n_contributing_per_voxel, n_subjects). A voxel only
    counts a subject if that subject's value there is non-zero (i.e. inside
    THAT subject's native brain mask before warping) -- ttest_1samp on a
    padded-with-zeros voxel would otherwise treat "outside this subject's
    scan coverage" as a real observed zero and bias the group statistic.
    """
    stack = np.stack([nib.load(str(p)).get_fdata().astype(np.float64) for p in paths], axis=0)
    n_subjects = stack.shape[0]
    covered = (stack != 0)
    n_covering = covered.sum(axis=0)
    group_t = np.zeros(stack.shape[1:], dtype=np.float32)
    # Only test voxels inside the standard brain mask AND covered by >=2 subjects
    # (a t-test needs >=2 observations; 1-covering voxels are left at 0, i.e.
    # not shown -- they're not a group result).
    testable = brain_mask & (n_covering >= 2)
    if testable.any():
        vals = stack[:, testable]  # (n_subjects, n_testable_voxels), some entries are legitimately 0
        with np.errstate(invalid="ignore", divide="ignore"):
            t_ttest, _ = ttest_1samp(np.where(vals != 0, vals, np.nan), popmean=0.0, axis=0, nan_policy="omit")
        group_t[testable] = np.nan_to_num(t_ttest, nan=0.0).astype(np.float32)
    return group_t, n_covering, n_subjects


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mni-maps-dir", required=True, type=Path,
                    help="Root written by run_brain_localization.py's --mni-maps-dir "
                         "(<dataset>/<subject>_<session>_<phenomenon>_tmap_mni.nii.gz).")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--datasets", nargs="+", default=None, help="Restrict to these dataset keys.")
    ap.add_argument("--phenomena", nargs="+", default=None, help="Restrict to these phenomena.")
    ap.add_argument("--sessions", nargs="+", default=None, help="Restrict to these session/age-bin labels.")
    ap.add_argument("--min-subjects", type=int, default=1,
                    help="Skip groups with fewer subjects than this (default: 1, i.e. render "
                         "everything found, single-subject groups included -- see --min-subjects-for-ttest "
                         "for when a group actually gets a real group statistic vs. one subject's own map).")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="|t| display threshold (voxels below are transparent). Default 2.0 "
                         "is a visualization threshold, NOT a corrected significance "
                         "threshold -- these are exploratory figures, not a cluster-corrected result.")
    ap.add_argument("--display-mode", choices=["ortho", "glass"], default="ortho",
                    help="ortho = 3-slice cuts through the template (plot_stat_map), "
                         "glass = glass-brain projection (plot_glass_brain).")
    ap.add_argument("--mni-resolution", type=int, default=2, choices=[1, 2])
    ap.add_argument("--save-nifti", action="store_true",
                    help="Also save the group-level t-map as a .nii.gz next to each figure.")
    args = ap.parse_args()

    from nilearn import datasets as nilearn_datasets
    from nilearn import plotting

    manifest = _discover(args.mni_maps_dir)
    if manifest.empty:
        raise SystemExit(
            f"No *_tmap_mni.nii.gz files found under {args.mni_maps_dir}. These are written by "
            "run_brain_localization.py when called with --mask-cache-dir/--mni-maps-dir, which "
            "itself requires preprocessing to have run with --save-native-maps first -- see "
            "scripts/run_brain_localization.py's module docstring."
        )
    if args.datasets:
        manifest = manifest[manifest["dataset"].isin(args.datasets)]
    if args.phenomena:
        manifest = manifest[manifest["phenomenon"].isin(args.phenomena)]
    if args.sessions:
        manifest = manifest[manifest["session"].isin(args.sessions)]
    if manifest.empty:
        raise SystemExit("Nothing left after --datasets/--phenomena/--sessions filtering.")

    template = nilearn_datasets.load_mni152_template(resolution=args.mni_resolution)
    brain_mask_img = nilearn_datasets.load_mni152_brain_mask(resolution=args.mni_resolution)
    brain_mask = np.asarray(brain_mask_img.get_fdata()) > 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    groups = manifest.groupby(["dataset", "session", "phenomenon"])
    print(f"{len(groups)} (dataset, session, phenomenon) groups from {len(manifest)} subject t-maps")
    for (dataset, session, phenomenon), sub in groups:
        paths = sorted(Path(p) for p in sub["path"])
        n = len(paths)
        if n < args.min_subjects:
            print(f"  {dataset}/{session}/{phenomenon}: {n} subject(s) < --min-subjects {args.min_subjects}, skipping")
            continue

        first_img = nib.load(str(paths[0]))
        if first_img.shape != template.shape or not np.allclose(first_img.affine, template.affine):
            print(f"  {dataset}/{session}/{phenomenon}: shape/affine mismatch with the MNI152 "
                  f"template ({first_img.shape} vs {template.shape}) -- skipping, this should not "
                  "happen for files written by export_native_tmap_to_mni at the same --mni-template-resolution")
            continue

        if n == 1:
            group_t = nib.load(str(paths[0])).get_fdata().astype(np.float32) * brain_mask
            n_covering = brain_mask.astype(int)
            title = f"{dataset} / {session} / {phenomenon} -- single subject ({paths[0].stem.split('_')[0]}), not a group result"
        else:
            group_t, n_covering, _ = _group_tmap(paths, brain_mask)
            title = f"{dataset} / {session} / {phenomenon} -- one-sample t, n={n} subjects"

        group_img = nib.Nifti1Image(group_t, template.affine)
        out_stem = args.output_dir / f"{dataset}_{session}_{phenomenon}"
        fig_path = out_stem.with_suffix(".png")

        fig = plt.figure(figsize=(11, 4.5))
        if args.display_mode == "glass":
            disp = plotting.plot_glass_brain(
                group_img, threshold=args.threshold, colorbar=True, title=title,
                plot_abs=False, figure=fig,
            )
        else:
            disp = plotting.plot_stat_map(
                group_img, bg_img=template, threshold=args.threshold, colorbar=True,
                title=title, figure=fig,
            )
        fig.savefig(fig_path, dpi=170)
        disp.close()
        plt.close(fig)
        print(f"  {dataset}/{session}/{phenomenon}: n={n}, |t|>{args.threshold} voxels="
              f"{int((np.abs(group_t) > args.threshold).sum())} -> {fig_path}")

        if args.save_nifti:
            nib.save(group_img, str(out_stem.with_suffix(".nii.gz")))

        summary_rows.append({
            "dataset": dataset, "session": session, "phenomenon": phenomenon, "n_subjects": n,
            "n_voxels_above_threshold": int((np.abs(group_t) > args.threshold).sum()),
            "max_abs_t": float(np.abs(group_t).max()),
            "figure": str(fig_path),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "brain_atlas_manifest.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {len(summary)} figures. Manifest: {summary_path}")


if __name__ == "__main__":
    main()
