#!/usr/bin/env python
"""Run the analysis pipeline separately for each ROI mask."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from nilearn import datasets


def run_command(cmd: list[str], description: str) -> None:
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise SystemExit(f"{description} failed with code {result.returncode}")


def sanitize_label(label: str) -> str:
    return (
        label.strip()
        .replace("/", "_")
        .replace(" ", "_")
        .replace(",", "_")
        .replace("__", "_")
    )


def roi_label_from_ids(roi: str, atlas_version: str) -> str:
    atlas = datasets.fetch_atlas_aal(version=atlas_version)

    index_to_label = {}
    for idx, label in zip(atlas.indices, atlas.labels):
        try:
            index_to_label[int(idx)] = str(label)
        except ValueError:
            continue

    selected_names = []
    for roi_id in roi.split(","):
        roi_id = roi_id.strip()
        if not roi_id:
            continue

        try:
            code = int(roi_id)
        except ValueError:
            selected_names.append(roi_id)
            continue

        selected_names.append(index_to_label.get(code, f"aal_{code}"))

    if not selected_names:
        return sanitize_label(f"roi_{roi}")

    return sanitize_label("__".join(selected_names))


def build_mask_command(
    roi: str,
    output_mask: Path,
    data_dir: Path,
    task: str,
    aal_version: str,
) -> list[str]:
    reference_bold = None
    for candidate in sorted(data_dir.glob(f"sub-*/ses-*/func/*task-{task}*_bold.nii.gz")):
        reference_bold = str(candidate)
        break

    cmd = [
        sys.executable,
        "src/preprocessing/prepare_language_mask.py",
        "--aal-rois",
        roi,
        "--aal-version",
        aal_version,
        "--output-mask",
        str(output_mask),
    ]

    if reference_bold:
        cmd.extend(["--reference-bold", reference_bold])

    return cmd


def build_pipeline_command(
    data_dir: Path,
    output_dir: Path,
    task: str,
    characteristics_dir: Path,
    mask_path: Path,
    subjects: list[str] | None,
    sessions: list[str] | None,
    smoothing_fwhm: float,
    high_pass: float,
    metric: str,
    aggregation: str,
    n_iter: int,
    features: int | None,
    comparison_method: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "run_analysis.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--task",
        task,
        "--characteristics-dir",
        str(characteristics_dir),
        "--mask-path",
        str(mask_path),
        "--metric",
        metric,
        "--aggregation",
        aggregation,
        "--comparison-method",
        comparison_method,
        "--semantic-distance-summary",
        "--roi-label",
        output_dir.name,
    ]

    if subjects:
        cmd.extend(["--subjects"] + subjects)
    if sessions:
        cmd.extend(["--sessions"] + sessions)

    if aggregation == "hyperalignment":
        cmd.extend(["--n-iter", str(n_iter)])
        if features is not None:
            cmd.extend(["--features", str(features)])

    if smoothing_fwhm is not None:
        cmd.extend(["--smoothing-fwhm", str(smoothing_fwhm)])
    if high_pass is not None:
        cmd.extend(["--high-pass", str(high_pass)])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the analysis pipeline once per ROI")
    parser.add_argument("--data-dir", type=str, default="data/brain/ds003604")
    parser.add_argument(
        "--task",
        type=str,
        default="Sem",
        choices=["Sem", "Phon", "Gram", "Plaus"],
        help="Task to run for each ROI",
    )
    parser.add_argument(
        "--characteristics-dir",
        type=str,
        default="data/brain/ds003604/stimuli/Stimulus_Characteristics",
    )
    parser.add_argument("--output-root", type=str, default="data/processed/fmri/roi_runs")
    parser.add_argument(
        "--rois",
        nargs="+",
        type=str,
        default=["7", "8", "9", "10", "11", "12", "67", "68", "69", "70", "85", "86"],
        help="AAL ROI codes or comma-separated code groups",
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--sessions", nargs="+", choices=["ses-5", "ses-7", "ses-9"])
    parser.add_argument("--smoothing-fwhm", type=float, default=6.0)
    parser.add_argument("--high-pass", type=float, default=0.01)
    parser.add_argument("--metric", type=str, default="correlation", choices=["correlation", "euclidean", "cosine"])
    parser.add_argument("--aggregation", type=str, default="hyperalignment", choices=["hyperalignment", "mean", "median", "stimulus_mean"])
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--features", type=int)
    parser.add_argument("--comparison-method", type=str, default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    characteristics_dir = Path(args.characteristics_dir)
    output_root = Path(args.output_root) / args.task
    output_root.mkdir(parents=True, exist_ok=True)

    mask_dir = output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for roi in args.rois:
        roi_label = roi_label_from_ids(roi, atlas_version="SPM12")
        roi_output_dir = output_root / roi_label
        roi_output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = mask_dir / f"{roi_label}.nii.gz"

        mask_cmd = build_mask_command(
            roi=roi,
            output_mask=mask_path,
            data_dir=data_dir,
            task=args.task,
            aal_version="SPM12",
        )

        pipeline_cmd = build_pipeline_command(
            data_dir=data_dir,
            output_dir=roi_output_dir,
            task=args.task,
            characteristics_dir=characteristics_dir,
            mask_path=mask_path,
            subjects=args.subjects,
            sessions=args.sessions,
            smoothing_fwhm=args.smoothing_fwhm,
            high_pass=args.high_pass,
            metric=args.metric,
            aggregation=args.aggregation,
            n_iter=args.n_iter,
            features=args.features,
            comparison_method=args.comparison_method,
        )

        print("\n" + "#" * 70)
        print(f"ROI: {roi_label}")
        print("#" * 70)

        if args.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(mask_cmd)}")
            print(f"[DRY RUN] Would execute: {' '.join(pipeline_cmd)}")
            continue

        run_command(mask_cmd, f"Build ROI mask for {roi_label}")
        run_command(pipeline_cmd, f"Run analysis pipeline for {roi_label}")


if __name__ == "__main__":
    main()