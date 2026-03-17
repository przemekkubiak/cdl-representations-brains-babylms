"""
Analysis pipeline for ds003604.

Runs preprocessing + RSA only.
"""

import sys
import argparse
import subprocess
from pathlib import Path


DEFAULT_LANGUAGE_ROIS = [7, 8, 9, 10, 11, 12, 67, 68, 69, 70, 85, 86]


def run_command(cmd: list, description: str):
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)


def find_reference_bold(data_dir: str):
    data_path = Path(data_dir)
    matches = sorted(data_path.glob("sub-*/ses-*/func/*task-Sem*_bold.nii.gz"))
    if matches:
        return str(matches[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing + RSA only")
    parser.add_argument("--data-dir", type=str, default="data/brain/ds003604")
    parser.add_argument("--output-dir", type=str, default="data/processed/fmri")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--sessions", nargs="+", choices=["ses-5", "ses-7", "ses-9"])

    parser.add_argument("--smoothing-fwhm", type=float, default=6.0)
    parser.add_argument("--high-pass", type=float, default=0.01)

    parser.add_argument("--mask-path", type=str)
    parser.add_argument(
        "--aal-rois",
        nargs="+",
        type=int,
        help="Optional override for hard-coded AAL ROI list"
    )
    parser.add_argument("--aal-version", type=str, default="SPM12")
    parser.add_argument("--generated-mask-path", type=str)

    parser.add_argument("--metric", type=str, default="correlation", choices=["correlation", "euclidean", "cosine"])
    parser.add_argument("--aggregation", type=str, default="hyperalignment", choices=["hyperalignment", "mean", "median"])
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--features", type=int)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists() and not args.dry_run:
        print(f"ERROR: data directory not found: {args.data_dir}")
        print("Run: python run_download.py")
        sys.exit(1)

    if args.mask_path and args.aal_rois:
        print("ERROR: Use either --mask-path or --aal-rois, not both.")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mask_path_for_preprocess = args.mask_path
    selected_aal_rois = args.aal_rois if args.aal_rois else DEFAULT_LANGUAGE_ROIS

    if not args.mask_path:
        generated_mask_path = args.generated_mask_path or str(Path(args.output_dir) / "language_mask_aal.nii.gz")
        cmd = [
            sys.executable,
            "src/preprocessing/prepare_language_mask.py",
            "--aal-rois",
        ] + [str(x) for x in selected_aal_rois] + [
            "--aal-version", args.aal_version,
            "--output-mask", generated_mask_path,
        ]

        print(f"Using AAL ROIs: {selected_aal_rois}")

        reference_bold = find_reference_bold(args.data_dir)
        if reference_bold:
            cmd.extend(["--reference-bold", reference_bold])

        if args.dry_run:
            print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}")
        else:
            run_command(cmd, "STEP 1: Building language mask from AAL ROIs")

        mask_path_for_preprocess = generated_mask_path

    preprocess_cmd = [
        sys.executable,
        "src/preprocessing/batch_preprocessing.py",
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--smoothing-fwhm", str(args.smoothing_fwhm),
        "--high-pass", str(args.high_pass),
    ]

    if mask_path_for_preprocess:
        preprocess_cmd.extend(["--mask-path", mask_path_for_preprocess])
    if args.subjects:
        preprocess_cmd.extend(["--subjects"] + args.subjects)
    if args.sessions:
        preprocess_cmd.extend(["--sessions"] + args.sessions)

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute: {' '.join(preprocess_cmd)}")
    else:
        run_command(preprocess_cmd, "STEP 2: Preprocessing fMRI data")

    rsa_cmd = [
        sys.executable,
        "src/rsa/session_based_rsa.py",
        "--pattern-dir", args.output_dir,
        "--output-dir", args.output_dir,
        "--metric", args.metric,
        "--aggregation", args.aggregation,
    ]

    if args.aggregation == "hyperalignment":
        rsa_cmd.extend(["--n-iter", str(args.n_iter)])
        if args.features:
            rsa_cmd.extend(["--features", str(args.features)])
    if args.subjects:
        rsa_cmd.extend(["--subjects"] + args.subjects)
    if args.sessions:
        rsa_cmd.extend(["--sessions"] + args.sessions)

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute: {' '.join(rsa_cmd)}")
    else:
        run_command(rsa_cmd, "STEP 3: Session-based RSA")

    print("\nAnalysis pipeline complete.")


if __name__ == "__main__":
    main()
