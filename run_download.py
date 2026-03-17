"""
Download pipeline for ds003604.

Runs only data bootstrap + BOLD download.
"""

import sys
import argparse
import subprocess
from pathlib import Path
import shutil


def run_command(cmd: list, description: str):
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)


def ensure_dataset_checkout(
    data_dir: str,
    repo_url: str = "https://github.com/OpenNeuroDatasets/ds003604.git",
    branch: str = "main",
    dry_run: bool = False,
):
    data_path = Path(data_dir)

    if data_path.exists() and any(data_path.glob("sub-*")):
        print(f"Dataset checkout found: {data_path}")
        return

    if dry_run:
        print(f"[DRY RUN] Would clone dataset metadata to: {data_path}")
        return

    if shutil.which("git") is None:
        print("ERROR: git is required to clone dataset metadata but was not found.")
        sys.exit(1)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = [
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--single-branch",
        "--branch", branch,
        repo_url,
        str(data_path),
    ]
    run_command(clone_cmd, "STEP 1: Cloning dataset metadata")


def main():
    parser = argparse.ArgumentParser(description="Download ds003604 data only")
    parser.add_argument("--data-dir", type=str, default="data/brain/ds003604")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--sessions", nargs="+", choices=["ses-5", "ses-7", "ses-9"])
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--dataset-repo-url", type=str, default="https://github.com/OpenNeuroDatasets/ds003604.git")
    parser.add_argument("--dataset-branch", type=str, default="main")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    ensure_dataset_checkout(
        data_dir=args.data_dir,
        repo_url=args.dataset_repo_url,
        branch=args.dataset_branch,
        dry_run=args.dry_run,
    )

    cmd = [
        sys.executable,
        "scripts/batch_download_bold.py",
        "--data-dir", args.data_dir,
        "--task", "Sem",
        "--workers", str(args.download_workers),
    ]

    if args.subjects:
        cmd.extend(["--subjects"] + args.subjects)
    if args.sessions:
        cmd.extend(["--sessions"] + args.sessions)
    if args.dry_run:
        cmd.append("--dry-run")

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}")
    else:
        run_command(cmd, "STEP 2: Downloading BOLD files")

    print("\nDownload pipeline complete.")


if __name__ == "__main__":
    main()
