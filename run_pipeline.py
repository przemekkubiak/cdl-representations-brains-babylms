"""
Full pipeline orchestration script.

Runs the complete analysis pipeline:
1. Download BOLD fMRI data for all subjects
2. Preprocess fMRI data (GLM, pattern extraction)
3. Compute session-based RDMs (ses-5, ses-7, ses-9)
"""

import sys
import argparse
import subprocess
from pathlib import Path
import json
from datetime import datetime


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")


def check_dependencies():
    """Check that required Python packages are installed."""
    try:
        import numpy
        import pandas
        import scipy
        import nibabel
        import nilearn
        import matplotlib
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run full fMRI RSA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for all subjects
  python run_pipeline.py
  
  # Run specific steps
  python run_pipeline.py --steps download preprocess
  
  # Run for specific subjects and sessions
  python run_pipeline.py --subjects sub-5007 sub-5008 --sessions ses-7 ses-9
  
  # Dry run (show what would be executed)
  python run_pipeline.py --dry-run
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/brain/ds003604",
        help="Path to BIDS dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/fmri",
        help="Path to save processed data"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (default: all)"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=["ses-5", "ses-7", "ses-9"],
        help="Sessions to process (default: all)"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["download", "preprocess", "rsa"],
        default=["download", "preprocess", "rsa"],
        help="Pipeline steps to run (default: all)"
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=6.0,
        help="Spatial smoothing FWHM in mm (default: 6.0)"
    )
    parser.add_argument(
        "--high-pass",
        type=float,
        default=0.01,
        help="High-pass filter cutoff in Hz (default: 0.01)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="correlation",
        choices=["correlation", "euclidean", "cosine"],
        help="Distance metric for RSA (default: correlation)"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="hyperalignment",
        choices=["hyperalignment", "mean", "median"],
        help="Aggregation method for session RDMs (default: hyperalignment)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of SRM iterations for hyperalignment (default: 10)"
    )
    parser.add_argument(
        "--features",
        type=int,
        help="Number of shared features for hyperalignment (default: auto)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency checks"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("fMRI RSA PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Steps: {', '.join(args.steps)}")
    if args.subjects:
        print(f"Subjects: {', '.join(args.subjects)}")
    if args.sessions:
        print(f"Sessions: {', '.join(args.sessions)}")
    print("="*70)
    
    # Check dependencies
    if not args.skip_checks and not args.dry_run:
        print("\nChecking dependencies...")
        if not check_dependencies():
            sys.exit(1)
    
    # Create output directories
    if not args.dry_run:
        Path("logs").mkdir(exist_ok=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download data
    if "download" in args.steps:
        cmd = [
            sys.executable,
            "scripts/batch_download_bold.py",
            "--data-dir", args.data_dir,
            "--task", "Sem",
            "--workers", str(args.download_workers)
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
            run_command(cmd, "STEP 1: Downloading BOLD data")
    
    # Step 2: Preprocess
    if "preprocess" in args.steps:
        cmd = [
            sys.executable,
            "src/preprocessing/batch_preprocessing.py",
            "--data-dir", args.data_dir,
            "--output-dir", args.output_dir,
            "--smoothing-fwhm", str(args.smoothing_fwhm),
            "--high-pass", str(args.high_pass)
        ]
        
        if args.subjects:
            cmd.extend(["--subjects"] + args.subjects)
        
        if args.sessions:
            cmd.extend(["--sessions"] + args.sessions)
        
        if args.dry_run:
            print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}")
        else:
            run_command(cmd, "STEP 2: Preprocessing fMRI data")
    
    # Step 3: Session-based RSA
    if "rsa" in args.steps:
        cmd = [
            sys.executable,
            "src/rsa/session_based_rsa.py",
            "--pattern-dir", args.output_dir,
            "--output-dir", args.output_dir,
            "--metric", args.metric,
            "--aggregation", args.aggregation
        ]
        
        if args.aggregation == "hyperalignment":
            cmd.extend(["--n-iter", str(args.n_iter)])
            if args.features:
                cmd.extend(["--features", str(args.features)])
        
        if args.subjects:
            cmd.extend(["--subjects"] + args.subjects)
        
        if args.sessions:
            cmd.extend(["--sessions"] + args.sessions)
        
        if args.dry_run:
            print(f"\n[DRY RUN] Would execute: {' '.join(cmd)}")
        else:
            run_command(cmd, "STEP 3: Session-based RSA analysis")
    
    # Summary
    print("\n" + "="*70)
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("PIPELINE COMPLETE!")
        print("="*70)
        print("\nOutput files:")
        print(f"  - Preprocessed patterns: {args.output_dir}/sub-*_patterns.npz")
        print(f"  - Session RDMs: {args.output_dir}/session_rdm_ses-*.npz")
        print(f"  - Visualizations: {args.output_dir}/session_rdm_ses-*.png")
        print(f"  - Comparison: {args.output_dir}/session_rdm_comparison.csv")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
