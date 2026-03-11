"""
Download BOLD fMRI files for all subjects from OpenNeuro.

This script downloads the actual BOLD .nii.gz files for all subjects
in the dataset, handling git-annex symlinks.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def get_annex_url(symlink_target: str, base_url: str = "https://github.com/OpenNeuroDatasets/ds003604/raw/main") -> str:
    """
    Convert git-annex symlink to download URL.
    
    Parameters
    ----------
    symlink_target : str
        The target of the symlink (e.g., '../../../.git/annex/objects/...')
    base_url : str
        Base URL for the dataset
    
    Returns
    -------
    str
        Direct download URL
    """
    if "annex/objects" in symlink_target:
        annex_path = symlink_target.split("annex/objects/")[1]
        return f"{base_url}/.git/annex/objects/{annex_path}"
    return None


def download_file(url: str, output_path: Path, chunk_size: int = 8192, max_retries: int = 3):
    """
    Download a file with progress bar and retry logic.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Where to save the file
    chunk_size : int
        Size of chunks to download
    max_retries : int
        Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=output_path.name,
                leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


def find_all_subjects(data_dir: Path) -> list:
    """Find all subject directories."""
    subjects = sorted([d.name for d in data_dir.glob("sub-*") if d.is_dir()])
    return subjects


def find_bold_files(
    data_dir: Path,
    subjects: list = None,
    task: str = "Sem",
    sessions: list = None
) -> list:
    """
    Find all BOLD files to download.
    
    Parameters
    ----------
    data_dir : Path
        Dataset directory
    subjects : list, optional
        Subject IDs to include (default: all)
    task : str
        Task name to filter
    sessions : list, optional
        Sessions to include (default: all)
        
    Returns
    -------
    list
        List of BOLD file paths
    """
    if subjects is None:
        subjects = find_all_subjects(data_dir)
    
    bold_files = []
    
    for subject_id in subjects:
        subject_path = data_dir / subject_id
        
        if not subject_path.exists():
            continue
        
        # Find BOLD files
        if sessions:
            for session in sessions:
                pattern = f"{session}/func/*task-{task}*_bold.nii.gz"
                bold_files.extend(subject_path.glob(pattern))
        else:
            pattern = f"ses-*/func/*task-{task}*_bold.nii.gz"
            bold_files.extend(subject_path.glob(pattern))
    
    return sorted(bold_files)


def download_bold_file(bold_file: Path, data_dir: Path) -> dict:
    """
    Download a single BOLD file.
    
    Returns
    -------
    dict
        Result information
    """
    result = {
        "file": str(bold_file.relative_to(data_dir)),
        "status": "unknown",
        "message": ""
    }
    
    # Check if already downloaded
    if not bold_file.is_symlink():
        result["status"] = "skipped"
        result["message"] = "Already downloaded"
        return result
    
    # Get symlink target
    target = os.readlink(str(bold_file))
    
    # Construct download URL
    url = get_annex_url(target)
    
    if not url:
        result["status"] = "error"
        result["message"] = "Could not construct URL"
        return result
    
    try:
        # Download to temporary location
        temp_file = bold_file.with_suffix('.nii.gz.tmp')
        download_file(url, temp_file)
        
        # Replace symlink with actual file
        bold_file.unlink()
        temp_file.rename(bold_file)
        
        result["status"] = "success"
        result["message"] = "Downloaded"
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        
        # Clean up temp file
        temp_file = bold_file.with_suffix('.nii.gz.tmp')
        if temp_file.exists():
            temp_file.unlink()
    
    return result


def batch_download(
    data_dir: str = "data/brain/ds003604",
    subjects: list = None,
    task: str = "Sem",
    sessions: list = None,
    max_workers: int = 4,
    dry_run: bool = False
):
    """
    Download BOLD files for multiple subjects.
    
    Parameters
    ----------
    data_dir : str
        Path to the dataset directory
    subjects : list, optional
        Subject IDs to include (default: all)
    task : str
        Task name to filter
    sessions : list, optional
        Sessions to include (default: all)
    max_workers : int
        Number of parallel download workers
    dry_run : bool
        If True, only list files without downloading
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Dataset directory not found: {data_path}")
        return
    
    # Find all BOLD files
    print("Scanning for BOLD files...")
    bold_files = find_bold_files(data_path, subjects=subjects, task=task, sessions=sessions)
    
    if not bold_files:
        print("No BOLD files found")
        return
    
    print(f"\nFound {len(bold_files)} BOLD files")
    print("=" * 70)
    
    # Check which files need downloading
    files_to_download = [f for f in bold_files if f.is_symlink()]
    files_already_downloaded = len(bold_files) - len(files_to_download)
    
    print(f"Already downloaded: {files_already_downloaded}")
    print(f"Need to download: {len(files_to_download)}")
    
    if dry_run:
        print("\nDRY RUN - Files to download:")
        for f in files_to_download:
            print(f"  {f.relative_to(data_path)}")
        return
    
    if not files_to_download:
        print("\nAll files already downloaded!")
        return
    
    # Download files
    print(f"\nDownloading {len(files_to_download)} files (workers: {max_workers})...")
    print("=" * 70)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_file = {
            executor.submit(download_bold_file, f, data_path): f
            for f in files_to_download
        }
        
        # Process completed downloads
        for future in tqdm(as_completed(future_to_file), total=len(files_to_download)):
            result = future.result()
            results.append(result)
            
            if result["status"] == "success":
                tqdm.write(f"✓ {result['file']}")
            elif result["status"] == "error":
                tqdm.write(f"✗ {result['file']}: {result['message']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print(f"Total files: {len(bold_files)}")
    print(f"Already downloaded: {files_already_downloaded}")
    print(f"Successfully downloaded: {success}")
    print(f"Errors: {errors}")
    
    if errors > 0:
        print("\nFiles with errors:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['file']}: {r['message']}")
    
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download BOLD fMRI files from OpenNeuro")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/brain/ds003604",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to download (default: all)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Sem",
        help="Task name to filter (default: Sem)"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=["ses-5", "ses-7", "ses-9"],
        help="Sessions to download (default: all)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading"
    )
    
    args = parser.parse_args()
    
    batch_download(
        data_dir=args.data_dir,
        subjects=args.subjects,
        task=args.task,
        sessions=args.sessions,
        max_workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
