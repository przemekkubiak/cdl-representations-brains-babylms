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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import get_dataset, DatasetSpec, UnresolvedDatasetError


def get_candidate_urls(bold_file: Path, data_dir: Path, spec: DatasetSpec) -> list:
    """Build candidate download URLs for a BOLD file.

    URL construction lives in the dataset registry (configs/neuro_datasets.yaml)
    so that pointing this script at another accession actually downloads that
    accession. It used to be hardcoded to ds003604 in three places, which made
    the cross-dataset arm impossible to run without fabricating data -- see
    PICKUP.md, "What was deliberately NOT run".
    """
    rel_path = bold_file.relative_to(data_dir).as_posix()
    annex_target = os.readlink(str(bold_file)) if bold_file.is_symlink() else None
    return spec.candidate_urls(rel_path, annex_target=annex_target)


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


def _normalize_tasks(task: str = None, tasks: list = None) -> list:
    """Return a deduplicated task list while preserving order."""
    resolved = []
    if tasks:
        resolved.extend(tasks)
    elif task:
        resolved.append(task)

    unique_tasks = []
    seen = set()
    for task_name in resolved:
        if task_name not in seen:
            unique_tasks.append(task_name)
            seen.add(task_name)

    return unique_tasks


def find_bold_files(
    data_dir: Path,
    subjects: list = None,
    task: str = None,
    tasks: list = None,
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
    task : str, optional
        Single task name to filter
    tasks : list, optional
        Multiple task names to filter
    sessions : list, optional
        Sessions to include (default: all)
        
    Returns
    -------
    list
        List of BOLD file paths
    """
    if subjects is None:
        subjects = find_all_subjects(data_dir)

    task_names = _normalize_tasks(task=task, tasks=tasks)
    if not task_names:
        raise ValueError("At least one task must be provided")
    
    bold_files = []
    
    for subject_id in subjects:
        subject_path = data_dir / subject_id
        
        if not subject_path.exists():
            continue
        
        # Find BOLD files for each requested task
        for task_name in task_names:
            if sessions:
                for session in sessions:
                    pattern = f"{session}/func/*task-{task_name}*_bold.nii.gz"
                    bold_files.extend(subject_path.glob(pattern))
            else:
                pattern = f"ses-*/func/*task-{task_name}*_bold.nii.gz"
                bold_files.extend(subject_path.glob(pattern))
    
    return sorted(bold_files)


def download_bold_file(bold_file: Path, data_dir: Path, spec: DatasetSpec) -> dict:
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
    
    candidate_urls = get_candidate_urls(bold_file, data_dir, spec)
    last_error = None

    for url in candidate_urls:
        try:
            temp_file = bold_file.with_suffix('.nii.gz.tmp')
            download_file(url, temp_file)

            # Replace symlink with actual file
            bold_file.unlink()
            temp_file.rename(bold_file)

            result["status"] = "success"
            result["message"] = f"Downloaded from {url}"
            return result

        except Exception as e:
            last_error = str(e)
            temp_file = bold_file.with_suffix('.nii.gz.tmp')
            if temp_file.exists():
                temp_file.unlink()

    result["status"] = "error"
    result["message"] = f"All URL candidates failed. Last error: {last_error}"
    
    return result


def batch_download(
    dataset: str = "ds003604",
    data_dir: str = None,
    subjects: list = None,
    task: str = None,
    tasks: list = None,
    sessions: list = None,
    max_workers: int = 4,
    dry_run: bool = False
):
    """
    Download BOLD files for multiple subjects.
    
    Parameters
    ----------
    dataset : str
        Registry key or OpenNeuro accession (configs/neuro_datasets.yaml)
    data_dir : str, optional
        Path to the dataset directory (default: data/brain/<accession>)
    subjects : list, optional
        Subject IDs to include (default: all)
    task : str, optional
        Single task name to filter
    tasks : list, optional
        Multiple task names to filter
    sessions : list, optional
        Sessions to include (default: all)
    max_workers : int
        Number of parallel download workers
    dry_run : bool
        If True, only list files without downloading
    """
    spec = get_dataset(dataset)
    spec.require_downloadable()
    data_path = Path(data_dir) if data_dir else spec.data_dir()
    task_names = _normalize_tasks(task=task, tasks=tasks)

    print(f"Dataset: {spec.accession} -- {spec.name}")
    if spec.needs_within_run_norm:
        print(
            f"  NOTE: run/stimulus structure is '{spec.run_stimulus}'. Voxel patterns "
            "must be normalised WITHIN RUN before aggregating across runs, or the "
            "resulting RDMs measure acquisition structure rather than language."
        )
    
    if not data_path.exists():
        print(f"Error: Dataset directory not found: {data_path}")
        return
    
    # Find all BOLD files
    print("Scanning for BOLD files...")
    bold_files = find_bold_files(data_path, subjects=subjects, tasks=task_names, sessions=sessions)
    
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
            executor.submit(download_bold_file, f, data_path, spec): f
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
        "--dataset",
        type=str,
        default="ds003604",
        help="Registry key or OpenNeuro accession (see configs/neuro_datasets.yaml)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset directory (default: data/brain/<accession>)"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to download (default: all)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Single task name to filter (legacy; use --tasks for multiple tasks)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task names to filter (e.g. Sem Phon Gram Plaus)"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="Sessions to download (default: all). Valid values are dataset-specific."
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
    
    spec = get_dataset(args.dataset)
    # Surface an unresolved accession before anything else: there is no safe
    # fallback, because falling back would download a different study into a
    # directory named for this one.
    try:
        spec.require_downloadable()
    except UnresolvedDatasetError as e:
        parser.error(str(e))

    if args.tasks:
        task_names = args.tasks
    elif args.task:
        task_names = [args.task]
    elif spec.tasks:
        task_names = spec.tasks
    else:
        parser.error(
            f"dataset '{spec.key}' declares no tasks in configs/neuro_datasets.yaml; "
            "pass --tasks explicitly (and record them in the registry)"
        )

    if spec.sessions and args.sessions:
        unknown = [s_ for s_ in args.sessions if s_ not in spec.sessions]
        if unknown:
            parser.error(
                f"sessions {unknown} are not in {spec.key}; known: {spec.sessions}"
            )

    batch_download(
        dataset=args.dataset,
        data_dir=args.data_dir,
        subjects=args.subjects,
        task=task_names[0] if len(task_names) == 1 else None,
        tasks=task_names,
        sessions=args.sessions,
        max_workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
