"""
Download BOLD fMRI files from OpenNeuro using direct HTTP downloads.

This script downloads the actual BOLD .nii.gz files that are stored
as git-annex objects.
"""

import os
import sys
from pathlib import Path
import hashlib
import requests
from tqdm import tqdm


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
    # Extract the annex path from symlink
    if "annex/objects" in symlink_target:
        annex_path = symlink_target.split("annex/objects/")[1]
        return f"{base_url}/.git/annex/objects/{annex_path}"
    return None


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """
    Download a file with progress bar.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Where to save the file
    chunk_size : int
        Size of chunks to download
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=output_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_bold_files(
    data_dir: str = "data/brain/ds003604",
    subject_id: str = "sub-5007",
    task: str = "Sem"
):
    """
    Download all BOLD files for a subject and task.
    
    Parameters
    ----------
    data_dir : str
        Path to the dataset directory
    subject_id : str
        Subject identifier
    task : str
        Task name to filter files
    """
    data_path = Path(data_dir)
    subject_path = data_path / subject_id
    
    if not subject_path.exists():
        print(f"Error: Subject directory not found: {subject_path}")
        return
    
    # Find all BOLD files (symlinks)
    bold_files = list(subject_path.glob(f"ses-*/func/*task-{task}*_bold.nii.gz"))
    
    if not bold_files:
        print(f"No BOLD files found for task {task}")
        return
    
    print(f"Found {len(bold_files)} BOLD files to download")
    print("=" * 60)
    
    for bold_file in bold_files:
        print(f"\nProcessing: {bold_file.relative_to(data_path)}")
        
        # Check if it's a symlink
        if not bold_file.is_symlink():
            print(f"  Already downloaded (not a symlink)")
            continue
        
        # Get symlink target
        target = os.readlink(str(bold_file))
        print(f"  Symlink target: {target}")
        
        # Construct download URL
        url = get_annex_url(target)
        
        if not url:
            print(f"  Could not construct URL from symlink")
            continue
        
        print(f"  Downloading from: {url}")
        
        try:
            # Download to temporary location first
            temp_file = bold_file.with_suffix('.nii.gz.tmp')
            download_file(url, temp_file)
            
            # Remove symlink and replace with actual file
            bold_file.unlink()
            temp_file.rename(bold_file)
            
            print(f"  ✓ Downloaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error downloading: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    print("\n" + "=" * 60)
    print("Download complete!")


if __name__ == "__main__":
    download_bold_files()
