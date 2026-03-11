"""
Download BOLD fMRI files from OpenNeuro using datalad.

Datalad is the recommended tool for working with OpenNeuro datasets.
"""

import subprocess
from pathlib import Path
import sys


def download_with_datalad(
    data_dir: str = "data/brain/ds003604",
    subject_id: str = "sub-5007",
    task: str = "Sem"
):
    """
    Download BOLD files using datalad get command.
    
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
    
    if not data_path.exists():
        print(f"Error: Dataset directory not found: {data_path}")
        return False
    
    print("Downloading BOLD files with datalad...")
    print("=" * 60)
    
    # Pattern to match all Semantic task BOLD files for this subject
    pattern = f"{subject_id}/ses-*/func/*task-{task}*_bold.nii.gz"
    
    try:
        # Run datalad get command
        cmd = ["datalad", "get", pattern]
        print(f"Running: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(
            cmd,
            cwd=str(data_path),
            check=True,
            capture_output=False,
            text=True
        )
        
        print("\n" + "=" * 60)
        print("Download complete!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}")
        return False
    except FileNotFoundError:
        print("\nError: datalad command not found.")
        print("Please install datalad: pip install datalad")
        return False


if __name__ == "__main__":
    success = download_with_datalad()
    sys.exit(0 if success else 1)
