"""
Script to download neuroimaging data from OpenNeuro.
This is a Python alternative to the bash script.
"""

import os
import subprocess
import shutil
from pathlib import Path


def check_git_installed():
    """Check if git is installed."""
    return shutil.which("git") is not None


def get_disk_space(path="."):
    """Get available disk space in GB."""
    stat = shutil.disk_usage(path)
    return stat.free / (1024**3)  # Convert to GB


def download_dataset(
    repo_url="https://github.com/OpenNeuroDatasets/ds003604.git",
    data_dir="data/brain",
    dataset_name="ds003604"
):
    """
    Download neuroimaging dataset from OpenNeuro.
    
    Parameters
    ----------
    repo_url : str
        Git repository URL
    data_dir : str
        Directory to download data to
    dataset_name : str
        Name of the dataset
    """
    print("=" * 50)
    print("OpenNeuro Dataset Download Script")
    print(f"Dataset: {dataset_name}")
    print("=" * 50)
    print()
    
    # Check git
    if not check_git_installed():
        print("Error: git is not installed. Please install git first.")
        return False
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_path.absolute()}")
    
    # Check disk space
    free_space = get_disk_space()
    print(f"Available disk space: {free_space:.2f} GB")
    
    if free_space < 10:
        print("Warning: Less than 10 GB of free space available!")
    
    # Confirm download
    response = input("\nThis dataset is large. Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return False
    
    # Clone repository
    target_path = data_path / dataset_name
    if target_path.exists():
        print(f"Warning: {target_path} already exists!")
        response = input("Delete and re-download? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(target_path)
        else:
            print("Download cancelled.")
            return False
    
    print(f"\nCloning dataset from {repo_url}")
    print("This may take a while...")
    
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(target_path)],
            check=True
        )
        print("\n" + "=" * 50)
        print("Download complete!")
        print(f"Dataset location: {target_path.absolute()}")
        print("=" * 50)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during download: {e}")
        return False


if __name__ == "__main__":
    download_dataset()
