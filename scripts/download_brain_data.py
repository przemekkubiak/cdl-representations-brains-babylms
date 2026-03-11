"""
Script to download neuroimaging data from OpenNeuro.
Supports sparse checkout to download only specific folders.
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


def download_dataset_sparse(
    repo_url="https://github.com/OpenNeuroDatasets/ds003604.git",
    data_dir="data/brain",
    dataset_name="ds003604",
    folders=None,
    skip_confirmation=False
):
    """
    Download specific folders from neuroimaging dataset using sparse checkout.
    
    Parameters
    ----------
    repo_url : str
        Git repository URL
    data_dir : str
        Directory to download data to
    dataset_name : str
        Name of the dataset
    folders : list of str
        List of folders to download (e.g., ['stimuli', 'sub-5007'])
        If None, downloads entire dataset
    skip_confirmation : bool
        If True, skip user confirmation
    """
    print("=" * 60)
    print("OpenNeuro Dataset Download Script (Sparse Checkout)")
    print(f"Dataset: {dataset_name}")
    if folders:
        print(f"Folders to download: {', '.join(folders)}")
    else:
        print("Mode: Full dataset download")
    print("=" * 60)
    print()
    
    # Check git
    if not check_git_installed():
        print("Error: git is not installed. Please install git first.")
        return False
    
    # Check git version supports sparse checkout
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Git version: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("Warning: Could not determine git version")
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_path.absolute()}")
    
    # Check disk space
    free_space = get_disk_space()
    print(f"Available disk space: {free_space:.2f} GB")
    
    if free_space < 5:
        print("Warning: Less than 5 GB of free space available!")
    
    # Confirm download
    if not skip_confirmation:
        if folders:
            response = input(f"\nDownload {len(folders)} folder(s)? (y/n): ").strip().lower()
        else:
            response = input("\nThis dataset is large. Continue? (y/n): ").strip().lower()
        
        if response != 'y':
            print("Download cancelled.")
            return False
    
    # Setup target path
    target_path = data_path / dataset_name
    if target_path.exists():
        print(f"Warning: {target_path} already exists!")
        if not skip_confirmation:
            response = input("Delete and re-download? (y/n): ").strip().lower()
            if response != 'y':
                print("Download cancelled.")
                return False
        shutil.rmtree(target_path)
    
    print(f"\nSetting up repository at {target_path}")
    
    try:
        # Initialize git repository
        target_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init"],
            cwd=target_path,
            check=True,
            capture_output=True
        )
        
        # Add remote
        subprocess.run(
            ["git", "remote", "add", "origin", repo_url],
            cwd=target_path,
            check=True,
            capture_output=True
        )
        
        if folders:
            # Enable sparse checkout
            print("Enabling sparse checkout...")
            subprocess.run(
                ["git", "config", "core.sparseCheckout", "true"],
                cwd=target_path,
                check=True,
                capture_output=True
            )
            
            # Configure sparse checkout patterns
            sparse_checkout_file = target_path / ".git" / "info" / "sparse-checkout"
            with open(sparse_checkout_file, "w") as f:
                for folder in folders:
                    f.write(f"{folder}/\n")
                    print(f"  - {folder}/")
        
        # Pull from remote
        print(f"\nDownloading from {repo_url}")
        print("This may take a few minutes...")
        
        subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=target_path,
            check=True
        )
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print(f"Dataset location: {target_path.absolute()}")
        
        # Show what was downloaded
        if folders:
            print("\nDownloaded folders:")
            for folder in folders:
                folder_path = target_path / folder
                if folder_path.exists():
                    print(f"  ✓ {folder}")
                else:
                    print(f"  ✗ {folder} (not found)")
        
        print("=" * 60)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during download: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        return False


def download_dataset(
    repo_url="https://github.com/OpenNeuroDatasets/ds003604.git",
    data_dir="data/brain",
    dataset_name="ds003604"
):
    """
    Download entire neuroimaging dataset from OpenNeuro.
    
    Parameters
    ----------
    repo_url : str
        Git repository URL
    data_dir : str
        Directory to download data to
    dataset_name : str
        Name of the dataset
    """
    return download_dataset_sparse(
        repo_url=repo_url,
        data_dir=data_dir,
        dataset_name=dataset_name,
        folders=None
    )


if __name__ == "__main__":
    # Download specific folders: stimuli and sub-5007
    download_dataset_sparse(
        folders=["stimuli", "sub-5007"],
        skip_confirmation=False
    )
