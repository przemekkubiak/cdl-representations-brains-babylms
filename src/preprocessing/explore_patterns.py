"""
Explore the extracted fMRI patterns.

This script loads and examines the preprocessed stimulus-specific brain activity patterns.
"""

import numpy as np
from pathlib import Path


def explore_patterns(pattern_file: str):
    """
    Load and explore a patterns file.
    
    Parameters
    ----------
    pattern_file : str
        Path to .npz file containing stimulus patterns
    """
    print(f"Loading: {pattern_file}")
    print("=" * 70)
    
    # Load the patterns
    data = np.load(pattern_file)
    
    # Get list of stimuli
    stimuli = list(data.keys())
    print(f"\nNumber of stimuli: {len(stimuli)}")
    print(f"\nFirst 10 stimuli:")
    for i, stim in enumerate(stimuli[:10], 1):
        pattern = data[stim]
        print(f"  {i:2d}. {stim:50s} | shape: {pattern.shape} | mean: {pattern.mean():.4f}")
    
    # Show example pattern statistics
    example_stim = stimuli[0]
    example_pattern = data[example_stim]
    
    print(f"\nExample pattern: {example_stim}")
    print(f"  Shape: {example_pattern.shape}")
    print(f"  Number of voxels: {example_pattern.shape[0]:,}")
    print(f"  Mean activity: {example_pattern.mean():.4f}")
    print(f"  Std activity: {example_pattern.std():.4f}")
    print(f"  Min activity: {example_pattern.min():.4f}")
    print(f"  Max activity: {example_pattern.max():.4f}")
    
    # Compute pattern similarities (quick example)
    print(f"\nPattern similarity analysis (first 5 stimuli):")
    patterns_matrix = np.array([data[stim] for stim in stimuli[:5]])
    
    # Compute correlation matrix
    from scipy.stats import pearsonr
    n = len(stimuli[:5])
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = pearsonr(patterns_matrix[i], patterns_matrix[j])[0]
    
    print("\nCorrelation matrix:")
    print("     ", "  ".join([f"S{i+1:2d}" for i in range(n)]))
    for i in range(n):
        print(f"S{i+1:2d}  ", "  ".join([f"{corr_matrix[i, j]:4.2f}" for j in range(n)]))
    
    print("\n" + "=" * 70)


def main():
    """Explore all pattern files."""
    pattern_dir = Path("data/processed/fmri")
    pattern_files = sorted(pattern_dir.glob("*.npz"))
    
    if not pattern_files:
        print("No pattern files found in data/processed/fmri/")
        return
    
    print(f"\nFound {len(pattern_files)} pattern files\n")
    
    # Explore first file in detail
    explore_patterns(str(pattern_files[0]))
    
    # Summary of all files
    print("\n" + "=" * 70)
    print("Summary of all pattern files:")
    print("=" * 70)
    
    for pf in pattern_files:
        data = np.load(str(pf))
        stimuli = list(data.keys())
        example_shape = data[stimuli[0]].shape
        print(f"{pf.name:50s} | {len(stimuli)} stimuli | {example_shape[0]:,} voxels")


if __name__ == "__main__":
    main()
