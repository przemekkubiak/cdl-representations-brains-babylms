"""
Analyze and summarize neural RSA results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.rsa.semantic_metadata import semantic_categories_from_trial_types


def load_neural_rdm(filepath: str):
    """Load saved neural RDM."""
    data = np.load(filepath, allow_pickle=True)
    metadata = {}

    for key in [
        "stimuli",
        "n_subjects",
        "subject_ids",
        "metric",
        "aggregation",
        "trial_types",
        "semantic_categories",
    ]:
        if key in data.files:
            metadata[key] = data[key]

    return data["rdm"], data["stimuli"], metadata


def _categories_from_metadata(stimuli: list, metadata: dict):
    """Infer semantic categories from saved metadata when available."""
    trial_types = metadata.get("trial_types")
    if trial_types is not None and len(trial_types) == len(stimuli):
        return semantic_categories_from_trial_types(trial_types), "trial_types"

    semantic_categories = metadata.get("semantic_categories")
    if semantic_categories is not None and len(semantic_categories) == len(stimuli):
        return [str(category) for category in semantic_categories], "semantic_categories"

    return None, None


def analyze_rdm_structure(rdm: np.ndarray, stimuli: list):
    """
    Analyze the structure of an RDM.
    
    Parameters
    ----------
    rdm : np.ndarray
        Representational Dissimilarity Matrix
    stimuli : list
        Stimulus names
    """
    print("\n" + "=" * 70)
    print("RDM Structure Analysis")
    print("=" * 70)
    
    triu_idx = np.triu_indices_from(rdm, k=1)
    
    print(f"\nNumber of stimuli: {len(stimuli)}")
    print(f"Number of unique pairwise dissimilarities: {len(rdm[triu_idx])}")
    
    # Most similar pairs
    print(f"\nMost similar stimulus pairs (lowest dissimilarity):")
    flat_idx = np.argsort(rdm[triu_idx])[:5]
    for i, idx in enumerate(flat_idx, 1):
        row, col = np.unravel_index(np.ravel_multi_index(
            ([triu_idx[0][idx]], [triu_idx[1][idx]]), rdm.shape
        ), rdm.shape)
        stim1 = Path(stimuli[row[0]]).stem
        stim2 = Path(stimuli[col[0]]).stem
        dissim = rdm[row[0], col[0]]
        print(f"  {i}. {stim1} <-> {stim2}: {dissim:.4f}")
    
    # Most dissimilar pairs
    print(f"\nMost dissimilar stimulus pairs (highest dissimilarity):")
    flat_idx = np.argsort(rdm[triu_idx])[-5:][::-1]
    for i, idx in enumerate(flat_idx, 1):
        row, col = np.unravel_index(np.ravel_multi_index(
            ([triu_idx[0][idx]], [triu_idx[1][idx]]), rdm.shape
        ), rdm.shape)
        stim1 = Path(stimuli[row[0]]).stem
        stim2 = Path(stimuli[col[0]]).stem
        dissim = rdm[row[0], col[0]]
        print(f"  {i}. {stim1} <-> {stim2}: {dissim:.4f}")





def analyze_stimulus_categories(rdm: np.ndarray, stimuli: list):
    """
    Analyze dissimilarities by stimulus category (if encoded in filename).
    
    Parameters
    ----------
    rdm : np.ndarray
        RDM
    stimuli : list
        Stimulus names
    """
    print("\n" + "=" * 70)
    print("Stimulus Category Analysis")
    print("=" * 70)
    
    categories = None
    category_source = None

    if isinstance(stimuli, tuple) and len(stimuli) == 2:
        stim_values, metadata = stimuli
        categories, category_source = _categories_from_metadata(list(stim_values), metadata)
        stimuli = list(stim_values)

    if categories is None:
        # Fallback: infer categories from stimulus names only.
        categories = []
        for stim in stimuli:
            stem = Path(stim).stem
            if 'Cont' in stem:
                categories.append('control')
            elif 'SU' in stem:
                categories.append('unrelated')
            elif 'SH' in stem:
                categories.append('high_association')
            elif 'SL' in stem:
                categories.append('low_association')
            elif 'SC' in stem:
                categories.append('control')
            else:
                categories.append('unknown')
    
    # Count categories
    from collections import Counter
    cat_counts = Counter(categories)
    
    if category_source:
        print(f"\nStimulus categories found (source: {category_source}):")
    else:
        print(f"\nStimulus categories found (source: filename heuristic):")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} stimuli")
    
    # Compute within vs between category dissimilarities
    within_cat = []
    between_cat = []
    
    for i in range(len(stimuli)):
        for j in range(i + 1, len(stimuli)):
            dissim = rdm[i, j]
            if categories[i] == categories[j]:
                within_cat.append(dissim)
            else:
                between_cat.append(dissim)
    
    if within_cat and between_cat:
        print(f"\nWithin-category dissimilarity:")
        print(f"  Mean: {np.mean(within_cat):.4f} (±{np.std(within_cat):.4f})")
        print(f"  N pairs: {len(within_cat)}")
        
        print(f"\nBetween-category dissimilarity:")
        print(f"  Mean: {np.mean(between_cat):.4f} (±{np.std(between_cat):.4f})")
        print(f"  N pairs: {len(between_cat)}")
        
        # Statistical test
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(within_cat, between_cat)
        print(f"\nMann-Whitney U test:")
        print(f"  Statistic: {stat:.2f}")
        print(f"  P-value: {pval:.4e}")
        
        if pval < 0.05:
            print(f"  Result: Significant difference (p < 0.05)")
        else:
            print(f"  Result: No significant difference (p >= 0.05)")


def main():
    """Run RSA analysis summary."""
    print("=" * 70)
    print("Neural RSA Analysis Summary")
    print("=" * 70)
    
    # Load averaged RDM
    rdm_path = "data/processed/fmri/neural_rdm_averaged.npz"
    
    if not Path(rdm_path).exists():
        print(f"\nError: {rdm_path} not found")
        print("Run neural_rsa.py first to compute RDMs")
        return
    
    rdm, stimuli, metadata = load_neural_rdm(rdm_path)
    
    print(f"\nLoaded neural RDM from: {rdm_path}")
    if metadata:
        for key, val in metadata.items():
            print(f"  {key}: {val}")
    
    # Analyze RDM structure
    analyze_rdm_structure(rdm, list(stimuli))
    
    # Analyze by stimulus categories
    analyze_stimulus_categories(rdm, (list(stimuli), metadata))
    
    # Load and summarize comparison results
    comparison_path = "data/processed/fmri/rdm_comparison.csv"
    if Path(comparison_path).exists():
        print("\n" + "=" * 70)
        print("Within-Subject RDM Reliability")
        print("=" * 70)
        
        df = pd.read_csv(comparison_path)
        print(f"\nRDM correlations across runs:")
        print(df.to_string(index=False))
        
        print(f"\nSummary:")
        print(f"  Mean correlation: {df['correlation'].mean():.4f}")
        print(f"  Std correlation: {df['correlation'].std():.4f}")
        print(f"  Range: [{df['correlation'].min():.4f}, {df['correlation'].max():.4f}]")
        
        print("\nNote: Low correlations are expected for fMRI due to noise.")
        print("Averaging across runs improves signal-to-noise ratio.")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    print("\nGenerated files:")
    output_dir = Path("data/processed/fmri")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f}")
    for f in sorted(output_dir.glob("*.npz")):
        print(f"  - {f}")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
