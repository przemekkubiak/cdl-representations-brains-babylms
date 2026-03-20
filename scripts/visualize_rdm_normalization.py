#!/usr/bin/env python
"""
Visualize brain RDM, language model RDM (raw), and language model RDM (z-normalized).
Useful for comparing how z-normalization affects the structure.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rsa import z_normalize_rdm


def load_rdm(filepath: str) -> np.ndarray:
    """Load RDM from .npz file."""
    data = np.load(filepath)
    return data["rdm"]


def visualize_rdm_comparison(
    brain_session: str = "ses-7",
    lm_rdm_path: str = None,
    brain_rdm_dir: str = "data/processed/fmri",
    output_path: str = None,
    vmin: float = None,
    vmax: float = None,
):
    """
    Create 3-panel visualization: brain RDM, LM RDM (raw), LM RDM (normalized).
    
    Args:
        brain_session: Which brain session to load (ses-5, ses-7, ses-9)
        lm_rdm_path: Path to language model RDM .npz file
        brain_rdm_dir: Directory containing brain RDMs
        output_path: Where to save the figure (if None, displays)
        vmin, vmax: Colormap limits (if None, computed from data)
    """
    # Load RDMs
    brain_rdm_file = Path(brain_rdm_dir) / f"session_rdm_{brain_session}.npz"
    if not brain_rdm_file.exists():
        raise FileNotFoundError(f"Brain RDM not found: {brain_rdm_file}")
    
    brain_rdm = load_rdm(str(brain_rdm_file))
    
    if lm_rdm_path is None:
        raise ValueError("Must provide --lm-rdm-path")
    
    lm_rdm_file = Path(lm_rdm_path)
    if not lm_rdm_file.exists():
        raise FileNotFoundError(f"LM RDM not found: {lm_rdm_file}")
    
    lm_rdm = load_rdm(str(lm_rdm_file))
    lm_rdm_normalized = z_normalize_rdm(lm_rdm)
    
    # Get LM model name from filename
    lm_name = lm_rdm_file.stem.replace("lm_rdm_", "").replace("_Sem_layer-1", "")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Set common colormap limits
    if vmin is None:
        vmin = min(brain_rdm.min(), lm_rdm.min())
    if vmax is None:
        vmax = max(brain_rdm.max(), lm_rdm.max())
    
    # Brain RDM
    sns.heatmap(
        brain_rdm,
        ax=axes[0],
        cbar_kws={"label": "Dissimilarity"},
        cmap="viridis",
        square=True,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"Brain RDM ({brain_session})\n{brain_rdm.shape}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Stimulus")
    axes[0].set_ylabel("Stimulus")
    
    # LM RDM (raw)
    sns.heatmap(
        lm_rdm,
        ax=axes[1],
        cbar_kws={"label": "Dissimilarity"},
        cmap="viridis",
        square=True,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"LM RDM (Raw)\n{lm_rdm.shape}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Stimulus")
    axes[1].set_ylabel("Stimulus")
    
    # LM RDM (normalized)
    # For normalized, use symmetric colormap centered at 0
    lm_norm_lim = np.abs(lm_rdm_normalized).max()
    sns.heatmap(
        lm_rdm_normalized,
        ax=axes[2],
        cbar_kws={"label": "Z-score"},
        cmap="RdBu_r",
        square=True,
        vmin=-lm_norm_lim,
        vmax=lm_norm_lim,
    )
    axes[2].set_title(f"LM RDM (Z-Normalized)\nmean={lm_rdm_normalized.mean():.2e}, std={lm_rdm_normalized.std():.2f}", 
                      fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Stimulus")
    axes[2].set_ylabel("Stimulus")
    
    plt.suptitle(
        f"RDM Comparison: Brain vs {lm_name}",
        fontsize=14,
        fontweight="bold",
        y=1.00
    )
    
    fig.tight_layout()
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        plt.show()
    
    # Print statistics
    print(f"\n=== RDM Statistics ===")
    print(f"\nBrain RDM ({brain_session}):")
    print(f"  Shape: {brain_rdm.shape}")
    print(f"  Mean: {brain_rdm.mean():.4f}, Std: {brain_rdm.std():.4f}")
    print(f"  Min: {brain_rdm.min():.4f}, Max: {brain_rdm.max():.4f}")
    
    print(f"\nLM RDM (Raw):")
    print(f"  Shape: {lm_rdm.shape}")
    print(f"  Mean: {lm_rdm.mean():.4f}, Std: {lm_rdm.std():.4f}")
    print(f"  Min: {lm_rdm.min():.4f}, Max: {lm_rdm.max():.4f}")
    
    print(f"\nLM RDM (Z-Normalized):")
    print(f"  Shape: {lm_rdm_normalized.shape}")
    print(f"  Mean: {lm_rdm_normalized.mean():.4e}, Std: {lm_rdm_normalized.std():.4f}")
    print(f"  Min: {lm_rdm_normalized.min():.4f}, Max: {lm_rdm_normalized.max():.4f}")
    
    # Compute correlation between brain and LM (raw vs normalized)
    from scipy.stats import spearmanr
    triu_idx = np.triu_indices_from(brain_rdm, k=1)
    brain_vec = brain_rdm[triu_idx]
    lm_vec = lm_rdm[triu_idx]
    lm_vec_norm = lm_rdm_normalized[triu_idx]
    
    corr_raw, p_raw = spearmanr(brain_vec, lm_vec)
    corr_norm, p_norm = spearmanr(brain_vec, lm_vec_norm)
    
    print(f"\n=== Correlation with Brain RDM ===")
    print(f"\nRaw LM RDM:")
    print(f"  Spearman r: {corr_raw:.4f}, p-value: {p_raw:.4e}")
    
    print(f"\nZ-Normalized LM RDM:")
    print(f"  Spearman r: {corr_norm:.4f}, p-value: {p_norm:.4e}")
    print(f"\nDifference (normalized - raw): {corr_norm - corr_raw:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize brain RDM, LM RDM (raw), and LM RDM (z-normalized)"
    )
    parser.add_argument(
        "--brain-session",
        default="ses-7",
        choices=["ses-5", "ses-7", "ses-9"],
        help="Brain session to compare with",
    )
    parser.add_argument(
        "--lm-rdm-path",
        required=True,
        help="Path to language model RDM .npz file (e.g., data/processed/language_models/checkpoint_trajectory/lm_rdm_*.npz)",
    )
    parser.add_argument(
        "--brain-rdm-dir",
        default="data/processed/fmri",
        help="Directory containing brain RDMs",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the figure (if None, displays)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Colormap minimum (auto if not specified)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Colormap maximum (auto if not specified)",
    )
    
    args = parser.parse_args()
    
    visualize_rdm_comparison(
        brain_session=args.brain_session,
        lm_rdm_path=args.lm_rdm_path,
        brain_rdm_dir=args.brain_rdm_dir,
        output_path=args.output_path,
        vmin=args.vmin,
        vmax=args.vmax,
    )
