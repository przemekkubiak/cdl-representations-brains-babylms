"""
Representational Similarity Analysis utilities.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Optional


def compute_rdm(
    representations: np.ndarray,
    metric: str = "correlation"
) -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix (RDM).
    
    Parameters
    ----------
    representations : np.ndarray
        Matrix of representations (n_stimuli x n_features)
    metric : str
        Distance metric ('correlation', 'euclidean', 'cosine')
        
    Returns
    -------
    np.ndarray
        RDM (n_stimuli x n_stimuli)
    """
    distances = pdist(representations, metric=metric)
    rdm = squareform(distances)
    return rdm


def compare_rdms(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    method: str = "spearman"
) -> Tuple[float, float]:
    """
    Compare two RDMs.
    
    Parameters
    ----------
    rdm1 : np.ndarray
        First RDM
    rdm2 : np.ndarray
        Second RDM
    method : str
        Correlation method ('spearman', 'pearson')
        
    Returns
    -------
    Tuple[float, float]
        Correlation coefficient and p-value
    """
    # Get upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(rdm1, k=1)
    vec1 = rdm1[triu_idx]
    vec2 = rdm2[triu_idx]
    
    if method == "spearman":
        corr, pval = spearmanr(vec1, vec2)
    elif method == "pearson":
        corr, pval = pearsonr(vec1, vec2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr, pval


def permutation_test(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    n_permutations: int = 1000,
    method: str = "spearman"
) -> Tuple[float, float]:
    """
    Permutation test for RDM comparison.
    
    Parameters
    ----------
    rdm1 : np.ndarray
        First RDM
    rdm2 : np.ndarray
        Second RDM
    n_permutations : int
        Number of permutations
    method : str
        Correlation method
        
    Returns
    -------
    Tuple[float, float]
        Observed correlation and permutation p-value
    """
    obs_corr, _ = compare_rdms(rdm1, rdm2, method=method)
    
    triu_idx = np.triu_indices_from(rdm1, k=1)
    vec1 = rdm1[triu_idx]
    vec2 = rdm2[triu_idx]
    
    null_dist = []
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(len(vec2))
        if method == "spearman":
            null_corr, _ = spearmanr(vec1, vec2[perm_idx])
        else:
            null_corr, _ = pearsonr(vec1, vec2[perm_idx])
        null_dist.append(null_corr)
    
    null_dist = np.array(null_dist)
    p_value = np.mean(np.abs(null_dist) >= np.abs(obs_corr))
    
    return obs_corr, p_value
