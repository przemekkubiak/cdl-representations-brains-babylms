"""
Unit tests for RSA utilities.
"""

import pytest
import numpy as np
from src.rsa import compute_rdm, compare_rdms


def test_compute_rdm():
    """Test RDM computation."""
    # Create simple test data
    representations = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    rdm = compute_rdm(representations, metric='euclidean')
    
    # Check shape
    assert rdm.shape == (3, 3)
    
    # Check diagonal is zero
    assert np.allclose(np.diag(rdm), 0)
    
    # Check symmetry
    assert np.allclose(rdm, rdm.T)


def test_compare_rdms():
    """Test RDM comparison."""
    rdm1 = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ])
    
    rdm2 = rdm1.copy()
    
    # Identical RDMs should have correlation of 1
    corr, pval = compare_rdms(rdm1, rdm2, method='spearman')
    assert np.isclose(corr, 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
