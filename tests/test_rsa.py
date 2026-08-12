"""
Unit tests for RSA utilities.
"""

import pytest
import numpy as np
from src.rsa import compute_rdm, compare_rdms
from src.rsa.semantic_metadata import (
    load_semantic_metadata,
    semantic_category_from_trial_type,
    semantic_categories_from_trial_types,
)
from src.rsa.semantic_distance_analysis import (
    pairwise_semantic_rows,
    summarize_semantic_distance,
    summarize_directory,
)


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


def test_semantic_category_from_trial_type():
    assert semantic_category_from_trial_type("S_C") == "control"
    assert semantic_category_from_trial_type("S_U") == "unrelated"
    assert semantic_category_from_trial_type("S_H") == "high_association"
    assert semantic_category_from_trial_type("S_L") == "low_association"


def test_semantic_categories_from_trial_types():
    categories = semantic_categories_from_trial_types(["S_C", "S_U", "S_H", "S_L", "mystery"])
    assert categories == [
        "control",
        "unrelated",
        "high_association",
        "low_association",
        "unknown",
    ]


def test_load_semantic_metadata(tmp_path):
    stim_dir = tmp_path / "Stimulus_Characteristics"
    stim_dir.mkdir(parents=True)
    char_file = stim_dir / "task-Sem_Stimulus_Characteristics.tsv"
    char_file.write_text(
        "stim_file\ttrial_type\n"
        "word1.wav\tS_U\n"
        "word2.wav\tS_H\n"
        "word3.wav\tS_L\n"
    )

    metadata = load_semantic_metadata(
        ["/tmp/path/word1.wav", "word2.wav", "word3.wav"],
        task="Sem",
        characteristics_dir=str(stim_dir),
    )

    assert metadata["trial_types"].tolist() == ["S_U", "S_H", "S_L"]
    assert metadata["semantic_categories"].tolist() == [
        "unrelated",
        "high_association",
        "low_association",
    ]


def test_summarize_semantic_distance():
    rdm = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.7],
            [0.8, 0.7, 0.0],
        ]
    )
    stimuli = ["u.wav", "l.wav", "h.wav"]
    metadata = {
        "semantic_categories": np.array(["unrelated", "low_association", "high_association"], dtype=object)
    }

    pairwise, contrast = summarize_semantic_distance(
        rdm=rdm,
        stimuli=stimuli,
        metadata=metadata,
        roi_label="ATL",
        session="ses-7",
    )

    assert len(pairwise) == 3
    assert set(pairwise["pair_type"]) == {"between"}
    assert len(contrast) == 1
    assert np.isclose(contrast.loc[0, "between_mean"], pairwise["dissimilarity"].mean())


def test_summarize_directory(tmp_path):
    input_dir = tmp_path / "processed"
    input_dir.mkdir()

    rdm = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.8],
            [0.9, 0.8, 0.0],
        ]
    )
    stimuli = np.array(["u.wav", "l.wav", "h.wav"], dtype=object)
    np.savez_compressed(
        input_dir / "session_rdm_ses-5.npz",
        rdm=rdm,
        stimuli=stimuli,
        trial_types=np.array(["S_U", "S_L", "S_H"], dtype=object),
        semantic_categories=np.array(["unrelated", "low_association", "high_association"], dtype=object),
    )

    pairwise_df, contrast_df = summarize_directory(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "out"),
        roi_label="AG",
    )

    assert len(pairwise_df) == 3
    assert len(contrast_df) == 1
    assert (tmp_path / "out" / "semantic_distance_pairwise.csv").exists()
    assert (tmp_path / "out" / "semantic_distance_contrast.csv").exists()


if __name__ == "__main__":
    pytest.main([__file__])
