"""
Semantic-distance summaries for saved neural RDMs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.rsa.semantic_metadata import (
    semantic_category_from_trial_type,
    semantic_categories_from_trial_types,
)


DEFAULT_TARGET_CATEGORIES = ("unrelated", "low_association", "high_association")


def load_session_rdm(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load a saved session RDM and its metadata."""
    data = np.load(filepath, allow_pickle=True)

    metadata = {}
    for key in ["stimuli", "n_subjects", "subject_ids", "metric", "aggregation", "trial_types", "semantic_categories"]:
        if key in data.files:
            metadata[key] = data[key]

    return data["rdm"], data["stimuli"], metadata


def _stimulus_categories(stimuli: Sequence[str], metadata: Dict[str, np.ndarray]) -> List[str]:
    """Resolve semantic categories from saved metadata or, if needed, trial types."""
    semantic_categories = metadata.get("semantic_categories")
    if semantic_categories is not None and len(semantic_categories) == len(stimuli):
        return [str(category) for category in semantic_categories]

    trial_types = metadata.get("trial_types")
    if trial_types is not None and len(trial_types) == len(stimuli):
        return semantic_categories_from_trial_types(trial_types)

    return [semantic_category_from_trial_type(None) for _ in stimuli]


def pairwise_semantic_rows(
    rdm: np.ndarray,
    stimuli: Sequence[str],
    metadata: Optional[Dict[str, np.ndarray]] = None,
    roi_label: Optional[str] = None,
    session: Optional[str] = None,
    target_categories: Sequence[str] = DEFAULT_TARGET_CATEGORIES,
) -> pd.DataFrame:
    """Return one row per stimulus pair with semantic labels attached."""
    metadata = metadata or {}
    categories = _stimulus_categories(stimuli, metadata)

    rows = []
    for i in range(len(stimuli)):
        for j in range(i + 1, len(stimuli)):
            category_i = categories[i]
            category_j = categories[j]

            if category_i not in target_categories or category_j not in target_categories:
                continue

            rows.append(
                {
                    "session": session,
                    "roi": roi_label,
                    "stimulus_i": str(stimuli[i]),
                    "stimulus_j": str(stimuli[j]),
                    "category_i": category_i,
                    "category_j": category_j,
                    "pair_type": "within" if category_i == category_j else "between",
                    "category_pair": "__".join(sorted([category_i, category_j])),
                    "dissimilarity": float(rdm[i, j]),
                }
            )

    return pd.DataFrame(rows)


def summarize_semantic_distance(
    rdm: np.ndarray,
    stimuli: Sequence[str],
    metadata: Optional[Dict[str, np.ndarray]] = None,
    roi_label: Optional[str] = None,
    session: Optional[str] = None,
    target_categories: Sequence[str] = DEFAULT_TARGET_CATEGORIES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a pairwise semantic summary table and a compact contrast table.
    """
    pairwise = pairwise_semantic_rows(
        rdm=rdm,
        stimuli=stimuli,
        metadata=metadata,
        roi_label=roi_label,
        session=session,
        target_categories=target_categories,
    )

    if pairwise.empty:
        contrast = pd.DataFrame([
            {
                "session": session,
                "roi": roi_label,
                "within_mean": np.nan,
                "within_std": np.nan,
                "within_n": 0,
                "between_mean": np.nan,
                "between_std": np.nan,
                "between_n": 0,
                "separation": np.nan,
                "mannwhitney_u": np.nan,
                "p_value": np.nan,
            }
        ])
        return pairwise, contrast

    grouped = (
        pairwise.groupby(["session", "roi", "category_pair", "pair_type"], dropna=False)["dissimilarity"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_dissimilarity", "std": "std_dissimilarity", "count": "n_pairs"})
    )

    within = pairwise[pairwise["pair_type"] == "within"]["dissimilarity"].to_numpy()
    between = pairwise[pairwise["pair_type"] == "between"]["dissimilarity"].to_numpy()

    if len(within) and len(between):
        u_stat, p_value = mannwhitneyu(within, between, alternative="two-sided")
        separation = float(np.mean(between) - np.mean(within))
    else:
        u_stat = np.nan
        p_value = np.nan
        separation = np.nan

    contrast = pd.DataFrame(
        [
            {
                "session": session,
                "roi": roi_label,
                "within_mean": float(np.mean(within)) if len(within) else np.nan,
                "within_std": float(np.std(within)) if len(within) else np.nan,
                "within_n": int(len(within)),
                "between_mean": float(np.mean(between)) if len(between) else np.nan,
                "between_std": float(np.std(between)) if len(between) else np.nan,
                "between_n": int(len(between)),
                "separation": separation,
                "mannwhitney_u": float(u_stat) if np.isfinite(u_stat) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        ]
    )

    return grouped, contrast


def summarize_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    roi_label: Optional[str] = None,
    sessions: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all session RDM files in a directory."""
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("session_rdm_ses-*.npz"))
    if sessions is not None:
        wanted = {str(session) for session in sessions}
        files = [path for path in files if path.stem.split("_")[-1] in wanted]

    pairwise_tables = []
    contrast_tables = []

    for file_path in files:
        rdm, stimuli, metadata = load_session_rdm(str(file_path))
        session = file_path.stem.split("_")[-1]
        pairwise, contrast = summarize_semantic_distance(
            rdm=rdm,
            stimuli=stimuli,
            metadata=metadata,
            roi_label=roi_label or input_path.name,
            session=session,
        )
        if not pairwise.empty:
            pairwise_tables.append(pairwise)
        contrast_tables.append(contrast)

    pairwise_df = pd.concat(pairwise_tables, ignore_index=True) if pairwise_tables else pd.DataFrame()
    contrast_df = pd.concat(contrast_tables, ignore_index=True) if contrast_tables else pd.DataFrame()

    pairwise_path = output_path / "semantic_distance_pairwise.csv"
    contrast_path = output_path / "semantic_distance_contrast.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    contrast_df.to_csv(contrast_path, index=False)

    return pairwise_df, contrast_df