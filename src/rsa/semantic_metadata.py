"""
Shared helpers for semantic stimulus metadata.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def normalize_stimulus_name(stimulus: str) -> str:
    """Return the filename stem used to match stimulus labels."""
    return Path(str(stimulus)).name


def normalize_trial_type(trial_type: Optional[str]) -> str:
    """Normalize trial-type strings for robust semantic grouping."""
    if trial_type is None:
        return ""

    value = str(trial_type).strip().upper()
    return "".join(ch for ch in value if ch.isalnum())


def semantic_category_from_trial_type(trial_type: Optional[str]) -> str:
    """Map a raw task label to a coarse semantic category."""
    normalized = normalize_trial_type(trial_type)

    if not normalized:
        return "unknown"

    if (
        normalized in {"SC", "CONTROL", "CONTROLS"}
        or "CONTROL" in normalized
        or "CONT" in normalized
    ):
        return "control"

    if (
        normalized in {"SU", "UNRELATED"}
        or "UNREL" in normalized
        or "UNRELATED" in normalized
    ):
        return "unrelated"

    if normalized in {"SH", "HIGH"} or "HIGH" in normalized:
        return "high_association"

    if normalized in {"SL", "LOW"} or "LOW" in normalized:
        return "low_association"

    return "unknown"


def semantic_categories_from_trial_types(trial_types: Iterable[Optional[str]]) -> List[str]:
    """Vectorize semantic category inference across labels."""
    return [semantic_category_from_trial_type(trial_type) for trial_type in trial_types]


def load_semantic_metadata(
    stimuli: Iterable[str],
    task: str = "Sem",
    characteristics_dir: str = "data/brain/ds003604/stimuli/Stimulus_Characteristics",
) -> Dict[str, np.ndarray]:
    """
    Load stimulus metadata for a list of stimuli.

    The returned arrays match the provided stimulus order.
    """
    char_file = Path(characteristics_dir) / f"task-{task}_Stimulus_Characteristics.tsv"
    if not char_file.exists():
        return {}

    try:
        characteristics = pd.read_csv(
            str(char_file),
            sep="\t",
            keep_default_na=False,
            na_values=[""],
        )
    except Exception:
        return {}

    if "stim_file" not in characteristics.columns or "trial_type" not in characteristics.columns:
        return {}

    stim_to_trial_type = {}
    for _, row in characteristics.iterrows():
        stim_key = normalize_stimulus_name(row.get("stim_file", ""))
        stim_to_trial_type[stim_key] = row.get("trial_type", "")

    ordered_stimuli = [normalize_stimulus_name(stimulus) for stimulus in stimuli]
    trial_types = [stim_to_trial_type.get(stimulus, "unknown") for stimulus in ordered_stimuli]
    semantic_categories = semantic_categories_from_trial_types(trial_types)

    return {
        "trial_types": np.asarray(trial_types, dtype=object),
        "semantic_categories": np.asarray(semantic_categories, dtype=object),
    }