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

    # Stimulus TEXT, reconstructed from the same row.
    #
    # WHY THIS IS HERE. The RDM used to store only `stimuli`, which are the audio file
    # names ("stereo_1SH01A0.wav", "PC009.wav"), and scripts/run_devai_grid.py feeds that
    # list straight into ActivationExtractor.extract(), which tokenizes it. The brain-LM
    # alignment was therefore about to be computed from LM activations over *filenames*
    # rather than over the linguistic stimuli -- real numbers, no meaning. The text is
    # recoverable from the characteristics table (word_A/word_B for the word tasks, the
    # sentence constituents for the sentence tasks), which is exactly what
    # src/contrast_spec.reconstruct_text already does for the LM localizer contrasts, so
    # both sides now derive their text from one function.
    try:
        from src.contrast_spec import CONTRAST_SPEC, reconstruct_text
        kind = CONTRAST_SPEC.get(task, {}).get("kind", "word_pair")
    except Exception:
        CONTRAST_SPEC, reconstruct_text, kind = None, None, None

    stim_to_trial_type = {}
    stim_to_text = {}
    for _, row in characteristics.iterrows():
        stim_key = normalize_stimulus_name(row.get("stim_file", ""))
        stim_to_trial_type[stim_key] = row.get("trial_type", "")
        if reconstruct_text is not None:
            try:
                stim_to_text[stim_key] = reconstruct_text(row.to_dict(), kind)
            except Exception:
                pass

    ordered_stimuli = [normalize_stimulus_name(stimulus) for stimulus in stimuli]
    trial_types = [stim_to_trial_type.get(stimulus, "unknown") for stimulus in ordered_stimuli]
    semantic_categories = semantic_categories_from_trial_types(trial_types)
    stimulus_texts = [stim_to_text.get(stimulus, "") for stimulus in ordered_stimuli]

    return {
        "trial_types": np.asarray(trial_types, dtype=object),
        "semantic_categories": np.asarray(semantic_categories, dtype=object),
        "stimulus_texts": np.asarray(stimulus_texts, dtype=object),
    }