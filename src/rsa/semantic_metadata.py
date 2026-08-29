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


def texts_from_pair_stimuli(stimuli: Iterable[str]) -> List[str]:
    """Recover stimulus text from a pair-of-filenames stimulus KEY.

    ds001894/ds006239/ds002236 identify a trial by the two stimulus files it
    presented, so `stimuli` holds keys like "bad.WAV|wad.WAV" or
    "T3_post.bmp|F1_lost.bmp" -- and the presented WORDS are already in those
    filenames (configs/neuro_datasets.yaml, `stimuli.kind: stim_pair_filename`).
    ds003604 instead identifies a trial by one opaque audio filename
    ("stereo_1SH01A0.wav") whose text is only recoverable from its
    Stimulus_Characteristics table, which is what load_semantic_metadata below
    does.

    This matters because scripts/run_devai_grid.py feeds `stimulus_texts`
    straight to the language model: an empty text means that cell contributes no
    alignment row at all, which is what produced "0 alignment files" for every
    new dataset. Returns [] if these keys are not pair keys, so the caller can
    fall through to the table.

    Output format matches reconstruct_text's word_pair kind exactly -- the two
    words, space-separated ("bad wad") -- so the LM sees the same shape of input
    on every dataset.
    """
    from src.contrast_spec import text_from_stim_filename

    texts, saw_pair = [], False
    for stim in stimuli:
        raw = str(stim)
        if "|" not in raw:
            texts.append("")
            continue
        saw_pair = True
        words = [text_from_stim_filename(part) for part in raw.split("|")]
        texts.append(" ".join(w for w in words if w).strip())
    return texts if saw_pair else []


def conditions_from_stim_lookup(
    stimuli: Iterable[str], dataset: str, task: str,
) -> Optional[np.ndarray]:
    """Real positive/negative labels for stim_pair_filename datasets, via the
    SAME classify_trials()-derived lookup src/rsa/brain_localization.py's
    build_stim_lookup_for_dataset uses to build the RDM's own stimulus set --
    so a label here can never disagree with what a stimulus was actually
    treated as when its pattern was extracted. This is what
    scripts/positive_control.py's `condition` control reads (via
    load_semantic_metadata's `trial_types`); before this it was hardcoded to
    the placeholder "unknown" for every stimulus in these three datasets,
    which made that control permanently degenerate for them.

    Verified 2026-08-29 against the real published ds002236/ds006239 RDMs
    (BrainAlign/ds003604-session-rdms on HF): 100% of every cell's stimuli
    matched this lookup, with a balanced positive/negative split in every
    case (e.g. ds002236 Sem 24/24, ds006239 Phon 48/48).

    Returns None (not a placeholder array) if the lookup itself can't be
    built at all (e.g. this dataset's raw events.tsv aren't on disk -- the
    caller then falls back the same way it always has). Returns "unknown"
    only for an individual stimulus this lookup has no entry for, never for
    the whole array at once, so a partial-coverage dataset still gets real
    labels for the stimuli it can find.
    """
    try:
        from src.rsa.brain_localization import build_stim_lookup_for_dataset
    except Exception:
        return None
    try:
        lookup = build_stim_lookup_for_dataset(dataset, phenomena=[task])
    except Exception:
        return None
    if not lookup:
        return None

    def _pair_key(s: str) -> str:
        parts = str(s).split("|")
        return "|".join(Path(p).name for p in parts)

    labels = []
    for s in stimuli:
        entries = lookup.get(_pair_key(s), [])
        cond = next((c for p, c in entries if p == task), None)
        labels.append(cond or "unknown")
    return np.asarray(labels, dtype=object)


def load_semantic_metadata(
    stimuli: Iterable[str],
    task: str = "Sem",
    characteristics_dir: str = "data/brain/ds003604/stimuli/Stimulus_Characteristics",
    dataset: str = "ds003604",
) -> Dict[str, np.ndarray]:
    """
    Load stimulus metadata for a list of stimuli.

    The returned arrays match the provided stimulus order.
    """
    # Pair-filename datasets carry the words in the stimulus key itself and ship
    # no per-task Stimulus_Characteristics table (ds002236's is a single
    # misspelled file; ds001894/ds006239 have none in this layout), so the key is
    # the only source of text -- and the authoritative one, since it is what the
    # RDM was actually built over. Checked BEFORE the table so a dataset that has
    # both is still read from the thing the RDM is keyed by.
    pair_texts = texts_from_pair_stimuli(stimuli)

    char_file = Path(characteristics_dir) / f"task-{task}_Stimulus_Characteristics.tsv"
    if not char_file.exists():
        if pair_texts:
            stimuli = list(stimuli)
            conditions = conditions_from_stim_lookup(stimuli, dataset, task)
            if conditions is None:
                conditions = np.asarray(["unknown"] * len(stimuli), dtype=object)
            return {
                # positive/negative doubles as the coarse category here --
                # these datasets have no finer-grained semantic taxonomy than
                # the binary contrast itself (see contrast_spec.py).
                "trial_types": conditions,
                "semantic_categories": conditions,
                "stimulus_texts": np.asarray(pair_texts, dtype=object),
            }
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
    # A table that exists but does not key on these stimuli leaves every text
    # empty, which is indistinguishable downstream from "no text exists". Prefer
    # the pair key in that case rather than emitting a silent all-empty column.
    if pair_texts and not any(t for t in stimulus_texts):
        stimulus_texts = pair_texts

    return {
        "trial_types": np.asarray(trial_types, dtype=object),
        "semantic_categories": np.asarray(semantic_categories, dtype=object),
        "stimulus_texts": np.asarray(stimulus_texts, dtype=object),
    }