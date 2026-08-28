"""Dataset-aware stimulus identity and trial classification for events.tsv.

Two families exist today, keyed by `stimuli.kind` in configs/neuro_datasets.yaml:

  "github_tsv" (ds003604 only). events.tsv carries a single `stim_file`
    column identifying the trial. Condition (positive/negative/perceptual
    control) comes from a SEPARATE per-task Stimulus_Characteristics.tsv,
    cross-referenced downstream in session_based_rsa.py -- NOT from this
    module. `classify_trials` for this kind deliberately does NOT filter by
    condition: it reproduces the exact pre-2026-08-26 behaviour (every unique
    stim_file becomes a GLM regressor; control trials are excluded later,
    where they always were) so ds003604's published numbers cannot shift.

  "stim_pair_filename" (ds001894, ds006239, ds002236). events.tsv carries its
    OWN inline trial_type code and TWO filename columns identifying a word
    pair. There is no separate characteristics file, so condition comes
    directly from src.contrast_spec.condition_of() against that trial_type,
    and `classify_trials` filters to positive/negative trials only -- control/
    perceptual/null trials never become a GLM regressor in the first place,
    rather than being fit and discarded downstream (there is no downstream
    mechanism for these datasets that would discard them, which is exactly
    the gap this module exists to close -- see MASKING.md-adjacent notes /
    the infra discussion this module was written for).

Both paths return the same shape (a list of Trial), so callers (
src/preprocessing/fmri_preprocessing.py) do not need to know which kind they
are dealing with beyond passing `dataset` through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.contrast_spec import (
    condition_of,
    get_contrast_spec,
    reconstruct_pair_text,
)
from src.datasets import get_dataset


@dataclass
class Trial:
    stim_id: str          # canonical identifier -- unique per distinct trial content
    text: str              # reconstructed stimulus text, "" if not derivable
    condition: Optional[str]  # 'positive' / 'negative' / None (github_tsv: always None)
    row: dict               # the original events.tsv row, for callers that need more


def _stim_id_pair(row: dict, cols: tuple) -> str:
    """Basenames joined with '|', matching scripts/inspect_dataset.py's own
    run-crossing check -- same identity convention used to verify these
    datasets' run/stimulus structure, not a second, possibly-inconsistent one."""
    parts = [Path((row.get(c) or "").strip()).name for c in cols]
    return "|".join(parts)


def classify_trials(events_rows: List[dict], dataset: str, phenomenon: str) -> List[Trial]:
    """Every row in one task's events.tsv -> the trials that should become
    GLM regressors for `phenomenon`, with their reconstructed text and
    (for stim_pair_filename datasets) their condition label.

    Raises ContrastSpecUnavailable (from get_contrast_spec) if `phenomenon`
    has no verified spec for `dataset` -- never silently returns an empty or
    partial list.
    """
    spec = get_dataset(dataset)
    kind = (spec.stimuli or {}).get("kind")

    if kind == "github_tsv":
        # Unchanged from the pre-2026-08-26 behaviour: every unique stim_file
        # is a trial; no condition filtering here (see module docstring).
        out = []
        for row in events_rows:
            stim_file = (row.get("stim_file") or "").strip()
            if stim_file and stim_file.lower() != "n/a":
                out.append(Trial(stim_id=stim_file, text="", condition=None, row=row))
        return out

    if kind == "stim_pair_filename":
        get_contrast_spec(dataset)  # raises clearly if unregistered -- fail before scanning rows
        cols = tuple((spec.stimuli or {}).get("columns") or ())
        if len(cols) != 2:
            raise ValueError(
                f"dataset '{dataset}' is stim_pair_filename but its "
                f"configs/neuro_datasets.yaml `stimuli.columns` is not a 2-tuple: {cols}"
            )
        out = []
        for row in events_rows:
            trial_type = row.get("trial_type", "")
            condition = condition_of(trial_type, task=phenomenon, dataset=dataset)
            if condition is None:
                continue  # perceptual control / null / off-contrast trial -- not a regressor
            if any(not (row.get(c) or "").strip() for c in cols):
                continue  # incomplete row -- can't identify the stimulus
            out.append(Trial(
                stim_id=_stim_id_pair(row, cols),
                text=reconstruct_pair_text(row, cols),
                condition=condition,
                row=row,
            ))
        return out

    raise ValueError(
        f"dataset '{dataset}' has stimuli.kind={kind!r}, which "
        "src/datasets/stim_identity.py does not know how to classify trials "
        "for. Add a branch here rather than guessing a fallback -- the "
        "wrong stimulus identity silently produces a contrast that looks "
        "fine and means nothing (src/contrast_spec.py's own warning)."
    )
