"""Per-dataset condition->control contrast specifications.

Single source of truth for how each phenomenon's positive (condition) and
negative (control) trials are defined, used by BOTH:
  - scripts/build_contrasts.py  (LM text localizer contrasts)
  - src/rsa/brain_localization.py (brain voxel localizer contrasts)

so the LM per-unit t-map and the brain per-voxel t-map are the *same* contrast
(the CoDLA bridge; see PRIVATE_NOTES.md §5e, §6b). Trial-type codes come from the
task JSON sidecars in the neurodataset (github.com/suchirsalhan/neurodataset_babylm).
"""

from __future__ import annotations

import re
from pathlib import Path

# positive (condition) and negative (control) trial_type codes per phenomenon.
# *_C codes are *perceptual* controls (backward/scrambled speech) with no text;
# the LM side therefore contrasts against the linguistic control, which is the
# right within-language contrast anyway. The brain side may optionally use the
# perceptual control instead (set use_perceptual_control=True in the loader).
CONTRAST_SPEC = {
    "Sem":   {"positive": ["S_H"],          "negative": ["S_U"],        "perceptual": ["S_C"],  "kind": "word_pair"},
    "Phon":  {"positive": ["P_R", "P_O"],   "negative": ["P_U"],        "perceptual": ["P_C"],  "kind": "word_pair"},
    "Gram":  {"positive": ["G_G"],          "negative": ["G_F", "G_P"], "perceptual": ["G_C"],  "kind": "sentence"},
    "Plaus": {"positive": ["SP_S", "SP_W"], "negative": ["SP_I"],       "perceptual": ["SP_C"], "kind": "sentence"},
}

# Order in which sentence-task word columns are concatenated to reconstruct text.
SENTENCE_ORDER = ["carrier_phrase", "subject", "verb1", "verb2", "verb3", "number", "object"]

PHENOMENA = list(CONTRAST_SPEC.keys())

# ---------------------------------------------------------------------------
# Per-dataset registry.
#
# Each neuro dataset codes its trial types differently, so the contrast spec
# cannot stay a single global dict once more than one dataset is in play. Keys
# here match `contrast_spec:` in configs/neuro_datasets.yaml.
#
# A dataset MUST NOT be added here from the paper text alone. Trial-type codes
# are read off the dataset's own events.tsv files -- run
#   scripts/inspect_dataset.py --dataset <key>
# which reports the observed codes per task, and transcribe them. Guessing codes
# silently mislabels conditions and produces a contrast that looks fine and
# means nothing.

# --- ds001894 (Lytle et al. 2019) ------------------------------------------
# Verified against data/brain/ds001894/task-*_events.json "trial_type.Levels":
#   1=O+P+  2=O+P-  3=O-P+  4=O-P-  5=control(fixation)  6=perceptual(symbol)
# A 2x2 crossing of ORTHOGRAPHIC and PHONOLOGICAL similarity, which yields two
# contrasts that are decorrelated by design -- Orth is not confounded with Phon
# here, unlike in ds003604 where no orthographic manipulation exists.
# Task suffixes encode presentation modality: AA=audio/audio, VV=visual/visual,
# AV=audio/visual. That is a built-in low-level modality control.
CONTRAST_SPEC_DS001894 = {
    "Phon": {"positive": ["1", "3"], "negative": ["2", "4"], "perceptual": ["6"], "kind": "stim_pair_filename"},
    "Orth": {"positive": ["1", "2"], "negative": ["3", "4"], "perceptual": ["6"], "kind": "stim_pair_filename"},
}

# --- ds006239 (Wang et al. 2025) -------------------------------------------
# Verified against data/brain/ds006239/task-*_events.json "trial_type.Levels".
#   ReadPhon: 1=OyPy 2=OnPy 3=OyPn 4=OnPn  5=PercY 6=PercN 7=FixY 8=FixN
#   ReadMean: 1=HighY 2=LowY 3=UnrN       5/6=Perc 7/8=Fix
#   LocalSem: 1=PicY  3=PicN              5/6=Perc 7/8=Fix
# ReadMean mirrors ds003604's Sem (high / low / unrelated association), so we
# take the same positive=high, negative=unrelated contrast and leave the low
# condition out, for comparability with ds003604 rather than for any deeper
# reason. LocalSem is the ONLY confound-free cell found across all three
# datasets (stimuli recur across runs) -- see configs/neuro_datasets.yaml.
CONTRAST_SPEC_DS006239 = {
    "Phon":     {"positive": ["1", "2"], "negative": ["3", "4"], "perceptual": ["5", "6"], "kind": "stim_pair_filename", "task": "ReadPhon"},
    "Orth":     {"positive": ["1", "3"], "negative": ["2", "4"], "perceptual": ["5", "6"], "kind": "stim_pair_filename", "task": "ReadPhon"},
    "Sem":      {"positive": ["1"],      "negative": ["3"],      "perceptual": ["5", "6"], "kind": "stim_pair_filename", "task": "ReadMean"},
    "SemLocal": {"positive": ["1"],      "negative": ["3"],      "perceptual": ["5", "6"], "kind": "stim_pair_filename", "task": "LocalSem"},
}

# --- ds002236 (Lytle et al. 2020) -------------------------------------------
# Verified against data/brain/ds002236 events.tsv (scripts/inspect_dataset.py,
# 2026-08-26; codes read from the trial_type VALUES directly -- this dataset's
# task JSON sidecars don't carry a Levels map, so the labels below come from
# the events.tsv trial_type text itself, cross-checked across all 91 subjects'
# files, not guessed):
#   AudRhyme: 1=O+P+ 2=O-P+ 3=O+P- 4=O-P- 5=Single Tone 6=Three Tone
#   AudSem:   1=High Related 2=Low Related 3=Non-Related 4=Single Tone 5=Three Tone
# Six tasks exist (Aud/Vis x Rhyme/Sem/Spell); Rhyme and Spell share the
# IDENTICAL O+-P+ crossing, just under a different judgment (rhyme vs.
# spelling) -- Phon is drawn from Rhyme specifically, not Spell, so the
# contrast isn't confounded with an orthographic-judgment task demand. Orth
# (from Spell) and the Vis-modality tasks are deliberately not included yet --
# scoped decision, not an oversight; see MASKING.md-adjacent notes / ask before
# adding them. Unlike ds001894 (where Phon/Orth pool across all six of ITS
# tasks because the codes are modality-invariant there), each entry here names
# its ONE source task explicitly, because we are restricting to Aud only.
CONTRAST_SPEC_DS002236 = {
    "Phon": {"positive": ["1", "2"], "negative": ["3", "4"], "perceptual": ["5", "6"], "kind": "stim_pair_filename", "task": "AudRhyme"},
    "Sem":  {"positive": ["1"],      "negative": ["3"],      "perceptual": ["4", "5"], "kind": "stim_pair_filename", "task": "AudSem"},
}

CONTRAST_SPECS: dict[str, dict] = {
    "ds003604": CONTRAST_SPEC,
    "ds001894": CONTRAST_SPEC_DS001894,
    "ds006239": CONTRAST_SPEC_DS006239,
    "ds002236": CONTRAST_SPEC_DS002236,
}


class ContrastSpecUnavailable(RuntimeError):
    """Raised when a dataset has no verified contrast spec yet."""


def get_contrast_spec(dataset: str = "ds003604") -> dict:
    """Return the condition>control spec for a dataset.

    Raises rather than falling back to ds003604: applying ds003604's trial-type
    codes to another dataset would match nothing (or, worse, match the wrong
    trials) while still producing an output file.
    """
    if dataset not in CONTRAST_SPECS:
        raise ContrastSpecUnavailable(
            f"no verified contrast spec for dataset '{dataset}'. "
            f"Known: {', '.join(sorted(CONTRAST_SPECS))}. "
            f"Run `python scripts/inspect_dataset.py --dataset {dataset}` to read the "
            "observed trial_type codes off its events.tsv, then add them to "
            "CONTRAST_SPECS in src/contrast_spec.py."
        )
    return CONTRAST_SPECS[dataset]


def phenomena_of(dataset: str = "ds003604") -> list[str]:
    return list(get_contrast_spec(dataset).keys())


def reconstruct_text(row: dict, kind: str) -> str:
    """Rebuild the stimulus text from a Stimulus_Characteristics row."""
    if kind == "word_pair":
        a = (row.get("word_A") or "").strip()
        b = (row.get("word_B") or "").strip()
        return f"{a} {b}".strip()
    parts = [
        (row.get(c) or "").strip()
        for c in SENTENCE_ORDER
        if row.get(c) and str(row[c]).strip().lower() != "n/a"
    ]
    return " ".join(" ".join(parts).split())


# Stimulus filenames carry the word itself: ds001894 writes `slour.bmp`,
# ds006239 writes `T3_post.bmp` (a `T<n>_` list prefix then the word). Strip the
# extension and any list prefix to recover the text for the LM-side contrast.
_STIM_PREFIX = re.compile(r"^(?:T\d+|_?F\d+)_")


def text_from_stim_filename(name: str) -> str:
    """Recover the presented word from a stimulus filename."""
    if not name or name.strip().lower() == "n/a":
        return ""
    stem = Path(name).stem
    stem = _STIM_PREFIX.sub("", stem)
    return stem.strip().lower()


def reconstruct_pair_text(row: dict, cols: tuple[str, str]) -> str:
    """Rebuild the two-word stimulus text from a pair of filename columns."""
    a = text_from_stim_filename(row.get(cols[0]) or "")
    b = text_from_stim_filename(row.get(cols[1]) or "")
    return f"{a} {b}".strip()


def normalise_trial_type(value: object) -> str:
    """Normalise a trial_type cell to its canonical string code.

    ds001894 writes trial_type as a float in a minority of files ("5.0" beside
    "5"), which silently splits a condition in two and drops those trials from
    the contrast. Fold them back together.
    """
    v = str(value).strip()
    if not v or v.lower() == "n/a":
        return ""
    try:
        f = float(v)
    except ValueError:
        return v
    return str(int(f)) if f.is_integer() else v


def condition_of(
    trial_type: str,
    task: str,
    use_perceptual_control: bool = False,
    dataset: str = "ds003604",
) -> str | None:
    """Return 'positive' / 'negative' for a trial_type under a task's contrast,
    or None if the trial is not part of the contrast."""
    dataset_spec = get_contrast_spec(dataset)
    if task not in dataset_spec:
        raise ContrastSpecUnavailable(
            f"'{task}' is not a registered phenomenon for dataset '{dataset}'. "
            f"Known: {sorted(dataset_spec)}."
        )
    spec = dataset_spec[task]
    trial_type = normalise_trial_type(trial_type)
    if trial_type in spec["positive"]:
        return "positive"
    neg = spec["perceptual"] if use_perceptual_control else spec["negative"]
    if trial_type in neg:
        return "negative"
    return None
