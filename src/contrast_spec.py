"""Shared ds003604 condition->control contrast specification.

Single source of truth for how each phenomenon's positive (condition) and
negative (control) trials are defined, used by BOTH:
  - scripts/build_contrasts.py  (LM text localizer contrasts)
  - src/rsa/brain_localization.py (brain voxel localizer contrasts)

so the LM per-unit t-map and the brain per-voxel t-map are the *same* contrast
(the CoDLA bridge; see PRIVATE_NOTES.md §5e, §6b). Trial-type codes come from the
task JSON sidecars in the neurodataset (github.com/suchirsalhan/neurodataset_babylm).
"""

from __future__ import annotations

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


def condition_of(trial_type: str, task: str, use_perceptual_control: bool = False) -> str | None:
    """Return 'positive' / 'negative' for a trial_type under a task's contrast,
    or None if the trial is not part of the contrast."""
    spec = CONTRAST_SPEC[task]
    if trial_type in spec["positive"]:
        return "positive"
    neg = spec["perceptual"] if use_perceptual_control else spec["negative"]
    if trial_type in neg:
        return "negative"
    return None
