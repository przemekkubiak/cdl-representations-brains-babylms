"""Brain-side functional localization: the voxel analogue of the LM unit t-map.

For each phenomenon P and session (age 5/7/9), compute a per-voxel `condition >
control` t-map — exactly the localizer GLM contrast — using the SAME condition
mapping as the LM side (src/contrast_spec.py). Then quantify specialization with
the SAME metrics as the LM side (Gini/entropy/selectivity-index/cross-phenomenon
overlap, reused from circuit_localization) so brain and model are directly
comparable. This produces L_brain(P, age) for CoDLA claim C3 (developmental
correspondence). See PRIVATE_NOTES.md §5c, §6, §6b.

Consumes the preprocessed stimulus-pattern files
(`sub-*_ses-*_run-*_patterns.npz`, keys = stim_file basenames, values = voxel
vectors) produced by src/preprocessing. Stimulus -> (task, condition) is resolved
from the four Stimulus_Characteristics TSVs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from src.contrast_spec import CONTRAST_SPEC, PHENOMENA, condition_of
# reuse the exact LM-side specialization metrics for apples-to-apples comparison
from src.language_models.circuit_localization import (
    gini,
    jaccard_overlap,
    normalized_entropy,
    selectivity_index,
    topk_mask,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stimulus -> (task, condition) lookup from the four characteristic TSVs
# --------------------------------------------------------------------------- #
def build_stim_lookup(
    characteristics_dir: str, use_perceptual_control: bool = False
) -> Dict[str, Tuple[str, str]]:
    """Map stim_file basename -> (task, 'positive'|'negative') across all tasks."""
    lookup: Dict[str, Tuple[str, str]] = {}
    cdir = Path(characteristics_dir)
    for task in PHENOMENA:
        tsv = cdir / f"task-{task}_Stimulus_Characteristics.tsv"
        if not tsv.exists():
            logger.warning(f"Missing characteristics TSV: {tsv}")
            continue
        df = pd.read_csv(tsv, sep="\t", keep_default_na=False)
        for _, row in df.iterrows():
            cond = condition_of(row.get("trial_type", ""), task, use_perceptual_control)
            if cond is None:
                continue
            key = Path(str(row["stim_file"])).name  # basename
            lookup[key] = (task, cond)
    return lookup


def _basename(k: str) -> str:
    return Path(str(k)).name


# --------------------------------------------------------------------------- #
# Load one subject-session into a single shared voxel space
# --------------------------------------------------------------------------- #
def gather_session_patterns(
    pattern_dir: str, subject: str, session: str
) -> Dict[str, np.ndarray]:
    """Collect {stim_basename: voxel_vector} across all runs of a subject-session,
    keeping only voxels at the modal length (one consistent mask/voxel space)."""
    pdir = Path(pattern_dir)
    files = sorted(pdir.glob(f"{subject}_{session}_*patterns.npz"))
    stim: Dict[str, np.ndarray] = {}
    lengths: List[int] = []
    for f in files:
        data = np.load(f)
        for key in data.files:
            vec = np.asarray(data[key]).ravel()
            stim[_basename(key)] = vec
            lengths.append(vec.size)
    if not stim:
        return {}
    # keep the modal voxel length (patterns from the same mask)
    modal = pd.Series(lengths).mode().iloc[0]
    return {k: v for k, v in stim.items() if v.size == modal}


# --------------------------------------------------------------------------- #
# Per-subject-session localization across phenomena (shared voxel space)
# --------------------------------------------------------------------------- #
def localize_subject_session(
    stim: Dict[str, np.ndarray],
    lookup: Dict[str, Tuple[str, str]],
    percentage: float = 5.0,
    min_per_condition: int = 3,
) -> Dict[str, dict]:
    """Return {phenomenon: {t, mask, gini, entropy, n_voxels}} for one subject-
    session, all on the same voxel space."""
    # group stimuli by (task, condition)
    grouped: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for k, vec in stim.items():
        if k in lookup:
            grouped.setdefault(lookup[k], []).append(vec)

    out: Dict[str, dict] = {}
    for task in PHENOMENA:
        pos = grouped.get((task, "positive"), [])
        neg = grouped.get((task, "negative"), [])
        if len(pos) < min_per_condition or len(neg) < min_per_condition:
            continue
        P = np.abs(np.vstack(pos))
        N = np.abs(np.vstack(neg))
        t, _p = ttest_ind(P, N, axis=0, equal_var=False)
        t = np.nan_to_num(t)  # [V]
        t2d = t[None, :]  # reshape to [1, V] for the shared metric helpers
        mask = topk_mask(t2d, percentage)
        out[task] = {
            "t": t2d,
            "mask": mask,
            "gini": gini(t),
            "entropy": normalized_entropy(t),
            "n_voxels": int(t.size),
            "n_selected": int(mask.sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# Session-level aggregation across subjects
# --------------------------------------------------------------------------- #
def _list_subject_sessions(pattern_dir: str) -> List[Tuple[str, str]]:
    pairs = set()
    for f in Path(pattern_dir).glob("sub-*_ses-*_*patterns.npz"):
        m = re.match(r"(sub-[A-Za-z0-9]+)_(ses-[A-Za-z0-9]+)_", f.name)
        if m:
            pairs.add((m.group(1), m.group(2)))
    return sorted(pairs)


def brain_specialization(
    pattern_dir: str,
    characteristics_dir: str,
    sessions: Optional[List[str]] = None,
    percentage: float = 5.0,
    use_perceptual_control: bool = False,
) -> pd.DataFrame:
    """Compute L_brain(P, session): per-phenomenon localization + differentiation,
    aggregated across subjects within each session.

    Returns long df: [phenomenon, session, brain_localization(gini), entropy,
    selectivity_index, mean_overlap_with_others, n_subjects, n_voxels]."""
    lookup = build_stim_lookup(characteristics_dir, use_perceptual_control)
    subj_sessions = _list_subject_sessions(pattern_dir)
    sess_filter = set(sessions) if sessions else None

    # accumulate per-session lists of per-subject scalars
    acc: Dict[str, Dict[str, list]] = {}
    for subject, session in subj_sessions:
        if sess_filter and session not in sess_filter:
            continue
        stim = gather_session_patterns(pattern_dir, subject, session)
        if not stim:
            continue
        per = localize_subject_session(stim, lookup, percentage)
        if len(per) < 1:
            continue
        # cross-phenomenon metrics on this subject's shared voxel space
        t_by = {p: per[p]["t"] for p in per}
        keys = list(per.keys())
        for p in keys:
            others = [q for q in keys if q != p]
            si = selectivity_index(t_by, p, per[p]["mask"]) if others else float("nan")
            ov = (float(np.mean([jaccard_overlap(per[p]["mask"], per[q]["mask"]) for q in others]))
                  if others else float("nan"))
            slot = acc.setdefault(session, {}).setdefault(p, [])
            slot.append(
                {
                    "gini": per[p]["gini"],
                    "entropy": per[p]["entropy"],
                    "selectivity_index": si,
                    "mean_overlap_with_others": ov,
                    "n_voxels": per[p]["n_voxels"],
                }
            )

    rows = []
    for session, byphen in acc.items():
        for phen, vals in byphen.items():
            df = pd.DataFrame(vals)
            rows.append(
                {
                    "phenomenon": phen,
                    "session": session,
                    "brain_localization": float(df["gini"].mean()),
                    "entropy": float(df["entropy"].mean()),
                    "selectivity_index": float(df["selectivity_index"].mean()),
                    "mean_overlap_with_others": float(df["mean_overlap_with_others"].mean()),
                    "n_subjects": len(df),
                    "n_voxels": int(df["n_voxels"].median()),
                }
            )
    return pd.DataFrame(rows)


# session -> child age, for the developmental (onset) axis
SESSION_TO_AGE = {"ses-5": 5, "ses-7": 7, "ses-9": 9}


def collapse_onsets(session_df: pd.DataFrame, frac: float = 0.5) -> pd.DataFrame:
    """Collapse the per-session table to one row per phenomenon with an
    `onset_age`: the earliest age at which brain localization reaches
    min + frac*(max-min) across sessions. Feeds CoDLA C3."""
    out = []
    for phen, sub in session_df.groupby("phenomenon"):
        sub = sub.copy()
        sub["age"] = sub["session"].map(SESSION_TO_AGE)
        sub = sub.dropna(subset=["age"]).sort_values("age")
        v = sub["brain_localization"].to_numpy(dtype=float)
        ages = sub["age"].to_numpy(dtype=float)
        if v.size == 0 or np.nanmax(v) == np.nanmin(v):
            onset = float("nan")
        else:
            thr = np.nanmin(v) + frac * (np.nanmax(v) - np.nanmin(v))
            hit = np.where(v >= thr)[0]
            onset = float(ages[hit[0]]) if hit.size else float("nan")
        out.append(
            {
                "phenomenon": phen,
                "brain_localization": float(np.nanmean(v)) if v.size else float("nan"),
                "onset_age": onset,
            }
        )
    return pd.DataFrame(out)
