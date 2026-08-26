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

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from src.contrast_spec import CONTRAST_SPEC, PHENOMENA, condition_of
from src.datasets.stim_identity import classify_trials
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
# Stimulus -> (phenomenon, condition) lookup
# --------------------------------------------------------------------------- #
def build_stim_lookup(
    characteristics_dir: str, use_perceptual_control: bool = False,
    phenomena: Optional[List[str]] = None,
) -> Dict[str, Tuple[str, str]]:
    """ds003604 path (stimuli.kind: github_tsv): map stim_file basename ->
    (phenomenon, 'positive'|'negative') from the four characteristic TSVs.
    Unchanged from before this module was generalized -- `phenomena` defaults
    to the ds003604 global for that reason."""
    lookup: Dict[str, Tuple[str, str]] = {}
    cdir = Path(characteristics_dir)
    for task in (phenomena or PHENOMENA):
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


def build_stim_lookup_from_events(
    dataset: str, phenomena: Optional[List[str]] = None, max_subjects_per_task: int = 3,
) -> Dict[str, Tuple[str, str]]:
    """stim_pair_filename path (ds001894/ds006239/ds002236): no separate
    characteristics file exists, so the (phenomenon, condition) lookup is
    built by re-scanning a few subjects' events.tsv through the SAME
    classify_trials() used at preprocessing time (src/datasets/
    stim_identity.py) -- so a stimulus's condition here can never disagree
    with what it was actually treated as when its pattern was extracted.

    A handful of subjects (not one) because a single subject's run can in
    principle drop a trial (a missed/excluded response), which would make
    that stimulus look absent from this dataset rather than merely absent
    from one subject; unioning across a few subjects is cheap and removes
    that risk. The underlying stimulus SET is fixed by the experimental
    design, not subject-specific, so this does not need to scan everyone.
    """
    from src.datasets import get_dataset

    spec = get_dataset(dataset)
    lookup: Dict[str, List[Tuple[str, str]]] = {}
    data_dir = spec.data_dir()

    for phenomenon in (phenomena or list(spec.phenomena)):
        real_tasks = spec.phenomena.get(phenomenon) or [phenomenon]
        for real_task in real_tasks:
            events_files = sorted(data_dir.glob(f"sub-*/**/*task-{real_task}*_events.tsv"))
            seen_subjects = set()
            for f in events_files:
                subj = f.name.split("_")[0]
                if subj in seen_subjects and len(seen_subjects) >= max_subjects_per_task:
                    continue
                seen_subjects.add(subj)
                import csv
                with open(f, newline="") as fh:
                    rows = list(csv.DictReader(fh, delimiter="\t"))
                for t in classify_trials(rows, dataset, phenomenon):
                    # APPEND, never overwrite: the same stimulus pair commonly
                    # supports more than one phenomenon's contrast at once
                    # (ds001894: the identical word pairs carry both the Phon
                    # and the Orth contrast, just classified positive/negative
                    # differently for each). Overwriting here silently dropped
                    # every one of ds001894's Phon rows the moment Orth was
                    # processed -- caught by cross-checking the combined
                    # multi-dataset figure against the expected coverage
                    # table, not by inspection.
                    entry = (phenomenon, t.condition)
                    slot = lookup.setdefault(t.stim_id, [])
                    if entry not in slot:
                        slot.append(entry)
                if len(seen_subjects) >= max_subjects_per_task:
                    break
    return lookup


def build_stim_lookup_for_dataset(
    dataset: str, characteristics_dir: Optional[str] = None,
    use_perceptual_control: bool = False, phenomena: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """Dispatches on stimuli.kind, same pattern as
    src/datasets/stim_identity.py -- the single entry point callers should
    use instead of picking build_stim_lookup vs. build_stim_lookup_from_events
    themselves.

    Always returns stim_id -> LIST of (phenomenon, condition) pairs, even for
    the ds003604 path (wrapped from build_stim_lookup's single-pair-per-key
    result) -- a stimulus commonly belongs to more than one phenomenon's
    contrast at once (see build_stim_lookup_from_events's docstring), and a
    single-value dict silently drops every phenomenon but the last one
    written for a shared stimulus.
    """
    from src.datasets import get_dataset

    spec = get_dataset(dataset)
    kind = (spec.stimuli or {}).get("kind")
    if kind == "github_tsv":
        cdir = characteristics_dir or str(spec.data_dir() / "stimuli" / "Stimulus_Characteristics")
        flat = build_stim_lookup(cdir, use_perceptual_control, phenomena=phenomena or list(spec.phenomena))
        return {k: [v] for k, v in flat.items()}
    if kind == "stim_pair_filename":
        if use_perceptual_control:
            raise NotImplementedError(
                "use_perceptual_control is not implemented for stim_pair_filename "
                "datasets -- classify_trials() only ever returns positive/negative "
                "(see src/datasets/stim_identity.py); perceptual-control trials are "
                "excluded before a pattern is ever extracted for them, so there is "
                "nothing to contrast against here."
            )
        return build_stim_lookup_from_events(dataset, phenomena=phenomena)
    raise ValueError(f"dataset '{dataset}' has stimuli.kind={kind!r}, unhandled here")


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
    phenomena: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Return {phenomenon: {t, mask, gini, entropy, n_voxels}} for one subject-
    session, all on the same voxel space. `phenomena` defaults to the
    ds003604 global for backward compatibility with existing callers.

    `lookup` maps stim_id -> LIST of (phenomenon, condition) pairs (see
    build_stim_lookup_for_dataset): one stimulus can and often does
    contribute to more than one phenomenon's contrast at once."""
    # group stimuli by (task, condition)
    grouped: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for k, vec in stim.items():
        for entry in lookup.get(k, []):
            grouped.setdefault(entry, []).append(vec)

    out: Dict[str, dict] = {}
    for task in (phenomena or PHENOMENA):
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
    # [A-Za-z0-9+]+, not [A-Za-z0-9]+: the "11+" age-group bin
    # (configs/age_groups.yaml) contains a literal "+", which the digits/
    # letters-only class silently failed to match -- every ses-11+ file was
    # dropped from this listing entirely rather than erroring, so
    # brain_specialization() never saw the oldest age-group bin for any
    # dataset that has one. Caught by cross-checking a real coverage table
    # against the combined multi-dataset figure, not by inspection.
    for f in Path(pattern_dir).glob("sub-*_ses-*_*patterns.npz"):
        m = re.match(r"(sub-[A-Za-z0-9]+)_(ses-[A-Za-z0-9+]+)_", f.name)
        if m:
            pairs.add((m.group(1), m.group(2)))
    return sorted(pairs)


def export_native_tmap_to_mni(
    t_flat: np.ndarray, subject: str, session: str, phenomenon: str,
    dataset: str, data_dir: Path, mask_cache_dir: Path, mni_maps_dir: Path,
) -> Optional[Path]:
    """Reconstruct one phenomenon's flat condition>control t-vector back into
    a real 3D image and warp it to MNI152 -- what makes an actual anatomical
    figure possible, as opposed to the scalar Gini/selectivity summary
    `localize_subject_session` already computes.

    Requires the whole-brain mask `fmri_preprocessing.py` saved for this
    subject-session (`save_native_maps=True` at preprocessing time) and its
    cached registration (same requirement, same cache -- see that flag's
    docstring). Returns the saved path, or None (logged, never raises) if
    either is missing -- a subject-session simply doesn't contribute a
    spatial map in that case, exactly like build_subject_roi_mask's fallback
    for ROI masking.

    `session` here is whatever session_based_rsa groups by, which for the
    three cross-sectional datasets (scripts/regroup_patterns_by_age.py) is an
    AGE-GROUP BIN ("ses-9"), not the real on-disk BIDS session the subject
    was actually scanned in ("ses-T1", "ses-all", ...) -- the mask and
    registration cache were written keyed by the REAL session (mask/
    registration are properties of one physical scan, not of an age bin), so
    an exact-match lookup here would always silently miss for those datasets.
    Fall back to finding this subject's one saved mask by glob and recovering
    the real session from its filename; skip (logged) if that's ambiguous
    (0 or >1 candidates) rather than guessing which scan produced this
    pattern.
    """
    from nilearn.masking import unmask
    from src.preprocessing import spatial_normalization as spatial_norm

    subj_mask_dir = Path(mask_cache_dir) / subject
    mask_path = subj_mask_dir / f"{subject}_{session}_wholebrain_mask.nii.gz"
    reg_session = session
    if not mask_path.exists():
        candidates = sorted(subj_mask_dir.glob(f"{subject}_*_wholebrain_mask.nii.gz")) \
            if subj_mask_dir.is_dir() else []
        if len(candidates) == 1:
            mask_path = candidates[0]
            # f"{subject}_{real_session}_wholebrain_mask.nii.gz" -- strip the
            # fixed prefix/suffix to recover real_session even though it may
            # itself contain underscores (e.g. none of these do, but don't
            # assume it).
            stem = mask_path.name[len(f"{subject}_"):-len("_wholebrain_mask.nii.gz")]
            reg_session = stem
            logger.info("  %s/%s: resolved to real on-disk session '%s' for mask/registration "
                        "lookup (session here is an age-group bin, not a BIDS session)",
                        subject, session, reg_session)
        elif len(candidates) > 1:
            logger.warning("  %s/%s: %d saved whole-brain masks for this subject (%s) -- "
                            "ambiguous which scan this age-group pattern came from, skipping "
                            "spatial map", subject, session, len(candidates),
                            [c.name for c in candidates])
            return None
        else:
            logger.warning("  %s/%s: no saved whole-brain mask under %s -- skipping spatial map "
                            "(preprocess with --save-native-maps first)", subject, session, subj_mask_dir)
            return None

    reg = spatial_norm.get_or_register(
        subject=subject, session=reg_session, data_dir=data_dir, cache_dir=mask_cache_dir,
        epi_ref_img=None,  # cache-only here: raw BOLD is long gone by localization time
        status_csv=Path(mask_cache_dir) / "roi_mask_status.csv",
        status_extra={"roi_set": "tmap_export"},
    )
    if reg is None:
        logger.warning("  %s/%s: no cached registration for real session '%s' -- skipping "
                        "spatial map (preprocess with --save-native-maps first, which "
                        "registers while raw BOLD still exists)", subject, session, reg_session)
        return None

    mask_img = nib.load(str(mask_path))
    try:
        native_map = unmask(t_flat, mask_img)
    except Exception as e:
        logger.warning("  %s/%s: could not reconstruct 3D t-map from mask %s (%s)",
                        subject, session, mask_path, e)
        return None

    try:
        mni_img = spatial_norm.warp_native_to_mni(reg, native_map)
    except Exception as e:
        logger.warning("  %s/%s: MNI warp of t-map failed (%s)", subject, session, e)
        return None

    out_dir = Path(mni_maps_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_{session}_{phenomenon}_tmap_mni.nii.gz"
    tmp = out_path.parent / out_path.name.replace(".nii.gz", ".tmp.nii.gz")
    nib.save(mni_img, str(tmp))
    tmp.rename(out_path)
    return out_path


def brain_specialization(
    pattern_dir: str,
    characteristics_dir: Optional[str] = None,
    sessions: Optional[List[str]] = None,
    percentage: float = 5.0,
    use_perceptual_control: bool = False,
    dataset: str = "ds003604",
    mask_cache_dir: Optional[str] = None,
    mni_maps_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Compute L_brain(P, session): per-phenomenon localization + differentiation,
    aggregated across subjects within each session.

    `dataset` picks the lookup path (build_stim_lookup_for_dataset dispatches
    on stimuli.kind -- ds003604's own characteristics-TSV path is unchanged;
    the other three datasets build their lookup from events.tsv directly, see
    build_stim_lookup_from_events). `characteristics_dir` is only consulted
    for ds003604 and defaults to its usual location when dataset="ds003604";
    ignored otherwise.

    `mask_cache_dir` + `mni_maps_dir` (both required together) additionally
    export each subject-session-phenomenon's condition>control t-map as a
    real MNI152-space NIfTI under `mni_maps_dir` -- see
    export_native_tmap_to_mni. Off by default (None): the scalar summary
    columns this function has always returned are unaffected either way.

    Returns long df: [phenomenon, session, brain_localization(gini), entropy,
    selectivity_index, mean_overlap_with_others, n_subjects, n_voxels]."""
    from src.datasets import get_dataset

    spec = get_dataset(dataset)
    phenomena = list(spec.phenomena) if dataset != "ds003604" else None  # None -> PHENOMENA default
    lookup = build_stim_lookup_for_dataset(
        dataset, characteristics_dir=characteristics_dir,
        use_perceptual_control=use_perceptual_control, phenomena=phenomena,
    )
    subj_sessions = _list_subject_sessions(pattern_dir)
    sess_filter = set(sessions) if sessions else None

    export_maps = bool(mask_cache_dir) and bool(mni_maps_dir)
    if bool(mask_cache_dir) != bool(mni_maps_dir):
        raise ValueError("mask_cache_dir and mni_maps_dir must be given together (or neither)")
    resolved_data_dir = Path(data_dir) if data_dir else spec.data_dir()

    # accumulate per-session lists of per-subject scalars
    acc: Dict[str, Dict[str, list]] = {}
    for subject, session in subj_sessions:
        if sess_filter and session not in sess_filter:
            continue
        stim = gather_session_patterns(pattern_dir, subject, session)
        if not stim:
            continue
        per = localize_subject_session(stim, lookup, percentage, phenomena=phenomena)
        if len(per) < 1:
            continue

        if export_maps:
            for phenomenon, vals in per.items():
                export_native_tmap_to_mni(
                    vals["t"].ravel(), subject, session, phenomenon, dataset,
                    resolved_data_dir, mask_cache_dir, mni_maps_dir,
                )

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


# session -> child age, for the developmental (onset) axis.
# ds003604's own three sessions (unchanged, exact values as before) plus the
# age-group bins from configs/age_groups.yaml, so a session_df built from a
# regrouped cross-sectional dataset (scripts/regroup_patterns_by_age.py --
# "ses-9"/"ses-11"/"ses-11+" instead of a real BIDS session) doesn't silently
# drop those rows via .map()'s NaN-for-unknown-key behaviour.
try:
    from src.datasets.age_groups import representative_age as _representative_age
    SESSION_TO_AGE = {"ses-5": 5, "ses-7": 7, "ses-9": 9}
    SESSION_TO_AGE.update({f"ses-{name}": age for name, age in _representative_age().items()})
except Exception:  # age_groups.yaml unavailable -- fall back to the original three
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
