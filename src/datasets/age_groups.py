"""Cross-dataset developmental age-group taxonomy.

Programmatic access to configs/age_groups.yaml (bin boundaries -- see that
file for the design rationale) plus per-subject age extraction for each
dataset in the registry. ds003604's three BIDS sessions map onto three of the
bins almost exactly (see age_groups.yaml `verified_against`); the other three
datasets are cross-sectional with continuous per-subject age and no BIDS
session that corresponds to a developmental timepoint, so their subjects are
binned individually by real age rather than by session label.

Each per-dataset age extractor below was verified against a real metadata
checkout (data/brain/<accession>/participants.tsv) on 2026-08-26, the same
way every trial-type code and accession in this project is verified rather
than guessed -- see the `note` fields for what was checked.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

_BINS_PATH = Path(__file__).resolve().parents[2] / "configs" / "age_groups.yaml"

# The BIDS session (or, for session-less datasets, the constant "single_session")
# whose subject pool has real per-subject age but no meaningful onset-collapse
# session identity of its own -- see per_subject_ages() below.
SINGLE_SESSION = "single_session"


def load_bins(path: Optional[Path] = None) -> list:
    p = Path(path) if path else _BINS_PATH
    with open(p) as fh:
        return yaml.safe_load(fh)["bins"]


def bin_of(age_years: float, bins: Optional[list] = None) -> str:
    """Which named bin an age falls into. Raises rather than returning a
    sentinel -- a NaN or out-of-range age should be caught by the caller, not
    silently dropped into the wrong bin."""
    if age_years is None or age_years != age_years:  # NaN check without numpy
        raise ValueError("age_years is None/NaN")
    for b in bins or load_bins():
        lo, hi = b["min"], b["max"]
        if (lo is None or age_years >= lo) and (hi is None or age_years < hi):
            return b["name"]
    raise ValueError(f"age {age_years} does not fall into any configured bin")


def representative_age(bins: Optional[list] = None) -> Dict[str, float]:
    """One representative numeric age per bin, for code that needs a single
    sortable number (e.g. brain_localization.py's onset-age threshold
    crossing). Midpoint for closed bins; the open end's edge for "5" and
    "11+". This reproduces ds003604's own nominal ages exactly (5.0/7.0/9.0)
    for the first three bins, so generalizing SESSION_TO_AGE to use this
    changes nothing for ds003604's existing numbers.
    """
    out = {}
    for b in bins or load_bins():
        lo, hi = b["min"], b["max"]
        if lo is None:
            out[b["name"]] = hi - 1.0
        elif hi is None:
            out[b["name"]] = lo
        else:
            out[b["name"]] = (lo + hi) / 2.0
    return out


def _read_tsv(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _parse_date(value: str, formats=("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y")) -> Optional[datetime.date]:
    v = (value or "").strip()
    if not v or v.lower() in ("n/a", "na"):
        return None
    for fmt in formats:
        try:
            return datetime.datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def _ages_ds003604(data_dir: Path) -> Dict[str, Dict[str, float]]:
    """birthdate minus per-session ses-<n>_date_ST. Dates are shifted for
    anonymity but differences are preserved (see neuro_datasets.yaml)."""
    rows = _read_tsv(data_dir / "participants.tsv")
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        bd = _parse_date(r.get("birthdate", ""))
        if not bd:
            continue
        subj = r["participant_id"]
        for ses in ("ses-5", "ses-7", "ses-9"):
            sd = _parse_date(r.get(f"{ses}_date_ST", ""))
            if sd:
                out.setdefault(subj, {})[ses] = (sd - bd).days / 365.25
    return out


def _ages_ds001894(data_dir: Path) -> Dict[str, Dict[str, float]]:
    """Direct age_ses-T1_*/age_ses-T2_* columns (already in years). Uses the
    first populated column per session in a fixed preference order (phenotype
    intake first, matching what's closest to "age at the visit" rather than
    a specific scan), so the choice is the same for every subject rather than
    conditional on which columns happen to be present."""
    rows = _read_tsv(data_dir / "participants.tsv")
    if not rows:
        return {}
    cols = list(rows[0].keys())
    prefer = ["phenotype", "T1w", "AAWord_run-01", "AVWord_run-01", "VVWord_run-01"]

    def cols_for(ses: str) -> list:
        avail = [c for c in cols if c.startswith(f"age_{ses}_")]
        ordered = [c for p in prefer for c in avail if c.endswith(p)]
        return ordered + [c for c in avail if c not in ordered]

    t1_cols, t2_cols = cols_for("ses-T1"), cols_for("ses-T2")
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        subj = r["participant_id"]
        for ses, cc in (("ses-T1", t1_cols), ("ses-T2", t2_cols)):
            for c in cc:
                v = (r.get(c) or "").strip()
                if v and v.lower() != "n/a":
                    try:
                        out.setdefault(subj, {})[ses] = float(v)
                        break
                    except ValueError:
                        continue
    return out


def _ages_ds006239(data_dir: Path) -> Dict[str, Dict[str, float]]:
    """birthdate minus mri_1_complete (REDCap "date this instrument was
    completed" -- verified to be real scan dates, not status flags: computed
    ages land at 10.13-16.87 across all 89 subjects, matching the paper's
    stated 10-17 range). US-format dates (M/D/Y), unlike ds003604's ISO
    dates. Falls back to st_complete/mri_2/mri_3 if mri_1 is missing for a
    subject, in that fixed order -- see neuro_datasets.yaml `ages.note`."""
    rows = _read_tsv(data_dir / "participants.tsv")
    out: Dict[str, Dict[str, float]] = {}
    candidates = ["mri_1_complete", "st_complete", "mri_2_complete", "mri_3_complete"]
    for r in rows:
        bd = _parse_date(r.get("birthdate", ""), formats=("%m/%d/%Y",))
        if not bd:
            continue
        subj = r["participant_id"]
        for c in candidates:
            sd = _parse_date(r.get(c, ""), formats=("%m/%d/%Y",))
            if sd:
                out.setdefault(subj, {})[SINGLE_SESSION] = (sd - bd).days / 365.25
                break
    return out


def _ages_ds002236(data_dir: Path) -> Dict[str, Dict[str, float]]:
    """Direct `age` column, already in years."""
    rows = _read_tsv(data_dir / "participants.tsv")
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        v = (r.get("age") or "").strip()
        if v:
            out.setdefault(r["participant_id"], {})[SINGLE_SESSION] = float(v)
    return out


_EXTRACTORS = {
    "ds003604": _ages_ds003604,
    "ds001894": _ages_ds001894,
    "ds006239": _ages_ds006239,
    "ds002236": _ages_ds002236,
}


def per_subject_ages(dataset: str, data_dir: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """{subject_id: {session_or_SINGLE_SESSION: age_years}} for one dataset.

    Raises KeyError for a dataset with no registered age extractor -- silently
    returning {} would be indistinguishable from "no subjects have age data",
    which is a much more dangerous failure to hide.
    """
    if dataset not in _EXTRACTORS:
        raise KeyError(
            f"no age extractor registered for dataset '{dataset}'. Known: "
            f"{sorted(_EXTRACTORS)}. Add one to src/datasets/age_groups.py, "
            "verified against a real metadata checkout -- do not guess the "
            "participants.tsv column layout."
        )
    from src.datasets import get_dataset

    spec = get_dataset(dataset)
    d = Path(data_dir) if data_dir else spec.data_dir()
    pfile = d / "participants.tsv"
    if not pfile.exists():
        raise FileNotFoundError(
            f"{pfile} not found -- bootstrap a metadata checkout first "
            f"(scripts/inspect_dataset.py --dataset {dataset} --bootstrap)"
        )
    return _EXTRACTORS[dataset](d)


def age_group_for_subject(
    dataset: str, subject: str, session: str, ages: Optional[Dict[str, Dict[str, float]]] = None
) -> Optional[str]:
    """The age-group bin for one (subject, session), or None if no age is on
    record for that subject-session (caller decides the fallback -- this
    never silently returns a guessed bin)."""
    ages = ages if ages is not None else per_subject_ages(dataset)
    subj_ages = ages.get(subject)
    if not subj_ages:
        return None
    key = session if session in subj_ages else SINGLE_SESSION
    age = subj_ages.get(key)
    return bin_of(age) if age is not None else None
