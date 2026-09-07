#!/usr/bin/env python
"""Shared loaders and binning for the supplementary figure set.

Every figure in scripts/make_supplementary_figures.py is built on this module
so that "a token bin", "a 95% CI" and "the colour for Gram" mean exactly one
thing across the whole set, and so each figure can dump the table it was drawn
from next to itself.

The token binning here reproduces the main-text convention: 16 log-spaced bins
spanning 2e6 to 3e11 cumulative training tokens. Keep it identical -- a
supplementary panel that silently rebinned would not be comparable to Figure 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from viz.plot_style import PALETTE_ACL  # noqa: E402  (needs REPO on sys.path)

# ── Token binning (must match the main text) ────────────────────────────────
TOKEN_MIN, TOKEN_MAX, N_BINS = 2e6, 3e11, 16
BIN_EDGES = np.logspace(np.log10(TOKEN_MIN), np.log10(TOKEN_MAX), N_BINS + 1)
BIN_CENTRES = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])

#: The 150M-token mark: BabyLM-GPT2's whole training set. Every trajectory
#: panel marks it, because "before/after a child-scale data budget" is the
#: comparison the paper is actually about.
BABYLM_TOKENS = 1.5e8

# ── Semantic colour assignments, all drawn from the MECO palette ────────────
DOMAIN_COLORS = {
    "Sem":      PALETTE_ACL["B1"],
    "Phon":     PALETTE_ACL["B2"],
    "Gram":     PALETTE_ACL["B3"],
    "Plaus":    PALETTE_ACL["B4"],
    "Orth":     PALETTE_ACL["B5"],
    "SemLocal": PALETTE_ACL["B6"],
}
DOMAIN_ORDER = ["Sem", "Phon", "Gram", "Plaus"]
DOMAIN_LABELS = {
    "Sem": "Semantic", "Phon": "Phonological",
    "Gram": "Grammatical", "Plaus": "Plausibility",
    "Orth": "Orthographic", "SemLocal": "Semantic (local)",
}

#: Masks: whole-brain is the reference, so it takes the neutral grey.
MASK_COLORS = {
    "whole-brain": PALETTE_ACL["mono"],
    "auditory":    PALETTE_ACL["B1"],
    "motor":       PALETTE_ACL["B2"],
    "phonology":   PALETTE_ACL["B3"],
}
MASK_PACKAGES = {
    "whole-brain": "ds003604",
    "auditory":    "ds003604-roiauditory",
    "motor":       "ds003604-roimotor",
    "phonology":   "ds003604-roiphonology",
}
MASK_ORDER = ["whole-brain", "auditory", "motor", "phonology"]

#: ds003604 sessions are child age in years; the other datasets' session
#: labels are already age bins. See scripts/plot_activation_by_age_domain.py.
SESSION_ORDER = ["ses-5", "ses-7", "ses-9", "ses-11", "ses-11+"]
SESSION_LABELS = {s: s.replace("ses-", "") for s in SESSION_ORDER}


def assign_token_bin(tokens: pd.Series) -> pd.Series:
    """Map cumulative training tokens to a 0-based bin index (NaN if outside)."""
    idx = np.digitize(tokens.to_numpy(dtype=float), BIN_EDGES) - 1
    idx = np.where((idx < 0) | (idx >= N_BINS), np.nan, idx)
    return pd.Series(idx, index=tokens.index)


def mean_ci(values: np.ndarray, alpha: float = 0.05):
    """Mean and t-distribution CI. Returns (mean, lo, hi, n); CI is NaN for n<2.

    A t CI rather than a normal one because occupied bins can hold as few as
    two checkpoints, which is also why bins with n<2 are reported as
    unoccupied everywhere in this figure set.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = v.mean()
    if n < 2:
        return m, np.nan, np.nan, n
    half = stats.t.ppf(1 - alpha / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return m, m - half, m + half, n


#: The unit a confidence interval is computed over. A checkpoint contributes
#: one row per (dataset, task, session) cell, so treating raw rows as
#: independent inflates n by up to 8x and shrinks every interval accordingly.
#: The main text's Table 4 reports n = 30 in the last token bin, which is the
#: checkpoint count, so checkpoint is also the unit that reproduces it.
DEFAULT_CI_UNIT = "model_ref"


def binned_ci(df: pd.DataFrame, value_col: str, by: list[str],
              unit: str | None = DEFAULT_CI_UNIT) -> pd.DataFrame:
    """Group, then summarise each group as mean + t CI. Drops n<2 groups.

    Values are averaged within ``unit`` inside each group before the interval
    is taken, so n is a count of independent-ish units rather than of rows.
    Pass unit=None to summarise the rows as given.
    """
    if unit is not None and unit in df.columns:
        df = df.groupby(list(by) + [unit], dropna=True,
                        observed=True, as_index=False)[value_col].mean()
    rows = []
    for keys, g in df.groupby(by, dropna=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        m, lo, hi, n = mean_ci(g[value_col].to_numpy())
        if n < 2:
            continue
        rows.append({**dict(zip(by, keys)), "mean": m, "ci_lo": lo,
                     "ci_hi": hi, "n": n})
    return pd.DataFrame(rows)


# ── Loaders ────────────────────────────────────────────────────────────────

def load_alignment_rows() -> pd.DataFrame:
    """The pooled per-(dataset, family, checkpoint, task, session) RSA table."""
    d = pd.read_csv(REPO / "results" / "alignment_rows.csv")
    d["token_bin"] = assign_token_bin(d["tokens"])
    return d


def load_package_checkpoints(pkg: str) -> pd.DataFrame:
    """One package's per-checkpoint table (brain RSA + mechanistic + behaviour)."""
    d = pd.read_csv(REPO / "hf_package" / pkg / "overall" / "by_checkpoint.csv")
    d["token_bin"] = assign_token_bin(d["tokens"])
    d["package"] = pkg
    return d


def _concat_by_model(pkg: str, filename: str) -> pd.DataFrame:
    paths = sorted((REPO / "hf_package" / pkg / "by-model").glob(f"*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"no {filename} under hf_package/{pkg}/by-model")
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d["package"] = pkg
    return d


def load_package_alignment(pkg: str) -> pd.DataFrame:
    """Per-checkpoint x task x session RSA for one package (whole-brain or ROI)."""
    d = _concat_by_model(pkg, "brain_alignment.csv")
    d["token_bin"] = assign_token_bin(d["tokens"])
    return d


def load_package_behaviour(pkg: str = "ds003604") -> pd.DataFrame:
    """Per-checkpoint minimal-pair accuracy, per phenomenon."""
    d = _concat_by_model(pkg, "behaviour.csv")
    d["token_bin"] = assign_token_bin(d["tokens"])
    return d


def load_package_isolation(pkg: str = "ds003604") -> pd.DataFrame:
    """Per-checkpoint circuit-localisation metrics, per phenomenon."""
    return _concat_by_model(pkg, "localisation_isolation.csv")


def load_package_layerwise(pkg: str = "ds003604") -> pd.DataFrame:
    """Per-checkpoint x layer sparsity metrics."""
    return _concat_by_model(pkg, "interp_layerwise.csv")


def load_brain_localization() -> pd.DataFrame:
    """Brain-side specialisation per phenomenon x session, across datasets.

    Reads the within-run-normalised localisation outputs -- the variant the
    paper's results are built on (results/roi_by_cell.csv, variant column).
    """
    base = REPO / "pipeline" / "data" / "processed" / "fmri_wrn"
    frames = []
    for path in sorted(base.glob("*/localization/brain_localization_by_session.csv")):
        d = pd.read_csv(path)
        d["dataset"] = path.parts[-3]
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no brain_localization_by_session.csv under {base}")
    return pd.concat(frames, ignore_index=True)


def architecture_of(family: str) -> str:
    """Model family -> architecture. Seeded parc-* families name theirs."""
    if family.startswith("parc-"):
        return family.split("-")[1]
    if family.startswith(("pythia", "polypythia")):
        return "pythia"
    if family.startswith("babylm-gpt2"):
        return "gpt2"
    return "unknown"


def savefig_note(stem: Path, table: pd.DataFrame) -> None:
    """Write the exact table a figure was drawn from, next to the figure."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"), index=False)


#: The 15-family "devai" grid, in its within-run-normalised form. This is a
#: SEPARATE run from grid/ (which feeds results/alignment_rows.csv): the two
#: share nine model families and three datasets, and where they overlap their
#: rho values differ by up to 0.067 -- larger than the whole effect range the
#: paper reports. So they must never be pooled. It is kept because it is the
#: only run that covers ds001894 at all.
DEVAI_WRN = ("pipeline/data/processed/language_models/devai_grid_wrn")
DEVAI_WRN_DATASETS = ["ds001894", "ds002236", "ds006239"]


def load_devai_wrn_alignment() -> pd.DataFrame:
    """Alignment rows from the devai within-run-normalised grid (3 datasets)."""
    frames = []
    for ds in DEVAI_WRN_DATASETS:
        paths = sorted((REPO / DEVAI_WRN / ds).glob("alignment_*.csv"))
        if not paths:
            raise FileNotFoundError(f"no alignment_*.csv under {DEVAI_WRN}/{ds}")
        frames.extend(pd.read_csv(p) for p in paths)
    d = pd.concat(frames, ignore_index=True)
    d["token_bin"] = assign_token_bin(d["tokens"])
    return d
