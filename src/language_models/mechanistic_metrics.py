"""Mechanistic / learning-dynamics metrics for LM checkpoints.

Implements the metric family used by pico-analyze (github.com/pico-lm/pico-analyze)
so brain alignment can be *correlated against mechanistic structure* over training:

  single-checkpoint:  norm, gini, hoyer (sparsity), per (proportional effective
                      rank), condition_number
  comparative:        linear CKA (checkpoint-to-checkpoint representational drift)

Unlike pico-analyze (which reads pico's stored `learning_dynamics/` tensors on the
training batch), we compute these on the SAME stimulus activations used for the
brain-LM RSA — so the mechanistic signal and the alignment signal come from one
forward pass on the ds003604 localizer stimuli. Metric definitions match.

Input activations have shape ``[n_stimuli, n_layers, hidden_dim]`` (as returned by
``circuit_localization.ActivationExtractor.extract``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .circuit_localization import gini  # reuse the exact Gini used LM/brain-side


def hoyer_sparsity(x: np.ndarray) -> float:
    """Hoyer sparsity in [0, 1] (1 = maximally sparse). x is a 1-D vector."""
    x = np.abs(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    if n <= 1:
        return 0.0
    l1 = x.sum()
    l2 = np.sqrt((x * x).sum())
    if l2 == 0:
        return 0.0
    sqrt_n = np.sqrt(n)
    return float((sqrt_n - l1 / l2) / (sqrt_n - 1))


def proportional_effective_rank(mat: np.ndarray) -> float:
    """PER: participation ratio of the singular-value spectrum, normalised to
    [0, 1] by the max possible rank. High = representation spread over many
    dimensions; low = collapsed onto few directions."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or min(mat.shape) < 2:
        return float("nan")
    mat = mat - mat.mean(0, keepdims=True)
    s = np.linalg.svd(mat, compute_uv=False)
    s2 = s ** 2
    denom = (s2 ** 2).sum()
    if denom == 0:
        return float("nan")
    eff = (s2.sum() ** 2) / denom          # participation ratio (effective rank)
    return float(eff / min(mat.shape))     # proportional


def condition_number(mat: np.ndarray) -> float:
    """Condition number of the centred activation matrix (ratio of largest to
    smallest non-negligible singular value)."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or min(mat.shape) < 2:
        return float("nan")
    mat = mat - mat.mean(0, keepdims=True)
    s = np.linalg.svd(mat, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return float("nan")
    return float(s[0] / s[-1])


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two [n, d] activation matrices (same n rows)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return float("nan")
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    # HSIC (linear) = ||Y^T X||_F^2 ; normalise by the self-similarities
    xy = np.linalg.norm(y.T @ x, "fro") ** 2
    xx = np.linalg.norm(x.T @ x, "fro")
    yy = np.linalg.norm(y.T @ y, "fro")
    if xx == 0 or yy == 0:
        return float("nan")
    return float(xy / (xx * yy))


def checkpoint_metrics(
    acts: np.ndarray, prev_acts: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Aggregate mechanistic metrics for one checkpoint.

    Parameters
    ----------
    acts : [n_stimuli, n_layers, hidden_dim] activations for this checkpoint.
    prev_acts : same-shaped activations for the previous checkpoint (for CKA drift),
        or None.

    Returns per-checkpoint scalars (mean over layers) plus per-layer arrays under
    keys suffixed ``_by_layer``.
    """
    acts = np.asarray(acts, dtype=np.float64)
    n, L, H = acts.shape
    norms, ginis, hoyers, pers, conds = [], [], [], [], []
    ckas = []
    for li in range(L):
        A = acts[:, li, :]                       # [n, H]
        mean_abs = np.abs(A).mean(0)             # per-unit mean activation
        norms.append(float(np.linalg.norm(A, axis=1).mean()))
        ginis.append(gini(mean_abs))             # concentration across units
        hoyers.append(hoyer_sparsity(mean_abs))
        pers.append(proportional_effective_rank(A))
        conds.append(condition_number(A))
        if prev_acts is not None and prev_acts.shape == acts.shape:
            ckas.append(linear_cka(A, prev_acts[:, li, :]))

    def _m(v):
        v = np.asarray(v, dtype=np.float64)
        return float(np.nanmean(v)) if v.size else float("nan")

    out = {
        "norm": _m(norms),
        "gini": _m(ginis),
        "hoyer": _m(hoyers),
        "per": _m(pers),
        "condition_number": _m(conds),
        "cka_to_prev": _m(ckas) if ckas else float("nan"),
        "norm_by_layer": np.asarray(norms),
        "gini_by_layer": np.asarray(ginis),
        "hoyer_by_layer": np.asarray(hoyers),
        "per_by_layer": np.asarray(pers),
        "condition_number_by_layer": np.asarray(conds),
    }
    return out


# The scalar metrics correlated against brain alignment downstream.
SCALAR_METRICS = ["norm", "gini", "hoyer", "per", "condition_number", "cka_to_prev"]
