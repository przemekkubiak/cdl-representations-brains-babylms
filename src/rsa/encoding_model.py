"""Voxelwise encoding model — LM features -> brain responses (T2.3, strong form).

Standard neuro-AI predictivity: fit ridge regression from LM activations (X:
[n_stimuli, n_features]) to the stimulus x voxel response matrix (Y: [n_stimuli,
n_voxels]), cross-validated over stimuli, and report the mean held-out voxelwise
correlation. This is a stronger, more standard alignment measure than RSA and is
what ICLR reviewers expect alongside representational similarity.

Requires the brain session ``patterns`` matrix saved by session_based_rsa; if a
session RDM lacks patterns (older runs), the encoding score is simply unavailable
and the pipeline falls back to RSA only.
"""

from __future__ import annotations

import numpy as np


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge: (X'X + alpha I)^-1 X'Y. X already has a bias column."""
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d)
    return np.linalg.solve(A, X.T @ Y)


def encoding_score(
    X: np.ndarray, Y: np.ndarray, n_folds: int = 5, alpha: float = 1.0, seed: int = 0
) -> float:
    """Cross-validated mean voxelwise correlation predicting Y from X.

    X : [n_stimuli, n_features]  LM activations for the stimuli.
    Y : [n_stimuli, n_voxels]    brain responses (same stimulus order).
    Returns the mean Pearson r across voxels, averaged over CV folds (nan-safe).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if n < 6 or Y.ndim != 2 or Y.shape[0] != n:
        return float("nan")
    # standardise features; keep voxels raw (corr is scale-free)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, min(n_folds, n))
    fold_scores = []
    for te in folds:
        tr = np.setdiff1d(order, te)
        if len(tr) < 3 or len(te) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = np.c_[np.ones(len(tr)), (X[tr] - mu) / sd]
        Xte = np.c_[np.ones(len(te)), (X[te] - mu) / sd]
        W = _ridge_fit(Xtr, Y[tr], alpha)
        pred = Xte @ W
        # per-voxel correlation on held-out stimuli
        p = pred - pred.mean(0)
        y = Y[te] - Y[te].mean(0)
        num = (p * y).sum(0)
        den = np.sqrt((p ** 2).sum(0) * (y ** 2).sum(0)) + 1e-12
        r = num / den
        fold_scores.append(np.nanmean(r))
    return float(np.nanmean(fold_scores)) if fold_scores else float("nan")
