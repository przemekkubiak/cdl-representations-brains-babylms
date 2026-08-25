"""Noise ceiling maths, with no project imports.

Kept separate from noise_ceiling.py so that session_based_rsa.py can compute a
ceiling at RDM-build time without a circular import (noise_ceiling.py imports
SessionBasedRSA to drive the pattern-directory entry point).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import spearmanr


def noise_ceiling_from_subject_rdms(
    subject_rdms: np.ndarray,
    method: str = "spearman",
) -> Optional[dict]:
    """Nili et al. (2014) noise ceiling from a stack of per-subject RDMs.

    upper : mean over subjects of corr(subject RDM, mean RDM of ALL subjects).
            Optimistic -- each subject is inside the group mean it is scored on.
    lower : mean over subjects of corr(subject RDM, mean of the OTHER subjects).
            Leave-one-out, so it is unbiased and is the bound a model must beat.

    A model's alignment should be read as a FRACTION of this range. Reporting an
    unnormalised RSA is what makes a near-zero result unreadable: it cannot be
    told apart from a dataset in which nothing is predictable at all.

    Parameters
    ----------
    subject_rdms : ndarray [n_subjects, n_stim, n_stim]

    Returns
    -------
    dict with lower, upper, n_subjects, n_stim -- or None if fewer than 3 subjects.
    """
    if subject_rdms is None or len(subject_rdms) < 3:
        return None

    n_sub, n_stim = subject_rdms.shape[0], subject_rdms.shape[1]
    iu = np.triu_indices(n_stim, k=1)
    vecs = np.stack([r[iu] for r in subject_rdms])          # [n_sub, n_pairs]

    ok = np.isfinite(vecs).all(axis=1)
    vecs = vecs[ok]
    if len(vecs) < 3:
        return None
    n_sub = len(vecs)

    group_mean = np.nanmean(vecs, axis=0)
    total = vecs.sum(axis=0)

    uppers, lowers = [], []
    for i in range(n_sub):
        loo_mean = (total - vecs[i]) / (n_sub - 1)
        if method == "spearman":
            u = spearmanr(vecs[i], group_mean).correlation
            l = spearmanr(vecs[i], loo_mean).correlation
        else:
            u = np.corrcoef(vecs[i], group_mean)[0, 1]
            l = np.corrcoef(vecs[i], loo_mean)[0, 1]
        if np.isfinite(u):
            uppers.append(u)
        if np.isfinite(l):
            lowers.append(l)

    if not lowers or not uppers:
        return None

    return {
        "lower": float(np.mean(lowers)),
        "upper": float(np.mean(uppers)),
        "lower_sem": float(np.std(lowers, ddof=1) / np.sqrt(len(lowers))),
        "upper_sem": float(np.std(uppers, ddof=1) / np.sqrt(len(uppers))),
        "n_subjects": int(n_sub),
        "n_stim": int(n_stim),
        "method": method,
    }


def ceiling_from_rdm_file(path: str, method: str = "spearman") -> Optional[dict]:
    """Recompute the ceiling from a saved session RDM that carries subject RDMs.

    Works without any pattern files, which is the point: once an RDM is built
    with subject RDMs retained, the ceiling never needs preprocessing again.
    """
    d = np.load(path, allow_pickle=True)
    if "subject_rdms" not in d.files:
        return None
    return noise_ceiling_from_subject_rdms(d["subject_rdms"], method=method)


