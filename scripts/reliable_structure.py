#!/usr/bin/env python
"""What IS the reliable structure in these brain RDMs, if no stimulus explains it?

`relational_ceiling.py`: the additive-residualised (relational) RDMs replicate
across disjoint halves of subjects at a median Spearman of 0.84 (0.98 on
ds003604). `stimulus_battery.py`: not one of 199 stimulus-derived RDMs --- the
studies' own design factors included --- predicts them above chance. Those two
facts together mean there is a large, reproducible, stimulus-indexed signal whose
content is unaccounted for.

This script characterises it, cheaply:

  * **effective dimensionality** of the residual RDM (participation ratio of the
    eigenvalue spectrum of its double-centred Gram matrix)
  * **eigenvector reliability**: does the top component itself replicate across
    disjoint subject halves, or is only the bulk reliable?
  * **row-index structure**: |i - j| in the RDM's own row order. Rows are ordered
    by stimulus filename, which in these datasets is condition- and item-ordered,
    so an index-distance effect is the signature of an acquisition-order or
    blocking artifact rather than a stimulus effect.
  * **top-component loadings vs surface features**, to see whether the dominant
    component is anything nameable at all.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RDM = ROOT / "data/ds003604-session-rdms"
RES = ROOT / "results"
RNG = np.random.default_rng(0)
N_SPLIT = 100


def ut(m):
    return m[np.triu_indices_from(m, k=1)]


def zscore(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def est(a, b):
    return float(stats.spearmanr(zscore(ut(a)), zscore(ut(b))).statistic)


def additive_fit(D):
    D = np.asarray(D, float).copy()
    n = D.shape[0]
    np.fill_diagonal(D, 0.0)
    R = D.sum(axis=1)
    S = (R.sum() / 2.0) / (n - 1.0)
    return (R - S) / (n - 2.0)


def residual(D):
    a = additive_fit(D)
    E = np.asarray(D, float) - (a[:, None] + a[None, :])
    np.fill_diagonal(E, 0.0)
    return E


def gram(D):
    """Classical MDS double-centring: G = -0.5 J D J."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ np.asarray(D, float) @ J


def spectrum(D):
    w = np.linalg.eigvalsh(gram(D))[::-1]
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        return w, np.nan, np.nan
    p = w / s
    eff = float((p.sum() ** 2) / (p ** 2).sum())   # participation ratio
    return w, eff, float(p[0])


def top_vec(D):
    G = gram(D)
    w, V = np.linalg.eigh(G)
    return V[:, -1]


def main() -> int:
    rows = []
    for f in sorted(glob.glob(str(RDM / "*/within-run-normalised/*/session_rdm_*.npz"))):
        q = Path(f)
        ds = q.relative_to(RDM).parts[0]
        task, ses = q.parts[-2], re.sub(r"session_rdm_(.*)\.npz", r"\1", q.name)
        d = np.load(f, allow_pickle=True)
        D = np.asarray(d["rdm"], float)
        S = np.asarray(d["subject_rdms"], float)
        n = D.shape[0]
        E = residual(D)

        _, eff_full, pc1_full = spectrum(D)
        _, eff_rel, pc1_rel = spectrum(E)

        # index-distance RDM: |i - j| in the RDM's own row order
        idx = np.arange(n, dtype=float)
        Idx = np.abs(idx[:, None] - idx[None, :])
        r_idx_full, r_idx_rel = est(D, Idx), est(E, Idx)

        # eigenvector reliability across disjoint subject halves
        k = S.shape[0]
        vals = []
        for _ in range(N_SPLIT):
            p = RNG.permutation(k)
            h = k // 2
            if h < 2:
                break
            v1 = top_vec(residual(S[p[:h]].mean(axis=0)))
            v2 = top_vec(residual(S[p[h:2 * h]].mean(axis=0)))
            vals.append(abs(float(stats.spearmanr(v1, v2).statistic)))
        pc1_rel_reliability = float(np.mean(vals)) if vals else np.nan

        # is the top component anything nameable?
        v = top_vec(E)
        texts = [str(t) for t in d["stimulus_texts"]]
        tt = np.asarray([str(x) for x in d["trial_types"]])
        pc1_vs_len = float(stats.spearmanr(v, [len(t) for t in texts]).statistic)
        pc1_vs_index = float(stats.spearmanr(v, idx).statistic)
        if len(set(tt)) > 1:
            groups = [v[tt == g] for g in sorted(set(tt))]
            pc1_cond_p = float(stats.kruskal(*groups).pvalue)
        else:
            pc1_cond_p = np.nan

        rows.append(dict(
            dataset=ds, task=task, session=ses, n_stim=n, n_subjects=k,
            eff_dim_full=round(eff_full, 2), eff_dim_relational=round(eff_rel, 2),
            pc1_var_share_relational=round(pc1_rel, 4),
            pc1_splithalf_reliability=round(pc1_rel_reliability, 4),
            rsa_vs_index_distance_full=round(r_idx_full, 4),
            rsa_vs_index_distance_relational=round(r_idx_rel, 4),
            pc1_vs_stim_length_rho=round(pc1_vs_len, 4),
            pc1_vs_row_index_rho=round(pc1_vs_index, 4),
            pc1_vs_trialtype_kruskal_p=None if pc1_cond_p != pc1_cond_p else round(pc1_cond_p, 4),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(RES / "reliable_structure.csv", index=False)
    print(df.to_string(index=False))
    print()
    cols = ["eff_dim_relational", "pc1_var_share_relational", "pc1_splithalf_reliability",
            "rsa_vs_index_distance_relational", "pc1_vs_row_index_rho"]
    print(df.groupby("dataset")[cols].median().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
