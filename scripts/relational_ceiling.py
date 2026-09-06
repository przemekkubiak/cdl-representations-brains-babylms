#!/usr/bin/env python
"""How much of the brain RDM's RELIABLE variance is relational?

The published noise ceilings (0.23-0.88) are ceilings for the FULL RDM. But
`scripts/detectability.py` found that a purely additive per-stimulus main effect
--- D_ij ~ a_i + a_j, which carries no relational structure at all --- is
significant in 24/26 cells and outscores every language model. That raises the
question this script answers:

    if the additive part is stripped out, is there any RELIABLE relational
    geometry left for RSA to find?

Decomposition. For a symmetric zero-diagonal RDM, fit the additive model
    D_ij ~= mu + a_i + a_j        (i != j)
by least squares in closed form, and call E = D - (a_i + a_j) the RELATIONAL
residual. RSA against a model RDM is only sensitive to relational structure in
so far as the model's own additive component does not do the work; the residual
is that structure isolated.

Ceilings. For each cell, split the subjects into two halves many times, average
each half, and score half against half with the sweep's own estimator
(z-normalise, Spearman on the upper triangle). Doing this for D and for E gives
a full ceiling (which should reproduce the published one) and a RELATIONAL
ceiling. If the relational ceiling is near zero the null is explained: there is
nothing reliable there for any model to align to.

Also reported: how well the per-stimulus amplitude a_i itself replicates across
subject halves, and how much of it is predicted by trivial surface properties
(character count, token count, word count) --- because if a_i is the reliable
signal, the constructive move is to model a_i, not the geometry.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RDM = ROOT / "data/ds003604-session-rdms"
RES = ROOT / "results"
N_SPLIT = 200
RNG = np.random.default_rng(0)


def ut(m):
    return m[np.triu_indices_from(m, k=1)]


def zscore(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def est(a, b):
    """The sweep's estimator: z-normalise both, Spearman on the upper triangle."""
    return float(stats.spearmanr(zscore(ut(a)), zscore(ut(b))).statistic)


def additive_fit(D):
    """Least-squares a_i for D_ij ~ a_i + a_j over i != j. Closed form.

    d/da_i sum_{i<j} (D_ij - a_i - a_j)^2 = 0  =>  (n-2) a_i + S = R_i,
    where R_i is the off-diagonal row sum and S = sum_k a_k = T / (n - 1)
    with T the sum over i<j of D_ij.
    """
    D = np.asarray(D, float).copy()
    n = D.shape[0]
    np.fill_diagonal(D, 0.0)
    R = D.sum(axis=1)
    T = R.sum() / 2.0
    S = T / (n - 1.0)
    a = (R - S) / (n - 2.0)
    return a


def residual(D):
    """D minus its best additive per-stimulus fit; diagonal zeroed."""
    D = np.asarray(D, float)
    a = additive_fit(D)
    E = D - (a[:, None] + a[None, :])
    np.fill_diagonal(E, 0.0)
    return E, a


def cells(dataset):
    out = {}
    for f in sorted(glob.glob(str(RDM / dataset / "within-run-normalised/*/session_rdm_*.npz"))):
        q = Path(f)
        out[(q.parts[-2], re.sub(r"session_rdm_(.*)\.npz", r"\1", q.name))] = np.load(f, allow_pickle=True)
    return out


def split_half(subject_rdms, transform, n_split=N_SPLIT, rng=RNG):
    """Mean |split-half| agreement under `transform`, over random subject splits."""
    S = np.asarray(subject_rdms, float)
    k = S.shape[0]
    if k < 4:
        return np.nan, np.nan
    vals = []
    for _ in range(n_split):
        p = rng.permutation(k)
        h = k // 2
        A = transform(S[p[:h]].mean(axis=0))
        B = transform(S[p[h:2 * h]].mean(axis=0))
        vals.append(est(A, B))
    v = np.asarray(vals, float)
    return float(np.nanmean(v)), float(np.nanstd(v))


def split_half_amplitude(subject_rdms, n_split=N_SPLIT, rng=RNG):
    """Same, but on the per-stimulus amplitude vector a_i (Spearman)."""
    S = np.asarray(subject_rdms, float)
    k = S.shape[0]
    if k < 4:
        return np.nan
    vals = []
    for _ in range(n_split):
        p = rng.permutation(k)
        h = k // 2
        a1 = additive_fit(S[p[:h]].mean(axis=0))
        a2 = additive_fit(S[p[h:2 * h]].mean(axis=0))
        vals.append(stats.spearmanr(a1, a2).statistic)
    return float(np.nanmean(vals))


def surface_r2(texts, a):
    """R^2 of a_i regressed on trivial surface features of the stimulus text."""
    if texts is None:
        return np.nan
    t = [str(x) for x in texts]
    X = np.column_stack([
        [len(s) for s in t],
        [len(s.split()) for s in t],
        [np.mean([len(w) for w in s.split()]) if s.split() else 0.0 for s in t],
    ]).astype(float)
    X = np.column_stack([np.ones(len(t)), (X - X.mean(0)) / (X.std(0) + 1e-12)])
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    resid = a - X @ beta
    ss = ((a - a.mean()) ** 2).sum()
    return float(1.0 - (resid ** 2).sum() / ss) if ss > 0 else np.nan


def main() -> int:
    rows = []
    for ds in ["ds003604", "ds002236", "ds006239"]:
        for (task, ses), d in cells(ds).items():
            D = np.asarray(d["rdm"], float)
            S = np.asarray(d["subject_rdms"], float)
            n = D.shape[0]
            E, a = residual(D)

            # variance share of the additive component, on the upper triangle
            vD, vE = ut(D).var(), ut(E).var()
            add_share = float(1.0 - vE / vD) if vD > 0 else np.nan

            full_m, full_s = split_half(S, lambda x: x)
            rel_m, rel_s = split_half(S, lambda x: residual(x)[0])
            amp_r = split_half_amplitude(S)

            texts = d["stimulus_texts"] if "stimulus_texts" in d.files else None
            rows.append(dict(
                dataset=ds, task=task, session=ses, n_stim=n, n_subjects=int(S.shape[0]),
                ceiling_published=round(float(d["noise_ceiling_lower"]), 4),
                ceiling_full_splithalf=round(full_m, 4), ceiling_full_sd=round(full_s, 4),
                ceiling_relational_splithalf=round(rel_m, 4), ceiling_relational_sd=round(rel_s, 4),
                relational_frac_of_full=round(rel_m / full_m, 4) if full_m and full_m > 0 else None,
                additive_var_share=round(add_share, 4),
                amplitude_splithalf_spearman=round(amp_r, 4),
                amplitude_surface_r2=round(surface_r2(texts, a), 4),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(RES / "relational_ceiling.csv", index=False)

    by_ds = df.groupby("dataset")[[
        "ceiling_full_splithalf", "ceiling_relational_splithalf",
        "relational_frac_of_full", "additive_var_share",
        "amplitude_splithalf_spearman", "amplitude_surface_r2"]].median().round(4)

    summary = dict(
        n_cells=int(len(df)),
        median_ceiling_full=round(float(df.ceiling_full_splithalf.median()), 4),
        median_ceiling_relational=round(float(df.ceiling_relational_splithalf.median()), 4),
        median_relational_frac_of_full=round(float(df.relational_frac_of_full.median()), 4),
        median_additive_var_share=round(float(df.additive_var_share.median()), 4),
        median_amplitude_splithalf=round(float(df.amplitude_splithalf_spearman.median()), 4),
        median_amplitude_surface_r2=round(float(df.amplitude_surface_r2.median()), 4),
        n_cells_relational_ceiling_below_0p1=int((df.ceiling_relational_splithalf < 0.1).sum()),
        by_dataset={k: v.to_dict() for k, v in by_ds.iterrows()},
    )
    (RES / "relational_ceiling_summary.json").write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    print()
    print(by_ds.to_string())
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
