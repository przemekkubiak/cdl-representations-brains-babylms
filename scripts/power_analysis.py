#!/usr/bin/env python
"""Minimum detectable effect for the sweep's RSA estimator, per cell.

`detectability.py` answered a related but different question: how strong a probe
BUILT FROM THE BRAIN RDM ITSELF the instrument recovers (w* = 0.02 in all 26
cells). That probe shares the brain's own rank and marginal distribution, so it is
the easiest possible signal. A model RDM does not.

Here the null is built the other way round: take a candidate RDM of the shape a
LANGUAGE MODEL actually produces (1 - corr over mean-pooled embeddings of the same
stimuli), permute the stimulus labels, and measure the spread of the estimator.
That spread is the noise floor a real model effect has to clear, and 2.80 x it is
the effect detectable at alpha = 0.05 two-sided with 80% power in a single cell.

Reported alongside two empirical yardsticks measured elsewhere in this repo, so
the floor can be read against something real:
  * the best observed model rsa for that cell (results/alignment_rows.csv)
  * `pair_len_diff` on ds003604/Phon --- a genuine, replicating, stimulus-locked
    effect found by `stimulus_battery.py` (rsa 0.104 / 0.099 / 0.061 across the
    three sessions, permutation p <= 0.003)
  * the measured relational ceiling (results/relational_ceiling.csv)

Also computed: the across-cell power. The sweep's headline tests aggregate 26
cells, so the quantity that matters for "is this family above zero" is the
smallest consistent per-cell shift a 26-cell Wilcoxon would detect.
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
RNG = np.random.default_rng(0)
N_PERM = 2000
Z80 = 2.802   # 1.96 + 0.842: alpha=0.05 two-sided, 80% power


def ut(m):
    return m[np.triu_indices_from(m, k=1)]


def zscore(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def est(a, b):
    return float(stats.spearmanr(zscore(ut(a)), zscore(ut(b))).statistic)


def corr_rdm(X):
    X = np.asarray(X, float)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    Xn = X / n
    return 1.0 - Xn @ Xn.T


def main() -> int:
    ceil = pd.read_csv(RES / "relational_ceiling.csv").set_index(["dataset", "task", "session"])
    align = pd.read_csv(RES / "alignment_rows.csv")
    best = align.groupby(["dataset", "task", "session"]).rsa.max()

    rows = []
    for f in sorted(glob.glob(str(RDM / "*/within-run-normalised/*/session_rdm_*.npz"))):
        q = Path(f)
        ds = q.relative_to(RDM).parts[0]
        task, ses = q.parts[-2], re.sub(r"session_rdm_(.*)\.npz", r"\1", q.name)
        D = np.asarray(np.load(f, allow_pickle=True)["rdm"], float)
        n = D.shape[0]

        # a model-shaped candidate: 1 - corr over a 768-dim embedding matrix
        M = corr_rdm(RNG.standard_normal((n, 768)))
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            p = RNG.permutation(n)
            null[k] = est(D, M[np.ix_(p, p)])
        sd = float(null.std())

        rc = float(ceil.loc[(ds, task, ses), "ceiling_relational_splithalf"])
        rows.append(dict(
            dataset=ds, task=task, session=ses, n_stim=n, n_pairs=n * (n - 1) // 2,
            perm_null_sd=round(sd, 5),
            mde_single_cell=round(Z80 * sd, 4),
            mde_as_frac_of_relational_ceiling=round(Z80 * sd / rc, 4) if rc > 0 else None,
            best_observed_model_rsa=round(float(best.get((ds, task, ses), np.nan)), 4),
            relational_ceiling=round(rc, 4)))

    df = pd.DataFrame(rows)

    # across-cell: 26 paired cells, one-sample test on the per-cell mean shift.
    # sd here is the between-cell sd of the per-cell null, i.e. how much cell-level
    # values scatter when there is no effect.
    sd_between = float(df.perm_null_sd.mean())
    for n_cells, label in [(26, "all_26_cells"), (12, "ds003604_12_cells"),
                           (6, "ds002236_6_cells"), (8, "ds006239_8_cells")]:
        df.loc[len(df)] = {}
    df = df.dropna(subset=["dataset"])

    across = {label: round(Z80 * sd_between / np.sqrt(n), 5)
              for n, label in [(26, "all_26_cells"), (12, "ds003604_12_cells"),
                               (6, "ds002236_6_cells"), (8, "ds006239_8_cells")]}

    df.to_csv(RES / "power_analysis.csv", index=False)
    summary = dict(
        n_cells=int(len(df)),
        median_perm_null_sd=round(float(df.perm_null_sd.median()), 5),
        median_mde_single_cell=round(float(df.mde_single_cell.median()), 4),
        median_mde_frac_of_relational_ceiling=round(
            float(df.mde_as_frac_of_relational_ceiling.median()), 4),
        median_best_observed_model_rsa=round(float(df.best_observed_model_rsa.median()), 4),
        n_cells_where_best_model_exceeds_mde=int(
            (df.best_observed_model_rsa > df.mde_single_cell).sum()),
        mde_across_cells_mean_shift=across,
        empirical_real_effect_anchor=dict(
            what="pair_len_diff on ds003604/Phon, replicating across 3 sessions",
            rsa=[0.1042, 0.0611, 0.0989],
            note="a genuine stimulus-locked effect reaches ~0.10, "
                 "i.e. ~8% of that cell's relational ceiling"),
    )
    (RES / "power_analysis.json").write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
