#!/usr/bin/env python
"""Does ANY stimulus-derived RDM predict these brain RDMs?

`scripts/relational_ceiling.py` shows the brain RDMs carry abundant RELIABLE
relational structure (median split-half 0.84 after removing the additive
per-stimulus main effect). So the near-zero language-model scores are not a
reliability problem. The question this script asks is whether the reliable
structure is stimulus-locked at all, and if so what predicts it.

Two things make this different from the upstream stimulus-property controls:

1. **Every stimulus is a word PAIR, and the pipeline never treats it as one.**
   A trial is `'bad wad'` (rhyme) or `'ape monkey'` (semantic relatedness) --- the
   experiment manipulates the RELATION between the two words, while the sweep
   feeds the concatenated string to an LM and mean-pools. Relation features
   (orthographic overlap, rime match) are therefore tested here as first-class
   candidate RDMs.

2. **ds006239 carries an untested design factor.** Its stimulus filenames encode
   a `T0`-`T3` condition on the second word (`T3_beef.bmp|T1_reef.bmp`), which is
   the study's own manipulation. `trial_types` is `'unknown'` for every ds006239
   and ds002236 cell, so no published control has ever used it.

Each candidate RDM is scored with the sweep's exact estimator (z-normalise both,
Spearman on the upper triangle) against the group RDM, and separately against the
additive-residualised (relational) RDM, with a stimulus-label permutation p-value
and the score expressed as a fraction of the measured relational ceiling.
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
N_PERM = 1000
RNG = np.random.default_rng(0)


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


# ---------------------------------------------------------------- candidates

def rdm_from_1d(x):
    """|x_i - x_j|. A per-trial scalar still yields genuine relational structure."""
    x = np.asarray(x, float)
    return np.abs(x[:, None] - x[None, :])


def rdm_from_labels(lab):
    """0 if same condition, 1 otherwise."""
    lab = np.asarray(lab)
    return (lab[:, None] != lab[None, :]).astype(float)


def edit(a, b):
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


def norm_edit(a, b):
    m = max(len(a), len(b))
    return edit(a, b) / m if m else 0.0


def bigrams(s):
    s = f"^{s}$"
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard_rdm(strs):
    B = [bigrams(s) for s in strs]
    n = len(B)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = len(B[i] | B[j])
            M[i, j] = M[j, i] = 1.0 - (len(B[i] & B[j]) / u if u else 0.0)
    return M


def edit_rdm(strs):
    n = len(strs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = norm_edit(strs[i], strs[j])
    return M


def rime(w, k=3):
    return w[-k:] if len(w) >= k else w


def pair_words(texts):
    out = []
    for t in texts:
        p = str(t).split()
        out.append((p[0], p[1]) if len(p) >= 2 else (p[0] if p else "", ""))
    return out


def candidates(texts, stimuli, trial_types, sem_cats):
    """All stimulus-derived RDMs available for this cell, keyed by name."""
    texts = [str(t) for t in texts]
    pw = pair_words(texts)
    C = {}

    # --- surface / whole-trial string ------------------------------------
    C["surface_length"] = rdm_from_1d([len(t) for t in texts])
    C["surface_nwords"] = rdm_from_1d([len(t.split()) for t in texts])
    C["string_edit"] = edit_rdm(texts)
    C["char_bigram_jaccard"] = jaccard_rdm(texts)

    # --- PAIR-RELATION features (the manipulated variable) ---------------
    C["pair_ortho_overlap"] = rdm_from_1d([1.0 - norm_edit(a, b) for a, b in pw])
    C["pair_rime_match"] = rdm_from_1d([float(rime(a) == rime(b)) for a, b in pw])
    C["pair_len_diff"] = rdm_from_1d([abs(len(a) - len(b)) for a, b in pw])
    C["pair_first_letter_match"] = rdm_from_1d(
        [float(bool(a) and bool(b) and a[0] == b[0]) for a, b in pw])

    # --- design factors ---------------------------------------------------
    tt = np.asarray([str(x) for x in trial_types])
    if len(set(tt)) > 1:
        C["design_trial_type"] = rdm_from_labels(tt)
    sc = np.asarray([str(x) for x in sem_cats])
    if len(set(sc)) > 1:
        C["design_semantic_category"] = rdm_from_labels(sc)

    # ds006239 encodes a T0-T3 condition on the second word of the filename
    codes = []
    for s in stimuli:
        parts = str(s).split("|")
        m = re.match(r"T(\d)_", parts[-1]) if len(parts) > 1 else None
        codes.append(int(m.group(1)) if m else None)
    if all(c is not None for c in codes) and len(set(codes)) > 1:
        C["design_Tcode_same"] = rdm_from_labels(np.asarray(codes))
        C["design_Tcode_ordinal"] = rdm_from_1d(codes)
    return C


def perm_p(brain, cand, obs, n_perm=N_PERM, rng=RNG):
    """Stimulus-label permutation of the candidate; two-sided."""
    null = np.empty(n_perm)
    n = cand.shape[0]
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = est(brain, cand[np.ix_(p, p)])
    return float((np.abs(null) >= abs(obs)).mean())


def main() -> int:
    ceil = pd.read_csv(RES / "relational_ceiling.csv").set_index(["dataset", "task", "session"])
    rows = []
    for f in sorted(glob.glob(str(RDM / "*/within-run-normalised/*/session_rdm_*.npz"))):
        q = Path(f)
        ds = q.relative_to(RDM).parts[0]
        task, ses = q.parts[-2], re.sub(r"session_rdm_(.*)\.npz", r"\1", q.name)
        d = np.load(f, allow_pickle=True)
        D = np.asarray(d["rdm"], float)
        E = residual(D)
        rc = float(ceil.loc[(ds, task, ses), "ceiling_relational_splithalf"])
        C = candidates(d["stimulus_texts"], d["stimuli"], d["trial_types"], d["semantic_categories"])
        for name, M in C.items():
            if np.allclose(M, M[0, 0]):
                continue
            r_full = est(D, M)
            r_rel = est(E, M)
            rows.append(dict(
                dataset=ds, task=task, session=ses, candidate=name, n_stim=D.shape[0],
                rsa_full=round(r_full, 4), rsa_relational=round(r_rel, 4),
                perm_p_full=perm_p(D, M, r_full),
                frac_of_relational_ceiling=round(r_rel / rc, 4) if rc > 0 else None))
        print(f"  {ds}/{task}/{ses}: {len(C)} candidates", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "stimulus_battery.csv", index=False)

    summ = (df.groupby("candidate")
            .agg(n_cells=("rsa_full", "size"),
                 median_rsa_full=("rsa_full", "median"),
                 max_abs_rsa_full=("rsa_full", lambda s: s.abs().max()),
                 median_rsa_relational=("rsa_relational", "median"),
                 n_sig_uncorrected=("perm_p_full", lambda s: int((s < 0.05).sum())),
                 median_frac_ceiling=("frac_of_relational_ceiling", "median"))
            .sort_values("max_abs_rsa_full", ascending=False).round(4))
    summ.to_csv(RES / "stimulus_battery_summary.csv")
    print()
    print(summ.to_string())
    print()
    print("Bonferroni threshold over %d tests: p < %.2e" % (len(df), 0.05 / len(df)))
    print("significant after Bonferroni:", int((df.perm_p_full < 0.05 / len(df)).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
