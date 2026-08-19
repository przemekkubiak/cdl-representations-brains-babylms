#!/usr/bin/env python
"""Does within-run normalisation remove the scanner-run confound from the session RDM?

THE PROBLEM. In ds003604 each stimulus is presented in exactly ONE run (run-01 holds 48 of
the 96 Phon stimuli, run-02 the other 48). The session RDM is built across all stimuli, so
every cross-run stimulus pair also differs by a run: scanner drift, baseline shift and
per-run scaling all land in the dissimilarity. Measured consequence -- "different run"
predicts brain dissimilarity at Spearman +0.49 to +0.87 across all twelve task x session
cells, which is far larger than any stimulus property (trial type, length, lexical overlap
are all ~0) and is why the RDMs look highly reproducible across independent subject cohorts
(rho 0.74-0.92): run assignment is fixed by the protocol, so the artefact repeats exactly.

THE STANDARD FIX is to normalise each voxel within run before combining runs, so that a
run's mean and scale cannot contribute to between-stimulus distance.

This script builds both RDMs from the same patterns and reports, for each:
  * how strongly it is predicted by run membership   (want: near 0 after the fix)
  * how strongly it agrees with the other RDM
  * its RSA against a language model               (the question that matters)

It changes nothing in the pipeline -- it is evidence for deciding whether to.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CH = "data/brain/ds003604/stimuli/Stimulus_Characteristics"


def load_subject_runs(pattern_dir: str, session: str):
    """-> {subject: {run: {stim_basename: vector}}}"""
    out = defaultdict(dict)
    for f in sorted(Path(pattern_dir).glob(f"*{session}*_patterns.npz")):
        m = re.match(r"(sub-[^_]+)_.*_(run-\d+)_patterns\.npz", f.name)
        if not m:
            continue
        sub, run = m.group(1), m.group(2)
        try:
            d = np.load(f, allow_pickle=True)
            out[sub][run] = {os.path.basename(k): d[k] for k in d.files}
        except Exception as e:      # a file still being written by a live prep job
            print(f"  skipping {f.name}: {type(e).__name__}")
            out[sub].pop(run, None)
    return out


def build_rdms(subject_runs, stimuli, voxels, seed=0):
    """Return (standard_rdm, within_run_normalised_rdm), averaged over subjects."""
    rng = np.random.default_rng(seed)
    std_list, nrm_list = [], []
    idx = None
    for sub, runs in subject_runs.items():
        # stimulus -> vector, and which run it came from
        vec, of_run = {}, {}
        for run, stims in runs.items():
            for s, v in stims.items():
                vec[s] = v
                of_run[s] = run
        if not all(s in vec for s in stimuli):
            continue
        # Voxel index is drawn PER SUBJECT: the brain masks differ between subjects
        # (917,439 vs 917,389 voxels here), so there is no shared voxel indexing. That is
        # fine for RSA -- each subject gets its own stimulus x stimulus RDM and the RDMs
        # are averaged, which is what session_based_rsa.py does too.
        n_vox = min(len(vec[s]) for s in stimuli)
        idx = rng.choice(n_vox, size=min(voxels, n_vox), replace=False)
        M = np.vstack([vec[s][:n_vox][idx] for s in stimuli])   # [n_stim, n_vox]
        runs_of = np.array([of_run[s] for s in stimuli])
        if not np.isfinite(M).all():
            continue
        std_list.append(1.0 - np.corrcoef(M))
        # within-run z-score per voxel, across the stimuli of that run
        Z = M.copy()
        for r in np.unique(runs_of):
            sel = runs_of == r
            block = Z[sel]
            mu = block.mean(0, keepdims=True)
            sd = block.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            Z[sel] = (block - mu) / sd
        nrm_list.append(1.0 - np.corrcoef(Z))
    if not std_list:
        raise SystemExit("no complete subjects")
    print(f"  aggregated {len(std_list)} subjects, ~{voxels} voxels each")
    return np.nanmean(std_list, axis=0), np.nanmean(nrm_list, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern-dir", default="data/processed/fmri_test/ds003604/Phon")
    ap.add_argument("--task", default="Phon")
    ap.add_argument("--session", default="ses-5")
    ap.add_argument("--voxels", type=int, default=50000)
    ap.add_argument("--model", default="beetle-humanscale-eng")
    a = ap.parse_args()

    subject_runs = load_subject_runs(a.pattern_dir, a.session)
    print(f"subjects with patterns: {len(subject_runs)}")

    ch = pd.read_csv(f"{CH}/task-{a.task}_Stimulus_Characteristics.tsv", sep="\t",
                     keep_default_na=False, na_values=[""])
    ch["b"] = ch["stim_file"].astype(str).map(os.path.basename)
    ch = ch[~ch["trial_type"].astype(str).str.endswith("_C")]          # drop controls
    runmap = dict(zip(ch["b"], ch["run"]))

    # Take the stimulus set from the characteristics table, then keep only the subjects
    # that actually have all of it. Intersecting across EVERY subject instead empties the
    # set the moment one subject has a single run's worth of data (a partially written or
    # partially downloaded subject), which is exactly what happened first time round.
    per_sub = {sub: {s for r in runs.values() for s in r}
               for sub, runs in subject_runs.items()}
    want = sorted(set(runmap) & set().union(*per_sub.values()))
    keep = {sub for sub, ss in per_sub.items() if set(want) <= ss}
    dropped = len(per_sub) - len(keep)
    subject_runs = {k: v for k, v in subject_runs.items() if k in keep}
    stimuli = want
    print(f"stimuli (controls excluded): {len(stimuli)} | subjects kept {len(keep)}"
          f" (dropped {dropped} with incomplete coverage)")

    std, nrm = build_rdms(subject_runs, stimuli, a.voxels)

    runs = np.array([runmap[s] for s in stimuli])
    n = len(stimuli); iu = np.triu_indices(n, 1)
    runM = (runs[:, None] != runs[None, :]).astype(float)

    print("\n--- how much of each brain RDM is scanner-run structure? ---")
    r_std = spearmanr(runM[iu], std[iu])[0]
    r_nrm = spearmanr(runM[iu], nrm[iu])[0]
    print(f"  standard RDM        vs run model: rho = {r_std:+.3f}")
    print(f"  within-run-normed   vs run model: rho = {r_nrm:+.3f}")
    print(f"  standard vs within-run-normed   : rho = {spearmanr(std[iu], nrm[iu])[0]:+.3f}")

    # --- LM alignment against both ---
    from run_devai_grid import _lm_rdm, _evict_hf_revision
    from src.language_models.babylm_integration import ModelZoo
    from src.language_models.circuit_localization import ActivationExtractor
    texts = dict(zip(ch["b"], [
        f"{str(r.get('word_A','')).strip()} {str(r.get('word_B','')).strip()}".strip()
        for _, r in ch.iterrows()]))
    stim_txt = [texts[s] for s in stimuli]
    ck = ModelZoo("configs/model_zoo.yaml").resolve_checkpoints(a.model)[-1]
    print(f"\nLM: {ck['ref']} (step={ck.get('step')})")
    ex = ActivationExtractor(ck["ref"])
    acts = ex.extract(stim_txt, "mean", 16)
    print("\n--- LM alignment, per layer: standard vs within-run-normalised ---")
    print(f"{'layer':>5}  {'raw':>8}  {'normed':>8}")
    best = (None, -9)
    for li in range(acts.shape[1]):
        lm = _lm_rdm(acts[:, li, :])
        a1 = spearmanr(lm[iu], std[iu])[0]
        a2 = spearmanr(lm[iu], nrm[iu])[0]
        print(f"{li:5d}  {a1:+8.4f}  {a2:+8.4f}")
        if a2 > best[1]:
            best = (li, a2)
    print(f"\nbest layer after normalisation: layer {best[0]} rho={best[1]:+.4f}")
    del ex
    _evict_hf_revision(ck["ref"])


if __name__ == "__main__":
    main()
