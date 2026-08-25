#!/usr/bin/env python
"""Noise ceiling + within-run correction, and model alignment measured against both.

This is the analysis the published results were missing. It answers, for one
task x session cell:

  1. What is the NOISE CEILING? -- the ceiling is what a perfect model could
     score. Without it a near-zero alignment cannot be told apart from a dataset
     in which nothing at all is predictable, which is exactly the ambiguity the
     current results are stuck in.
  2. How much of the raw RDM is SCANNER RUN? -- and how much survives within-run
     normalisation.
  3. Where does a language model actually land, as a FRACTION of the ceiling?

Everything is computed from per-subject patterns, so both RDM variants come from
identical data and differ only by the correction.

Outputs a CSV per section plus the figures, under --out.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.rsa.ceiling_core import noise_ceiling_from_subject_rdms  # noqa: E402

CH = "data/brain/ds003604/stimuli/Stimulus_Characteristics"


# ----------------------------------------------------------------- loading ---

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
        except Exception:
            out[sub].pop(run, None)   # a file still being written
    return out


def subject_rdm_stack(subject_runs, stimuli, voxels, seed=0):
    """Per-subject RDMs, raw and within-run normalised, from the same patterns.

    Returns (raw [n_sub, S, S], normalised [n_sub, S, S], subject_ids).
    """
    rng = np.random.default_rng(seed)
    raw, nrm, ids = [], [], []

    for sub, runs in sorted(subject_runs.items()):
        vec, of_run = {}, {}
        for run, stims in runs.items():
            for s, v in stims.items():
                vec[s] = v
                of_run[s] = run
        if not all(s in vec for s in stimuli):
            continue

        # Voxel index drawn per subject: brain masks differ between subjects, so
        # there is no shared voxel indexing. Each subject gets its own RDM and
        # the RDMs are averaged -- what session_based_rsa.py does too.
        n_vox = min(len(vec[s]) for s in stimuli)
        idx = rng.choice(n_vox, size=min(voxels, n_vox), replace=False)
        M = np.vstack([np.asarray(vec[s])[:n_vox][idx] for s in stimuli])
        if not np.isfinite(M).all():
            continue

        runs_of = np.array([of_run[s] for s in stimuli])
        Z = M.copy()
        for r in np.unique(runs_of):
            sel = runs_of == r
            block = Z[sel]
            mu, sd = block.mean(0, keepdims=True), block.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            Z[sel] = (block - mu) / sd

        raw.append(1.0 - np.corrcoef(M))
        nrm.append(1.0 - np.corrcoef(Z))
        ids.append(sub)

    if not raw:
        raise SystemExit("no subjects with complete stimulus coverage yet")
    return np.stack(raw), np.stack(nrm), ids


# ------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern-dir", default="data/processed/fmri_wrn/ds003604/Phon")
    ap.add_argument("--task", default="Phon")
    ap.add_argument("--session", default="ses-5")
    ap.add_argument("--voxels", type=int, default=50000)
    ap.add_argument("--models", nargs="+",
                    default=["beetle-humanscale-eng", "babylm-gpt2-3"])
    ap.add_argument("--out", default="paper_results/ceiling")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- stimuli, run map, text ------------------------------------------
    ch = pd.read_csv(f"{CH}/task-{a.task}_Stimulus_Characteristics.tsv", sep="\t",
                     keep_default_na=False, na_values=[""])
    ch["b"] = ch["stim_file"].astype(str).map(os.path.basename)
    ch = ch[~ch["trial_type"].astype(str).str.endswith("_C")]     # drop perceptual controls
    runmap = dict(zip(ch["b"], ch["run"]))

    subject_runs = load_subject_runs(a.pattern_dir, a.session)
    per_sub = {s: {k for r in runs.values() for k in r} for s, runs in subject_runs.items()}
    if not per_sub:
        raise SystemExit(f"no patterns in {a.pattern_dir}")
    stimuli = sorted(set(runmap) & set().union(*per_sub.values()))
    print(f"subjects with patterns: {len(subject_runs)} | stimuli (controls excluded): {len(stimuli)}")

    raw_s, nrm_s, ids = subject_rdm_stack(subject_runs, stimuli, a.voxels)
    print(f"subjects with complete coverage: {len(ids)}")

    n = len(stimuli)
    iu = np.triu_indices(n, 1)
    raw_g, nrm_g = np.nanmean(raw_s, axis=0), np.nanmean(nrm_s, axis=0)

    # ---- 1. noise ceiling -------------------------------------------------
    c_raw = noise_ceiling_from_subject_rdms(raw_s)
    c_nrm = noise_ceiling_from_subject_rdms(nrm_s)
    print("\n--- NOISE CEILING (Nili et al. 2014) ---")
    for nm, c in (("raw", c_raw), ("within-run normalised", c_nrm)):
        print(f"  {nm:22s} lower={c['lower']:+.4f}  upper={c['upper']:+.4f}  n={c['n_subjects']}")

    # ---- 2. run confound --------------------------------------------------
    runs_arr = np.array([runmap[s] for s in stimuli])
    runM = (runs_arr[:, None] != runs_arr[None, :]).astype(float)
    rho_raw = spearmanr(runM[iu], raw_g[iu])[0]
    rho_nrm = spearmanr(runM[iu], nrm_g[iu])[0]
    print("\n--- HOW MUCH OF THE RDM IS SCANNER RUN? ---")
    print(f"  raw                  rho = {rho_raw:+.3f}")
    print(f"  within-run normed    rho = {rho_nrm:+.3f}")
    print(f"  raw vs normed agree  rho = {spearmanr(raw_g[iu], nrm_g[iu])[0]:+.3f}")

    pd.DataFrame([
        {"variant": "raw", "ceiling_lower": c_raw["lower"], "ceiling_upper": c_raw["upper"],
         "ceiling_n": c_raw["n_subjects"], "run_confound_rho": rho_raw},
        {"variant": "within_run_normalised", "ceiling_lower": c_nrm["lower"],
         "ceiling_upper": c_nrm["upper"], "ceiling_n": c_nrm["n_subjects"],
         "run_confound_rho": rho_nrm},
    ]).assign(task=a.task, session=a.session, n_stim=n).to_csv(
        out / f"ceiling_{a.task}_{a.session}.csv", index=False)

    # ---- 3. model alignment vs both, normalised by the ceiling ------------
    from run_devai_grid import _lm_rdm, _evict_hf_revision            # noqa: E402
    from src.language_models.babylm_integration import ModelZoo       # noqa: E402
    from src.language_models.circuit_localization import ActivationExtractor  # noqa: E402

    texts = dict(zip(ch["b"], [
        f"{str(r.get('word_A','')).strip()} {str(r.get('word_B','')).strip()}".strip()
        for _, r in ch.iterrows()]))
    stim_txt = [texts[s] for s in stimuli]

    zoo = ModelZoo("configs/model_zoo.yaml")
    rows = []
    for fam in a.models:
        try:
            ck = zoo.resolve_checkpoints(fam)[-1]
        except Exception as e:
            print(f"  ! {fam}: {e}")
            continue
        print(f"\nLM: {fam} -> {ck['ref']} (step={ck.get('step')})")
        try:
            ex = ActivationExtractor(ck["ref"])
            acts = ex.extract(stim_txt, "mean", a.batch_size)
        except Exception as e:
            print(f"  ! extract failed: {e}")
            continue
        for li in range(acts.shape[1]):
            lm = _lm_rdm(acts[:, li, :])
            rows.append({
                "family": fam, "model_ref": ck["ref"], "step": ck.get("step"),
                "task": a.task, "session": a.session, "layer": li,
                "rsa_raw": spearmanr(lm[iu], raw_g[iu])[0],
                "rsa_normalised": spearmanr(lm[iu], nrm_g[iu])[0],
                "ceiling_lower_raw": c_raw["lower"], "ceiling_lower_norm": c_nrm["lower"],
                "n_stim": n, "n_subjects": len(ids),
            })
        del ex
        _evict_hf_revision(ck["ref"])

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nno model rows produced")
        return
    # Fraction of achievable signal: alignment / lower ceiling.
    df["frac_of_ceiling_raw"] = df["rsa_raw"] / df["ceiling_lower_raw"]
    df["frac_of_ceiling_norm"] = df["rsa_normalised"] / df["ceiling_lower_norm"]
    df.to_csv(out / f"alignment_vs_ceiling_{a.task}_{a.session}.csv", index=False)

    print("\n--- BEST LAYER PER MODEL ---")
    for fam, g in df.groupby("family"):
        b = g.loc[g["rsa_normalised"].idxmax()]
        print(f"  {fam:26s} layer {int(b['layer']):2d}  corrected rho={b['rsa_normalised']:+.4f}"
              f"  = {100*b['frac_of_ceiling_norm']:.1f}% of ceiling ({b['ceiling_lower_norm']:.3f})")

    json.dump({"ceiling_raw": c_raw, "ceiling_normalised": c_nrm,
               "run_confound_raw": float(rho_raw), "run_confound_normalised": float(rho_nrm),
               "task": a.task, "session": a.session, "n_stim": int(n),
               "n_subjects": len(ids)},
              open(out / f"summary_{a.task}_{a.session}.json", "w"), indent=2)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
