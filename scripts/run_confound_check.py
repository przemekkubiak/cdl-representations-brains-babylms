#!/usr/bin/env python
"""Is brain-LM alignment recoverable once the scanner-RUN confound is removed?

BACKGROUND. The ds003604 session RDMs are dominated by acquisition structure: whether two
stimuli were presented in the same scanner run predicts their brain dissimilarity at
Spearman +0.49 to +0.87 across all twelve task x session cells. Run assignment is fixed by
the protocol, so it is identical for every subject -- which is why the RDMs look highly
reproducible across independent cohorts (rho 0.74-0.92) while no stimulus property (trial
type, length, lexical overlap) predicts them at all. A language model cannot represent a
scanner run, so its RSA against that structure is ~0 by construction, and Tier 1's null is
a measurement of the confound rather than a result about models.

This script re-runs the alignment with the run effect partialled out of BOTH RDMs, and
reports raw vs partial side by side, per layer. If alignment appears only in the partial
version, the confound was the whole story.

    python scripts/run_confound_check.py --model beetle-humanscale-eng --max-checkpoints 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.language_models.babylm_integration import ModelZoo  # noqa: E402
from src.language_models.circuit_localization import ActivationExtractor  # noqa: E402
from run_devai_grid import _evict_hf_revision, _lm_rdm, _load_brain  # noqa: E402

TASKS = ("Sem", "Phon", "Gram", "Plaus")
SESSIONS = ("ses-5", "ses-7", "ses-9")
CH = "data/brain/ds003604/stimuli/Stimulus_Characteristics"


def rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def partial_spearman(a, b, c):
    """Spearman correlation of a and b with c partialled out (rank-linear residuals)."""
    ra, rb, rc = rank(a), rank(b), rank(c)
    X = np.column_stack([np.ones_like(rc), rc])
    resid = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(spearmanr(resid(ra), resid(rb))[0])


def run_model(task, stimuli):
    """Matrix that is 1 when two stimuli came from different scanner runs."""
    ch = pd.read_csv(f"{CH}/task-{task}_Stimulus_Characteristics.tsv", sep="\t",
                     keep_default_na=False, na_values=[""])
    ch["b"] = ch["stim_file"].astype(str).map(os.path.basename)
    m = dict(zip(ch["b"], ch["run"]))
    runs = [m.get(os.path.basename(str(s)), -1) for s in stimuli]
    n = len(runs)
    return np.array([[0.0 if runs[i] == runs[j] else 1.0 for j in range(n)]
                     for i in range(n)]), runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-zoo", default="configs/model_zoo.yaml")
    ap.add_argument("--brain-rdm-root", default="data/processed/fmri/ds003604")
    ap.add_argument("--max-checkpoints", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="data/processed/language_models/confound_check/ds003604")
    a = ap.parse_args()

    brain = {t: {s: _load_brain(a.brain_rdm_root, t, s) for s in SESSIONS} for t in TASKS}
    cks = ModelZoo(a.model_zoo).resolve_checkpoints(a.model)
    if a.max_checkpoints and len(cks) > a.max_checkpoints:
        cks = [cks[i] for i in np.unique(np.linspace(0, len(cks) - 1,
                                                    a.max_checkpoints).astype(int))]
    # the last checkpoint is the trained one -- that is the interesting case
    rows = []
    for ck in cks:
        ref, step = ck["ref"], ck.get("step")
        print(f"\n=== {ref} (step={step}) ===", flush=True)
        try:
            ex = ActivationExtractor(ref)
        except Exception as e:
            print(f"  ! failed to load: {e}")
            continue
        for task in TASKS:
            bt = brain.get(task, {})
            rs = next((s for s in SESSIONS if bt.get(s) and bt[s].get("texts")), None)
            if rs is None:
                continue
            stim_txt = bt[rs]["texts"]
            runM, runs = run_model(task, bt[rs]["stimuli"])
            if len(set(runs)) < 2:
                print(f"  {task}: single run, nothing to partial out")
                continue
            try:
                acts = ex.extract(stim_txt, "mean", a.batch_size)
            except Exception as e:
                print(f"  ! extract failed ({task}): {e}")
                continue
            iu = np.triu_indices(len(stim_txt), 1)
            for li in range(acts.shape[1]):
                lm = _lm_rdm(acts[:, li, :])
                for s in SESSIONS:
                    b = bt.get(s)
                    if not b or list(b.get("texts") or []) != list(stim_txt):
                        continue
                    if b["rdm"].shape != lm.shape:
                        continue
                    raw = float(spearmanr(lm[iu], b["rdm"][iu])[0])
                    par = partial_spearman(lm[iu], b["rdm"][iu], runM[iu])
                    rows.append({"family": a.model, "step": step, "task": task,
                                 "session": s, "layer": li, "rsa_raw": raw,
                                 "rsa_partial_run": par,
                                 "brain_run_rho": float(spearmanr(runM[iu], b["rdm"][iu])[0]),
                                 "lm_run_rho": float(spearmanr(runM[iu], lm[iu])[0])})
            print(f"  {task}: done ({acts.shape[1]} layers)", flush=True)
        del ex
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        _evict_hf_revision(ref)

    if not rows:
        print("no rows"); sys.exit(1)
    df = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"confound_check_{a.model}.csv"
    df.to_csv(p, index=False)
    print(f"\nSaved {p} ({len(df)} rows)")
    last = df[df.step == df.step.max()]
    print("\n--- LAST (most trained) checkpoint, mean over tasks/sessions, by layer ---")
    print(last.groupby("layer")[["rsa_raw", "rsa_partial_run"]].mean().round(4).to_string())
    print("\n--- overall ---")
    print(df[["rsa_raw", "rsa_partial_run", "brain_run_rho", "lm_run_rho"]].mean().round(4).to_string())
    best = last.groupby("layer")["rsa_partial_run"].mean()
    print(f"\nbest layer by partial RSA: layer {best.idxmax()} = {best.max():+.4f}"
          f"   (raw at that layer: {last[last.layer==best.idxmax()]['rsa_raw'].mean():+.4f})")


if __name__ == "__main__":
    main()
