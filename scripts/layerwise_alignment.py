#!/usr/bin/env python
"""Brain-LM alignment at EVERY layer, not just the last one.

WHY THIS EXISTS. scripts/run_devai_grid.py computes the LM RDM from `--layer -1`, the final
block. Tier 1 on 2026-08-19 returned a null across all ten families (one of ten families at
p<0.05, mean RSA slightly negative, held-out R^2 = -2.74), and the obvious explanations were
ruled out: the brain RDMs are highly reliable (rho 0.74-0.92 between independent subject
cohorts of the same task), so the null is not cohort noise. The remaining suspect is the
layer. A transformer's final block is specialised for next-token prediction; brain alignment
in this literature usually peaks in middle layers. A near-zero number measured only at the
output layer is exactly what that would look like.

This is cheap because ActivationExtractor.extract already returns [n_stimuli, n_layers,
hidden] from ONE forward pass -- the per-layer activations are being computed and thrown
away. Nothing needs to be re-run on GPU per layer.

    python scripts/layerwise_alignment.py --model babylm-gpt2-3 --max-checkpoints 3 \
        --out data/processed/language_models/layerwise/ds003604
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.language_models.babylm_integration import ModelZoo  # noqa: E402
from src.language_models.circuit_localization import ActivationExtractor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_devai_grid import _lm_rdm, _load_brain, _rsa, _evict_hf_revision  # noqa: E402

TASKS = ("Sem", "Phon", "Gram", "Plaus")
SESSIONS = ("ses-5", "ses-7", "ses-9")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-zoo", default="configs/model_zoo.yaml")
    ap.add_argument("--brain-rdm-root", default="data/processed/fmri/ds003604")
    ap.add_argument("--max-checkpoints", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--out", default="data/processed/language_models/layerwise/ds003604")
    a = ap.parse_args()

    brain = {t: {s: _load_brain(a.brain_rdm_root, t, s) for s in SESSIONS} for t in TASKS}
    cks = ModelZoo(a.model_zoo).resolve_checkpoints(a.model)
    if a.max_checkpoints and len(cks) > a.max_checkpoints:
        idx = np.unique(np.linspace(0, len(cks) - 1, a.max_checkpoints).astype(int))
        cks = [cks[i] for i in idx]
    print(f"> {a.model}: {len(cks)} checkpoints")

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
            ref_s = next((s for s in SESSIONS if bt.get(s) and bt[s].get("texts")), None)
            if ref_s is None:
                continue
            stim = bt[ref_s]["texts"]
            try:
                acts = ex.extract(stim, a.pooling, a.batch_size)  # [n, L, H] -- ONE pass
            except Exception as e:
                print(f"  ! extract failed ({task}): {e}")
                continue
            for li in range(acts.shape[1]):
                lm_rdm = _lm_rdm(acts[:, li, :])
                for s in SESSIONS:
                    b = bt.get(s)
                    if not b or list(b.get("texts") or []) != list(stim):
                        continue
                    if b["rdm"].shape != lm_rdm.shape:
                        continue
                    m = _rsa(lm_rdm, b["rdm"], normalize=True)
                    rows.append({"family": a.model, "model_ref": ref, "step": step,
                                 "task": task, "session": s, "layer": li,
                                 "n_layers": acts.shape[1], "n_stim": lm_rdm.shape[0], **m})
            print(f"  {task}: {acts.shape[1]} layers x sessions done", flush=True)
        del ex
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        _evict_hf_revision(ref)

    if not rows:
        print("no rows"); sys.exit(1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    p = out / f"layerwise_alignment_{a.model}.csv"
    df.to_csv(p, index=False)
    print(f"\nSaved {p}  ({len(df)} rows)")
    print("\nmean RSA by layer (across tasks/sessions/checkpoints):")
    print(df.groupby("layer")["rsa"].agg(["mean", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
