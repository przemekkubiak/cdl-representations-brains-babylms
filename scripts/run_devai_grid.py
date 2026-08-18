#!/usr/bin/env python
"""DevAI grid driver: brain alignment + isolation + mechanistic metrics per checkpoint.

For each checkpoint of a model-zoo family, in ONE hook-based forward pass on the
ds003604 localizer stimuli, compute:

  M1  brain-LM alignment (RSA)      -> alignment_<family>.csv   [step, task, session, rsa]
  M2  LM isolation (localization)   -> isolation_<family>.csv   [step, phenomenon, gini, ...]
  M3  pico-analyze mechanistic      -> mechanistic_<family>.csv [step, norm, gini, hoyer, per, ...]

Uses the hook-based ActivationExtractor / CircuitLocalizer (works for GPT-2, GPT-NeoX,
and PicoDecoderHF pico-lm/Beetle checkpoints alike — the RDM `AutoModel`+hidden_states
path does NOT, since PicoDecoderHF returns logits only).

Downstream, scripts/mechanistic_brain_analysis.py joins these three with the brain
specialization table to test the correlations (alignment~mechanistic; LM~brain isolation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.language_models.babylm_integration import ModelZoo  # noqa: E402
from src.language_models.circuit_localization import (  # noqa: E402
    ActivationExtractor,
    CircuitLocalizer,
    load_contrast_csv,
    specialization_summary,
)
from src.language_models.mechanistic_metrics import (  # noqa: E402
    SCALAR_METRICS,
    checkpoint_metrics,
)

try:
    from src.rsa import z_normalize_rdm
except Exception:  # pragma: no cover - fallback if helper unavailable
    def z_normalize_rdm(rdm):
        m, s = np.nanmean(rdm), np.nanstd(rdm)
        return (rdm - m) / s if s else rdm - m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model-zoo family (configs/model_zoo.yaml)")
    p.add_argument("--model-zoo", default="configs/model_zoo.yaml")
    p.add_argument("--contrast-dir", default="contrasts")
    p.add_argument("--phenomena", nargs="+", default=["Sem", "Phon", "Gram", "Plaus"])
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Tasks for RSA (default = phenomena)")
    p.add_argument("--sessions", nargs="+", default=["ses-5", "ses-7", "ses-9"])
    p.add_argument("--brain-rdm-root", default="data/processed/fmri",
                   help="Per-task brain RDMs under <root>/<Task>/session_rdm_<ses>.npz")
    p.add_argument("--layer", type=int, default=-1, help="Block index for the LM RDM")
    p.add_argument("--rdm-pooling", default="mean", choices=["mean", "last-token", "sum"])
    p.add_argument("--pooling", default="last-token", choices=["last-token", "mean", "sum"],
                   help="Pooling for the localizer (M2), matches syntax-units")
    p.add_argument("--percentage", type=float, default=1.0, help="Top-%% units in the circuit")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--normalize", action="store_true", help="z-normalize RDMs before RSA")
    p.add_argument("--max-checkpoints", type=int, default=0,
                   help="If >0, log-subsample to about this many checkpoints")
    p.add_argument("--output-dir", default="data/processed/language_models/devai_grid")
    return p.parse_args()


def _subsample(checkpoints: list, k: int) -> list:
    """Log-spaced subsample to ~k checkpoints (keep first and last)."""
    if k <= 0 or len(checkpoints) <= k:
        return checkpoints
    n = len(checkpoints)
    idx = np.unique(np.round(np.geomspace(1, n, k)).astype(int) - 1)
    idx = sorted(set(idx.tolist()) | {0, n - 1})
    return [checkpoints[i] for i in idx]


def _lm_rdm(acts_layer: np.ndarray) -> np.ndarray:
    """Correlation-distance RDM (1 - Pearson) across stimuli from [n, H] activations."""
    c = np.corrcoef(acts_layer)
    return 1.0 - c


def _load_brain(brain_root: str, task: str, session: str):
    f = Path(brain_root) / task / f"session_rdm_{session}.npz"
    if not f.exists():
        # also accept a flat layout (root/session_rdm_<ses>.npz)
        alt = Path(brain_root) / f"session_rdm_{session}.npz"
        f = alt if alt.exists() else f
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    stimuli = [str(s) for s in d["stimuli"]] if "stimuli" in d else None
    return {"rdm": d["rdm"], "stimuli": stimuli}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    family = args.model
    tasks = args.tasks or args.phenomena

    zoo = ModelZoo(args.model_zoo)
    checkpoints = zoo.resolve_checkpoints(family)
    checkpoints = _subsample(checkpoints, args.max_checkpoints)
    contrasts = {ph: load_contrast_csv(ph, str(Path(args.contrast_dir) / f"{ph}.csv"))
                 for ph in args.phenomena}

    # probe set for mechanistic metrics: dedup union of all contrast stimuli
    probe = []
    for c in contrasts.values():
        probe.extend(list(c.positive) + list(c.negative))
    probe = list(dict.fromkeys(probe))

    # preload brain RDMs per task
    brain = {t: {s: _load_brain(args.brain_rdm_root, t, s) for s in args.sessions}
             for t in tasks}

    print(f"> {family}: {len(checkpoints)} checkpoints | tasks={tasks} | probe={len(probe)}")

    align_rows, iso_rows, mech_rows, mech_layer_rows = [], [], [], []
    prev_probe_acts = None

    for ck in checkpoints:
        ref, step, tokens = ck["ref"], ck.get("step"), ck.get("tokens")
        print(f"\n=== {ref} (step={step}) ===")
        try:
            ex = ActivationExtractor(ref)
        except Exception as e:  # keep the sweep going
            print(f"  ! failed to load {ref}: {e}")
            continue

        # ---- M2: LM isolation (circuit localization) ----------------------
        try:
            loc = CircuitLocalizer(ex, percentage=args.percentage,
                                   pooling=args.pooling, batch_size=args.batch_size)
            results = {ph: loc.localize(c, step=step, tokens=tokens)
                       for ph, c in contrasts.items()}
            summ = specialization_summary(results)
            summ.insert(0, "step", step)
            summ.insert(0, "model_ref", ref)
            summ.insert(0, "family", family)
            # attach per-phenomenon gini/entropy from localize metrics
            gmap = {ph: results[ph].metrics.get("gini") for ph in results}
            summ["gini"] = summ["phenomenon"].map(gmap)
            iso_rows.append(summ)
        except Exception as e:
            print(f"  ! isolation failed: {e}")

        # ---- M1: brain-LM alignment (RSA) ---------------------------------
        for task in tasks:
            bt = brain.get(task, {})
            # gather this task's stimulus list from any available session
            stim = next((bt[s]["stimuli"] for s in args.sessions
                         if bt.get(s) and bt[s]["stimuli"]), None)
            if stim is None:
                continue
            try:
                acts = ex.extract(stim, args.rdm_pooling, args.batch_size)  # [n, L, H]
                lm_rdm = _lm_rdm(acts[:, args.layer, :])
            except Exception as e:
                print(f"  ! RSA extract failed ({task}): {e}")
                continue
            for session in args.sessions:
                b = bt.get(session)
                if not b:
                    continue
                brdm = b["rdm"]
                if brdm.shape != lm_rdm.shape:
                    print(f"  ~ shape mismatch {task}/{session}: LM {lm_rdm.shape} vs brain {brdm.shape}")
                    continue
                a = z_normalize_rdm(lm_rdm) if args.normalize else lm_rdm
                bb = z_normalize_rdm(brdm) if args.normalize else brdm
                iu = np.triu_indices_from(a, k=1)
                rsa, _ = spearmanr(a[iu], bb[iu])
                align_rows.append({"family": family, "model_ref": ref, "step": step,
                                   "tokens": tokens, "task": task, "session": session,
                                   "rsa": float(rsa), "n_stim": int(a.shape[0])})

        # ---- M3: mechanistic metrics --------------------------------------
        try:
            pa = ex.extract(probe, "mean", args.batch_size)  # [n_probe, L, H]
            m = checkpoint_metrics(pa, prev_probe_acts)
            prev_probe_acts = pa
            row = {"family": family, "model_ref": ref, "step": step, "tokens": tokens}
            row.update({k: m[k] for k in SCALAR_METRICS})
            mech_rows.append(row)
            # per-layer rows (for the layer x training-stage heatmap, Fig 3)
            for li in range(len(m["per_by_layer"])):
                mech_layer_rows.append({
                    "family": family, "step": step, "layer": li,
                    "per": float(m["per_by_layer"][li]),
                    "gini": float(m["gini_by_layer"][li]),
                    "hoyer": float(m["hoyer_by_layer"][li]),
                    "norm": float(m["norm_by_layer"][li]),
                    "condition_number": float(m["condition_number_by_layer"][li]),
                })
        except Exception as e:
            print(f"  ! mechanistic failed: {e}")

        del ex
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- write outputs ----------------------------------------------------
    def _save(rows, name, sort_cols):
        if not rows:
            print(f"  (no rows for {name})")
            return None
        df = pd.concat(rows, ignore_index=True) if isinstance(rows[0], pd.DataFrame) \
            else pd.DataFrame(rows)
        df = df.sort_values(sort_cols)
        path = out / f"{name}_{family}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path}  ({len(df)} rows)")
        return path

    _save(align_rows, "alignment", ["task", "session", "step"])
    _save(iso_rows, "isolation", ["phenomenon", "step"])
    _save(mech_rows, "mechanistic", ["step"])
    _save(mech_layer_rows, "mechanistic_layer", ["step", "layer"])
    print("\nDone.")


if __name__ == "__main__":
    main()
