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
    AblationValidator,
    ActivationExtractor,
    CircuitLocalizer,
    load_contrast_csv,
    specialization_summary,
)
from src.language_models.mechanistic_metrics import (  # noqa: E402
    SCALAR_METRICS,
    checkpoint_metrics,
)
from scipy.stats import kendalltau, pearsonr  # noqa: E402

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
    p.add_argument("--behaviour", action="store_true", default=True,
                   help="Minimal-pair behavioural accuracy per phenomenon (T2.2)")
    p.add_argument("--no-behaviour", dest="behaviour", action="store_false")
    p.add_argument("--ablate", action="store_true",
                   help="Causal test: ablate circuit -> alignment & behaviour drop (T2.1)")
    p.add_argument("--n-random", type=int, default=4, help="random-circuit controls for ablation")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Bootstrap resamples for RSA CI over RDM pairs (T2.5); 0=off")
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


def _rsa(lm_rdm, brdm, normalize=False, bootstrap=0, seed=0):
    """Multi-metric RSA (T2.3 robustness) + optional bootstrap CI (T2.5).

    Returns dict with Spearman/Pearson/Kendall RSA over the upper triangle and,
    if bootstrap>0, a 95% CI on the Spearman value by resampling RDM pairs."""
    a = z_normalize_rdm(lm_rdm) if normalize else lm_rdm
    b = z_normalize_rdm(brdm) if normalize else brdm
    iu = np.triu_indices_from(a, k=1)
    av, bv = a[iu], b[iu]
    out = {"rsa": float(spearmanr(av, bv)[0]),
           "rsa_pearson": float(pearsonr(av, bv)[0]),
           "rsa_kendall": float(kendalltau(av, bv)[0])}
    if bootstrap and len(av) > 3:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(av), len(av))
            r = spearmanr(av[idx], bv[idx])[0]
            if np.isfinite(r):
                vals.append(r)
        if vals:
            out["rsa_lo"], out["rsa_hi"] = (float(np.percentile(vals, 2.5)),
                                            float(np.percentile(vals, 97.5)))
    return out


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
    behav_rows, abl_align_rows, abl_behav_rows = [], [], []
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
        results = {}
        try:
            loc = CircuitLocalizer(ex, percentage=args.percentage,
                                   pooling=args.pooling, batch_size=args.batch_size)
            results = {ph: loc.localize(c, step=step, tokens=tokens)
                       for ph, c in contrasts.items()}
            summ = specialization_summary(results)
            summ.insert(0, "step", step)
            summ.insert(0, "model_ref", ref)
            summ.insert(0, "family", family)
            gmap = {ph: results[ph].metrics.get("gini") for ph in results}
            summ["gini"] = summ["phenomenon"].map(gmap)
            iso_rows.append(summ)
        except Exception as e:
            print(f"  ! isolation failed: {e}")

        # ablation validator reused for behaviour (T2.2) and causal tests (T2.1)
        val = AblationValidator(ex)

        # ---- T2.2: behavioural axis (minimal-pair accuracy per phenomenon) -
        if args.behaviour:
            for ph, c in contrasts.items():
                try:
                    acc = val.minimal_pair_accuracy(c, args.batch_size)
                    behav_rows.append({"family": family, "model_ref": ref, "step": step,
                                       "tokens": tokens, "phenomenon": ph, "mp_accuracy": acc})
                except Exception as e:
                    print(f"  ! behaviour failed ({ph}): {e}")

        # ---- M1: brain-LM alignment (RSA, multi-metric) + T2.1 causal ------
        for task in tasks:
            bt = brain.get(task, {})
            stim = next((bt[s]["stimuli"] for s in args.sessions
                         if bt.get(s) and bt[s]["stimuli"]), None)
            if stim is None:
                continue
            try:
                acts = ex.extract(stim, args.rdm_pooling, args.batch_size)
                lm_rdm = _lm_rdm(acts[:, args.layer, :])
            except Exception as e:
                print(f"  ! RSA extract failed ({task}): {e}")
                continue

            # optional ablated LM RDMs for the causal-alignment test
            abl_rdm = rnd_rdm = None
            if args.ablate and task in results:
                mask = results[task].mask
                try:
                    abl_rdm = _lm_rdm(val.extract_ablated(stim, mask, args.rdm_pooling,
                                                          args.batch_size)[:, args.layer, :])
                    rnd_rdm = _lm_rdm(val.extract_ablated(stim, mask, args.rdm_pooling,
                                                          args.batch_size, random=True)[:, args.layer, :])
                except Exception as e:
                    print(f"  ! ablated RSA failed ({task}): {e}")

            for session in args.sessions:
                b = bt.get(session)
                if not b:
                    continue
                brdm = b["rdm"]
                if brdm.shape != lm_rdm.shape:
                    print(f"  ~ shape mismatch {task}/{session}: {lm_rdm.shape} vs {brdm.shape}")
                    continue
                m = _rsa(lm_rdm, brdm, args.normalize, args.bootstrap, seed=step or 0)
                align_rows.append({"family": family, "model_ref": ref, "step": step,
                                   "tokens": tokens, "task": task, "session": session,
                                   "n_stim": int(lm_rdm.shape[0]), **m})
                if abl_rdm is not None:
                    abl_align_rows.append({
                        "family": family, "step": step, "tokens": tokens, "task": task,
                        "session": session,
                        "rsa_intact": m["rsa"],
                        "rsa_circuit_ablated": _rsa(abl_rdm, brdm, args.normalize)["rsa"],
                        "rsa_random_ablated": _rsa(rnd_rdm, brdm, args.normalize)["rsa"],
                    })

        # ---- T2.1: causal behaviour drop (circuit vs random ablation) ------
        if args.ablate:
            for ph, c in contrasts.items():
                if ph not in results:
                    continue
                try:
                    v = val.validate(c, results[ph].mask, n_random=args.n_random,
                                     batch_size=args.batch_size)
                    v.update({"family": family, "step": step, "tokens": tokens, "phenomenon": ph})
                    abl_behav_rows.append(v)
                except Exception as e:
                    print(f"  ! causal behaviour failed ({ph}): {e}")

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
    _save(behav_rows, "behaviour", ["phenomenon", "step"])
    _save(abl_align_rows, "ablation_alignment", ["task", "step"])
    _save(abl_behav_rows, "ablation_behaviour", ["phenomenon", "step"])
    print("\nDone.")


if __name__ == "__main__":
    main()
