#!/usr/bin/env python
"""Sweep circuit localization across checkpoints x phenomena.

For a model-zoo family, at every checkpoint, localize each phenomenon's circuit
(Sem / Phon / Gram / Plaus), then measure:
  - per-phenomenon localization  (gini, entropy, layer center-of-mass, ...)
  - cross-phenomenon differentiation (Jaccard overlap matrix, selectivity index)
  - split-half consistency (optional; --cross-validate)

Outputs (under --output-dir):
  localization_trajectory_<family>.csv        # one row per (step, phenomenon)
  overlap_<family>_step<STEP>.csv             # phenomenon x phenomenon Jaccard
  fig_localization_trajectory_<family>.png    # gini / overlap vs training step
  fig_overlap_final_<family>.png              # differentiation heatmap (last ckpt)

Contrasts: one CSV per phenomenon in --contrast-dir, named <Phenomenon>.csv, with
either `positive,negative` columns or syntax-units `stim*` + `+/-` marker layout.
See PRIVATE_NOTES.md §5-6 for the method and its brain analogue.

Example:
  python scripts/run_circuit_localization.py --model pythia-160m \
      --contrast-dir data/contrasts --phenomena Sem Phon Gram Plaus --percentage 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.language_models.babylm_integration import ModelZoo
from src.language_models.circuit_localization import (
    ActivationExtractor,
    AblationValidator,
    CircuitLocalizer,
    PhenomenonContrast,
    load_contrast_csv,
    overlap_matrix,
    specialization_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Model-zoo family (configs/model_zoo.yaml)")
    p.add_argument("--model-zoo", default="configs/model_zoo.yaml")
    p.add_argument("--checkpoints", nargs="*", help="Explicit repo@rev or paths (overrides --model)")
    p.add_argument("--contrast-dir", required=True, help="Dir with <Phenomenon>.csv contrasts")
    p.add_argument("--phenomena", nargs="+", default=["Sem", "Phon", "Gram", "Plaus"])
    p.add_argument("--percentage", type=float, default=1.0, help="Top-%% units in the circuit")
    p.add_argument("--pooling", default="last-token", choices=["last-token", "mean", "sum"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cross-validate", action="store_true", help="Also compute split-half consistency")
    p.add_argument("--ablate", action="store_true",
                   help="Causal validation: ablate the circuit (localize on 1st half, "
                        "score minimal pairs on 2nd half) vs random units")
    p.add_argument("--ablate-n-random", type=int, default=4)
    p.add_argument("--output-dir", default="data/processed/language_models/circuit_localization")
    return p.parse_args()


def _split_half(contrast: PhenomenonContrast):
    """Split a contrast into (localize-half, evaluate-half) keeping pairs aligned."""
    n = len(contrast)
    h = n // 2
    loc = PhenomenonContrast(contrast.name, contrast.positive[:h], contrast.negative[:h])
    ev = PhenomenonContrast(contrast.name, contrast.positive[h:], contrast.negative[h:])
    return loc, ev


def resolve_checkpoints(args) -> list[dict]:
    if args.checkpoints:
        return [{"ref": c, "step": i, "tokens": None, "name": c} for i, c in enumerate(args.checkpoints)]
    if not args.model:
        raise SystemExit("Provide --model <family> or --checkpoints ...")
    return ModelZoo(args.model_zoo).resolve_checkpoints(args.model)


def load_contrasts(contrast_dir: str, phenomena: list[str]) -> dict:
    out = {}
    for ph in phenomena:
        path = Path(contrast_dir) / f"{ph}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing contrast CSV: {path}")
        out[ph] = load_contrast_csv(ph, str(path))
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    family = args.model or "custom"

    checkpoints = resolve_checkpoints(args)
    contrasts = load_contrasts(args.contrast_dir, args.phenomena)
    print(f"> {len(checkpoints)} checkpoints x {len(contrasts)} phenomena")

    traj_rows = []
    last_results = None
    for ck in checkpoints:
        ref, step, tokens = ck["ref"], ck.get("step"), ck.get("tokens")
        print(f"\n=== {ref} (step={step}) ===")
        try:
            ex = ActivationExtractor(ref)
        except Exception as e:  # noqa: BLE001 - keep the sweep going
            print(f"  ! failed to load {ref}: {e}")
            continue
        loc = CircuitLocalizer(ex, percentage=args.percentage,
                               pooling=args.pooling, batch_size=args.batch_size)

        validator = AblationValidator(ex) if args.ablate else None

        results = {}
        for ph, contrast in contrasts.items():
            res = loc.localize(contrast, step=step, tokens=tokens)
            if args.cross_validate:
                res.metrics["consistency"] = loc.cross_validation_consistency(contrast)
            if validator is not None:
                loc_half, ev_half = _split_half(contrast)
                mask = loc.localize(loc_half, step=step, tokens=tokens).mask
                abl = validator.validate(ev_half, mask, n_random=args.ablate_n_random)
                res.metrics.update(abl)
            results[ph] = res

        # per-checkpoint differentiation snapshot
        overlap_matrix(results).to_csv(out / f"overlap_{family}_step{step}.csv")
        summ = specialization_summary(results)
        summ.insert(0, "step", step)
        summ.insert(0, "model_ref", ref)
        traj_rows.append(summ)
        last_results = results

        del ex, loc  # free VRAM between checkpoints
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    if not traj_rows:
        raise SystemExit("No checkpoints produced results.")

    traj = pd.concat(traj_rows, ignore_index=True)
    traj_csv = out / f"localization_trajectory_{family}.csv"
    traj.to_csv(traj_csv, index=False)
    print(f"\nSaved: {traj_csv}")

    _plot_trajectory(traj, out / f"fig_localization_trajectory_{family}.png", family)
    if last_results is not None:
        _plot_overlap(overlap_matrix(last_results),
                      out / f"fig_overlap_final_{family}.png", family)


def _plot_trajectory(traj: pd.DataFrame, path: Path, family: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ph, sub in traj.groupby("phenomenon"):
        sub = sub.sort_values("step")
        ax1.plot(sub["step"], sub["gini"], marker="o", label=ph)
        ax2.plot(sub["step"], sub["mean_overlap_with_others"], marker="o", label=ph)
    ax1.set(xlabel="checkpoint step", ylabel="Gini (localization)",
            title="Specialization ↑ over training")
    ax2.set(xlabel="checkpoint step", ylabel="mean cross-phenomenon overlap",
            title="Differentiation (overlap ↓ = more specialized)")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Circuit localization trajectory — {family}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_overlap(M: pd.DataFrame, path: Path, family: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(M.values, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(M.columns)), M.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(M.index)), M.index)
    for i in range(len(M.index)):
        for j in range(len(M.columns)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if M.values[i, j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, label="Jaccard overlap")
    ax.set_title(f"Circuit overlap (final ckpt) — {family}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
