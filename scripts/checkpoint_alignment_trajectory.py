#!/usr/bin/env python
"""Compute brain-model alignment trajectories across BabyLM checkpoints.

This script evaluates multiple checkpoint directories (or HF revisions/paths),
computes LM RDMs, compares them with brain RDMs, and saves a trajectory CSV/plot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

# Allow imports from repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_language_models import LanguageModelPipeline


STEP_PATTERNS = [
    re.compile(r"checkpoint[-_](\d+)"),
    re.compile(r"step[-_](\d+)"),
    re.compile(r"global[-_]?step[-_](\d+)"),
    re.compile(r"iter[-_](\d+)"),
]


def infer_step(label: str, fallback: int) -> int:
    name = label.lower()
    for pat in STEP_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return fallback


def resolve_checkpoints(args: argparse.Namespace) -> list[str]:
    checkpoints: list[str] = []
    if args.checkpoints:
        checkpoints.extend(args.checkpoints)

    if args.checkpoints_glob:
        matches = sorted(Path().glob(args.checkpoints_glob))
        checkpoints.extend([str(m) for m in matches])

    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
        raise ValueError("No checkpoints resolved. Use --checkpoints or --checkpoints-glob.")

    # De-duplicate while preserving order
    uniq = []
    seen = set()
    for c in checkpoints:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain alignment trajectory over checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Explicit checkpoint model paths/names (space-separated)",
    )
    parser.add_argument(
        "--checkpoints-glob",
        default=None,
        help="Glob pattern for checkpoints, e.g. 'checkpoints/gpt2-babylm-7/checkpoint-*'",
    )
    parser.add_argument("--task", default="Sem")
    parser.add_argument("--compare-sessions", nargs="+", default=["ses-5", "ses-7", "ses-9"])
    parser.add_argument("--output-dir", default="data/processed/language_models/checkpoint_trajectory")
    parser.add_argument("--brain-rdm-dir", default="data/processed/fmri")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--include-controls", action="store_true")

    args = parser.parse_args()

    checkpoints = resolve_checkpoints(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LanguageModelPipeline(
        output_dir=str(output_dir),
        brain_rdm_dir=args.brain_rdm_dir,
    )

    rows = []
    for idx, ckpt in enumerate(checkpoints):
        step = infer_step(ckpt, idx)
        print(f"\n=== Checkpoint: {ckpt} (step={step}) ===")

        lm_result = pipeline.compute_lm_rdm(
            model_name=ckpt,
            task=args.task,
            layer=args.layer,
            pooling=args.pooling,
            save=True,
            exclude_controls=not args.include_controls,
        )

        if lm_result is None:
            print(f"Skipping checkpoint due to LM RDM failure: {ckpt}")
            continue

        lm_rdm = lm_result["rdm"]

        for session in args.compare_sessions:
            comp = pipeline.compare_rdms(
                lm_rdm=lm_rdm,
                brain_session=session,
                task=args.task,
                exclude_controls=not args.include_controls,
            )
            if comp is None:
                rows.append(
                    {
                        "checkpoint": ckpt,
                        "step": step,
                        "brain_session": session,
                        "correlation": np.nan,
                        "p_value": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "checkpoint": ckpt,
                    "step": step,
                    "brain_session": session,
                    "correlation": comp["correlation"],
                    "p_value": comp["p_value"],
                }
            )

    if not rows:
        raise ValueError("No comparison rows were generated.")

    df = pd.DataFrame(rows).sort_values(["brain_session", "step"])
    csv_path = output_dir / "checkpoint_alignment_trajectory.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot trajectory
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for session in sorted(df["brain_session"].dropna().unique()):
        sub = df[df["brain_session"] == session].sort_values("step")
        ax.plot(sub["step"], sub["correlation"], marker="o", label=session)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Brain-LM RSA correlation")
    ax.set_title("Brain-LM alignment trajectory across checkpoints")
    ax.legend()
    fig.tight_layout()

    fig_path = output_dir / "checkpoint_alignment_trajectory.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
