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
    """Extract numeric step from checkpoint label.

    Returns -1 for the "main" revision so it can be remapped after all
    checkpoints are scanned (to max_step + 1).
    """
    name = label.lower()

    # Special case: main branch/revision
    if name == "main" or name.endswith("@main"):
        return -1

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

    # Hugging Face checkpoints: --hf-repo BrainAlign/gpt2-babylm-9 --hf-revisions checkpoint-01 checkpoint-02
    if args.hf_repo and args.hf_revisions:
        checkpoints.extend([f"{args.hf_repo}@{rev}" for rev in args.hf_revisions])

    # Auto-discover fallback if nothing was passed explicitly.
    if not checkpoints:
        root = Path(args.checkpoint_root)
        auto_patterns = [
            "checkpoint-*",
            "**/checkpoint-*",
            "step-*",
            "**/step-*",
        ]
        for pat in auto_patterns:
            checkpoints.extend([str(m) for m in sorted(root.glob(pat)) if m.is_dir()])

    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
        raise ValueError(
            "No checkpoints resolved. Use --checkpoints or --checkpoints-glob. "
            f"Searched from cwd='{Path.cwd()}' and checkpoint_root='{args.checkpoint_root}'."
        )

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
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="Hugging Face repo id, e.g. BrainAlign/gpt2-babylm-9",
    )
    parser.add_argument(
        "--hf-revisions",
        nargs="*",
        default=None,
        help="Hugging Face revisions/branches/tags, e.g. checkpoint-01 checkpoint-02",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints",
        help="Root directory for auto-discovery when no explicit checkpoints are given",
    )
    parser.add_argument("--task", default="Sem")
    parser.add_argument("--compare-sessions", nargs="+", default=["ses-5", "ses-7", "ses-9"])
    parser.add_argument("--output-dir", default="data/processed/language_models/checkpoint_trajectory")
    parser.add_argument("--brain-rdm-dir", default="data/processed/fmri")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument(
        "--bootstrap-ci",
        action="store_true",
        help="Estimate confidence intervals for RSA correlation via bootstrap",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CI (default: 1000)",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for CI (default: 0.95)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed for bootstrap CI reproducibility",
    )

    args = parser.parse_args()

    checkpoints = resolve_checkpoints(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model variant from HF repo if provided
    model_variant = ""
    if args.hf_repo:
        # Try babylm pattern: "BrainAlign/gpt2-babylm-5" -> "_babylm5"
        match = re.search(r"babylm[-_](\d+)", args.hf_repo)
        if match:
            model_variant = f"_babylm{match.group(1)}"
        else:
            # Try pythia pattern: "EleutherAI/pythia-160m" -> "_pythia160m"
            match = re.search(r"pythia[-_](\d+m)", args.hf_repo)
            if match:
                model_variant = f"_pythia{match.group(1).replace('-', '')}"

    pipeline = LanguageModelPipeline(
        output_dir=str(output_dir),
        brain_rdm_dir=args.brain_rdm_dir,
    )

    rows = []
    raw_steps: list[int] = []
    for idx, ckpt in enumerate(checkpoints):
        raw_step = infer_step(ckpt, idx)
        raw_steps.append(raw_step)
        print(f"\n=== Checkpoint: {ckpt} (step={raw_step}) ===")

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
                bootstrap_ci=args.bootstrap_ci,
                n_bootstrap=args.n_bootstrap,
                ci_level=args.ci_level,
                random_seed=args.random_seed,
            )
            if comp is None:
                rows.append(
                    {
                        "checkpoint": ckpt,
                        "step": raw_step,
                        "brain_session": session,
                        "correlation": np.nan,
                        "p_value": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "checkpoint": ckpt,
                    "step": raw_step,
                    "brain_session": session,
                    "correlation": comp["correlation"],
                    "p_value": comp["p_value"],
                    "ci_lower": comp.get("ci_lower", np.nan),
                    "ci_upper": comp.get("ci_upper", np.nan),
                }
            )

    if not rows:
        raise ValueError("No comparison rows were generated.")

    df = pd.DataFrame(rows)

    # Remap "main" sentinel (-1) to a finite value one step after the max.
    max_step = max([s for s in raw_steps if s >= 0], default=0)
    main_step = max_step + 1
    df["step"] = df["step"].replace(-1, main_step)
    df["step_label"] = df["checkpoint"].apply(
        lambda c: "main" if str(c).lower().endswith("@main") or str(c).lower() == "main" else None
    )
    df["step_label"] = df["step_label"].fillna(df["step"].astype(int).astype(str))
    df = df.sort_values(["brain_session", "step"])

    csv_path = output_dir / f"checkpoint_alignment_trajectory{model_variant}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot trajectory
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for session in sorted(df["brain_session"].dropna().unique()):
        sub = df[df["brain_session"] == session].sort_values("step")
        corr_vals = pd.to_numeric(sub["correlation"], errors="coerce").to_numpy(dtype=float)
        x_vals = sub["step"].to_numpy(dtype=float)
        finite_corr = np.isfinite(corr_vals)

        if not np.any(finite_corr):
            print(f"Skipping line plot for {session}: all correlations are NaN")
            continue

        line = ax.plot(x_vals, corr_vals, marker="o", label=session)[0]
        if args.bootstrap_ci:
            lo = pd.to_numeric(sub["ci_lower"], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(sub["ci_upper"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(lo) & np.isfinite(hi)
            if np.any(valid):
                ax.fill_between(
                    x_vals[valid],
                    lo[valid],
                    hi[valid],
                    alpha=0.15,
                    color=line.get_color(),
                )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Brain-LM RSA correlation")
    ax.set_title("Brain-LM alignment trajectory across checkpoints")
    ax.legend()

    # Use explicit tick positions/labels so final point is shown as "main".
    tick_df = df[["step", "step_label"]].drop_duplicates().sort_values("step")
    ax.set_xticks(tick_df["step"].to_list())
    ax.set_xticklabels(tick_df["step_label"].to_list())
    
    fig.tight_layout()

    fig_path = output_dir / f"checkpoint_alignment_trajectory{model_variant}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
