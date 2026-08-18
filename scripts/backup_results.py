#!/usr/bin/env python
"""Back up experiment results: full dump -> HuggingFace dataset; summaries -> git.

Results (CSVs, figures) are gitignored because they are large/regenerable. This
script provides two backups:

  1. HuggingFace dataset (primary backup): upload the full results tree to a
     dataset repo (e.g. BrainAlign/cdl-devai-results). Needs HF_TOKEN.
  2. Git summary (version-controlled numbers): copy the small text artifacts
     (claim summaries, isolation comparisons, held-out CV, LaTeX tables) into a
     tracked `paper_results/` dir so the paper's numbers live in the repo.

Usage:
  HF_TOKEN=... python scripts/backup_results.py \
      --hf-repo BrainAlign/cdl-devai-results \
      --results data/processed/language_models figures \
      --git-summary-dir paper_results
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def upload_hf(repo: str, folders, private: bool, token: str | None):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    for folder in folders:
        p = Path(folder)
        if not p.exists():
            print(f"  (skip missing {p})")
            continue
        print(f"  uploading {p} -> {repo}:{p.name}/ ...")
        api.upload_folder(repo_id=repo, repo_type="dataset",
                          folder_path=str(p), path_in_repo=p.name,
                          commit_message=f"backup {p.name}")
    print(f"HF dataset backup complete: https://huggingface.co/datasets/{repo}")


def git_summary(results, dst: Path):
    """Copy small, paper-relevant text artifacts into a git-tracked dir."""
    dst.mkdir(parents=True, exist_ok=True)
    patterns = ["devai_summary_*.csv", "isolation_comparison_*.csv",
                "heldout_predictor.csv", "table*.tex", "*_summary*.csv"]
    n = 0
    for folder in results:
        base = Path(folder)
        for pat in patterns:
            for f in base.rglob(pat):
                # preserve dataset subdir structure, flatten the rest
                rel = f.relative_to(base)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                n += 1
    print(f"Copied {n} summary files -> {dst}/ (git add {dst} && commit)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+",
                    default=["data/processed/language_models", "figures"],
                    help="Result dirs to back up")
    ap.add_argument("--hf-repo", default=None,
                    help="HF dataset repo id for the full backup (e.g. BrainAlign/cdl-devai-results)")
    ap.add_argument("--private", action="store_true", help="Create the HF dataset private")
    ap.add_argument("--git-summary-dir", default="paper_results",
                    help="Local dir for the small git-tracked summaries ('' to skip)")
    ap.add_argument("--token", default=None, help="HF token (else $HF_TOKEN / cached login)")
    args = ap.parse_args()

    import os
    token = args.token or os.environ.get("HF_TOKEN")

    if args.git_summary_dir:
        git_summary(args.results, Path(args.git_summary_dir))
    if args.hf_repo:
        upload_hf(args.hf_repo, args.results, args.private, token)
    else:
        print("(no --hf-repo given; skipped HuggingFace upload)")


if __name__ == "__main__":
    main()
