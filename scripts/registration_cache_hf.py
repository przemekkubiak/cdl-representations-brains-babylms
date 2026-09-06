#!/usr/bin/env python
"""Push/pull the per-(subject, session) registration cache to the Hub.

WHY. build_subject_roi_mask registers EPI->T1->MNI once per (subject, session)
and caches the result under MASK_CACHE_DIR. That registration is the expensive
per-subject step, and -- crucially -- it is ROI-SET-INDEPENDENT: the .npz holds
only `epi_to_t1_affine`, `t1_to_mni_affine` and the domain/codomain grids. The
ROI set only decides which MNI regions get warped THROUGH those affines. So a
cached registration is reusable by every variant (auditory, motor, phonology,
language, all), on any machine, forever.

Without this, standing the cache up again costs a full re-download of the raw
BOLD (ds003604 is ~578 GiB and the pipeline deletes each subject's BOLD the
moment it is preprocessed) plus the registration compute. With it, a fresh
checkout pulls 50 MB and skips straight to preprocessing.

What is pushed, under `registrations/<dataset>/`:
  *_registration.npz   the affines           -- ROI-independent, the valuable bit
  *_registration.json  per-registration metadata (status, dice, translations)
  *_roi-<set>_mask.nii.gz  the warped mask   -- ROI-specific, cheap to redo but free to keep
  *_roi-<set>_qc.png   QC overlay            -- MASKING.md asks for human review of these
  roi_mask_status.csv  the ledger            -- the file that tells you whether a
                       subject was really masked or silently fell back to whole-brain

Usage:
    python scripts/registration_cache_hf.py push --dataset ds003604
    python scripts/registration_cache_hf.py pull --dataset ds003604
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = os.environ.get("REG_CACHE_REPO", "BrainAlign/ds003604-session-rdms")
PREFIX = "registrations"


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path.home() / ".cache/huggingface/token",
              Path("/local/scratch/sas245/.cache/huggingface/token")):
        if p.is_file():
            return p.read_text().strip()
    return None


def cache_dir(dataset: str) -> Path:
    return Path(os.environ.get("MASK_CACHE_DIR",
                               f"data/processed/fmri/{dataset}/_masks"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--dataset", default="ds003604")
    ap.add_argument("--no-qc", action="store_true",
                    help="skip the QC pngs (34 MB of the 50 MB)")
    a = ap.parse_args()

    tok = _token()
    if not tok:
        print("[reg-cache] no HF_TOKEN and no token file; refusing to continue")
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    root = cache_dir(a.dataset)
    repo_dir = f"{PREFIX}/{a.dataset}"

    if a.action == "push":
        if not root.is_dir():
            print(f"[reg-cache] nothing to push: {root} does not exist")
            return 1
        pats = ["*_registration.npz", "*_registration.json", "*_mask.nii.gz",
                "roi_mask_status.csv"]
        if not a.no_qc:
            pats.append("*_qc.png")
        files = sorted({p for pat in pats for p in root.rglob(pat) if p.is_file()})
        if not files:
            print(f"[reg-cache] nothing matched under {root}")
            return 1
        mb = sum(p.stat().st_size for p in files) / 1048576
        print(f"[reg-cache] pushing {len(files)} files ({mb:.1f} MB) -> {REPO}/{repo_dir}")
        api.create_repo(REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(root), path_in_repo=repo_dir, repo_id=REPO,
            repo_type="dataset",
            allow_patterns=[f"**/{p}" for p in pats] + pats,
            commit_message=(f"registration cache for {a.dataset}: ROI-independent "
                            f"EPI->T1->MNI affines + status ledger"))
        print(f"[reg-cache] pushed. Restore with: "
              f"python scripts/registration_cache_hf.py pull --dataset {a.dataset}")
        return 0

    # pull
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    print(f"[reg-cache] pulling {REPO}/{repo_dir} -> {root}")
    tmp = snapshot_download(REPO, repo_type="dataset", token=tok,
                            allow_patterns=[f"{repo_dir}/**"])
    src = Path(tmp) / repo_dir
    if not src.is_dir():
        print("[reg-cache] no cache on the Hub for this dataset")
        return 1
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            dst = root / p.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(p.read_bytes())
                n += 1
    print(f"[reg-cache] restored {n} files into {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
