"""Session-RDM cache on the Hub, so the fMRI preprocessing is paid for once.

Why this exists. Turning ds003604 into session RDMs costs hours of CPU and, transiently, hundreds
of GB of voxel patterns -- and the patterns are then thrown away, because the RDMs are the product.
Nothing about that work is machine-specific or run-specific, so repeating it on every fresh
checkout is pure waste, and on a shared box it is waste that competes for disk with whatever else
is running. The RDMs themselves are small.

    python scripts/rdm_cache_hf.py pull --task Sem --session ses-5 --dir <out>
    python scripts/rdm_cache_hf.py push --task Sem --session ses-5 --dir <out>
    python scripts/rdm_cache_hf.py list

`pull` exits 0 only if it actually placed a file, so a caller can branch on it directly. Both are
no-ops without HF_TOKEN rather than errors: the cache is an optimisation, and losing it must never
be the reason a run fails. Layout on the Hub mirrors the local one, `{task}/session_rdm_{ses}.npz`,
so the repo is browsable and a human can drop a file in by hand.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = os.environ.get("RDM_CACHE_REPO", "BrainAlign/ds003604-session-rdms")


def _api():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("[rdm-cache] no HF_TOKEN; cache disabled", file=sys.stderr)
        return None, None
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[rdm-cache] huggingface_hub not installed; cache disabled", file=sys.stderr)
        return None, None
    return HfApi(token=tok), tok


# Cache layout. The original scheme was "{task}/session_rdm_{session}.npz" -- no
# dataset and no correction state -- which is unsafe in two ways now that both
# vary:
#
#   * A within-run-normalised RDM would overwrite the confounded one at the same
#     path, silently replacing published data with something different.
#   * A second dataset's tasks would collide with ds003604's wherever task names
#     coincide.
#
# Paths are therefore namespaced by dataset and variant. Legacy flat paths are
# still READ (so nothing already cached is orphaned) but never WRITTEN, which
# keeps the existing entries intact rather than overwriting them.
#
# ROI_SUBDIR (added 2026-08-30) namespaces by masking level too, for the same
# reason: without it, an ROI_SET=phonology pull for a dataset that already has
# a WHOLE-BRAIN RDM cached under the same dataset/variant/task/session (true
# today for ds002236 and ds003604) would silently HIT that whole-brain entry
# and place it under the phonology-labeled output path -- corrupting the run
# with no error, since `pull` succeeding is exactly what the caller wants to
# see. Caught before it ever ran for real: prepare_brain_rdms.sh's own
# ROI_SUBDIR ("roi-<set>", e.g. "roi-phonology") is passed straight through
# here rather than re-deriving it from ROI_SET, so there is one place that
# turns "phonology" into "roi-phonology" (prepare_brain_rdms.sh), not two
# that could drift apart.
VARIANT_RAW = "raw"
VARIANT_WRN = "within-run-normalised"


def variant_name(within_run_normalized: bool) -> str:
    return VARIANT_WRN if within_run_normalized else VARIANT_RAW


def remote_path(task: str, session: str, dataset: str = "ds003604",
                variant: str = VARIANT_RAW, roi_subdir: str | None = None) -> str:
    if roi_subdir:
        return f"{dataset}/{variant}/{roi_subdir}/{task}/session_rdm_{session}.npz"
    return f"{dataset}/{variant}/{task}/session_rdm_{session}.npz"


def legacy_remote_path(task: str, session: str) -> str:
    """The pre-2026-08-25 flat layout: ds003604, uncorrected, by task only."""
    return f"{task}/session_rdm_{session}.npz"


def rdm_is_normalized(path) -> bool:
    """Read the correction flag off an RDM so pushes are self-labelling.

    Reading the file rather than trusting a CLI flag means a corrected RDM cannot
    be filed under 'raw' (or vice versa) by a caller that forgot to pass it.
    """
    import numpy as np
    try:
        d = np.load(str(path), allow_pickle=True)
        if "within_run_normalized" in d.files:
            return bool(d["within_run_normalized"])
    except Exception:
        pass
    return False


def cmd_pull(a) -> int:
    api, tok = _api()
    if api is None:
        return 1
    dest = Path(a.dir) / f"session_rdm_{a.session}.npz"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[rdm-cache] already local: {dest}")
        return 0
    from huggingface_hub import hf_hub_download
    variant = getattr(a, "variant", None) or VARIANT_RAW
    dataset = getattr(a, "dataset", None) or "ds003604"
    roi_subdir = getattr(a, "roi_subdir", None) or None

    candidates = [remote_path(a.task, a.session, dataset, variant, roi_subdir)]
    # Only the raw ds003604 variant has a legacy flat equivalent; never serve a
    # legacy (uncorrected) file into a request for the corrected variant, AND
    # never into an ROI-restricted request -- the legacy path predates ROI
    # masking entirely and is definitionally whole-brain.
    if dataset == "ds003604" and variant == VARIANT_RAW and not roi_subdir:
        candidates.append(legacy_remote_path(a.task, a.session))

    got = None
    for rp in candidates:
        try:
            got = hf_hub_download(REPO, rp, repo_type="dataset", token=tok, local_dir=a.dir)
            break
        except Exception:
            continue
    if got is None:
        print(f"[rdm-cache] miss {dataset}/{variant}/{roi_subdir or 'whole-brain'}/{a.task}/{a.session}")
        return 1
    got = Path(got)
    # hf_hub_download preserves the remote subdirectory; the pipeline wants the file flat.
    if got.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        got.replace(dest)
        try:
            got.parent.rmdir()
        except OSError:
            pass
    print(f"[rdm-cache] HIT  {a.task}/{a.session} -> {dest} ({dest.stat().st_size/2**20:.1f} MB)")
    return 0


def cmd_push(a) -> int:
    api, tok = _api()
    if api is None:
        return 0                      # never fail a run because the cache is unavailable
    src = Path(a.dir) / f"session_rdm_{a.session}.npz"
    if not src.exists() or src.stat().st_size == 0:
        print(f"[rdm-cache] nothing to push for {a.task}/{a.session}")
        return 0
    dataset = getattr(a, "dataset", None) or "ds003604"
    variant = variant_name(rdm_is_normalized(src))     # read off the file, not the CLI
    roi_subdir = getattr(a, "roi_subdir", None) or None
    rp = remote_path(a.task, a.session, dataset, variant, roi_subdir)
    try:
        api.create_repo(REPO, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(src), path_in_repo=rp,
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"session RDM: {dataset}/{variant}/{roi_subdir or 'whole-brain'}/{a.task}/{a.session}")
    except Exception as e:
        print(f"[rdm-cache] push failed for {rp}: {e}")
        return 0
    print(f"[rdm-cache] pushed {rp} ({src.stat().st_size/2**20:.1f} MB)")
    return 0


def cmd_list(a) -> int:
    api, _ = _api()
    if api is None:
        return 1
    try:
        files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".npz")]
    except Exception as e:
        print(f"[rdm-cache] cannot list {REPO}: {e}")
        return 1
    print(f"{REPO}: {len(files)} cached session RDM(s)")
    for f in sorted(files):
        print("  ", f)
    return 0


def cmd_sync(a) -> int:
    """Upload every local session RDM that is not already cached.

    Exists because the cache push is easy to miss: launch_full_sweep.sh pins
    RDM_CACHE=0 for stage 1, so a full run can finish with twelve corrected RDMs
    on disk and none on the Hub. This is idempotent and safe to re-run at any
    point -- it skips what is already there and never deletes.
    """
    api, tok = _api()
    if api is None:
        return 1
    root = Path(a.root)
    local = sorted(root.rglob("session_rdm_ses-*.npz"))
    if not local:
        print(f"[rdm-cache] no session RDMs under {root}")
        return 0

    # ROI_SUBDIR, inferred from the root itself rather than a separate CLI
    # flag: prepare_brain_rdms.sh always calls this with --root "$RDM_ROOT",
    # and $RDM_ROOT already ends in ".../roi-<set>" whenever ROI_SET is set
    # (run_new_datasets.sh's own RDM_ROOT does too) -- one source of truth
    # for "which masking level is this", not a second one that could
    # disagree with where the files actually live.
    roi_subdir = root.name if root.name.startswith("roi-") else None
    if roi_subdir:
        print(f"[rdm-cache] syncing ROI-restricted RDMs under roi_subdir={roi_subdir!r}")

    try:
        api.create_repo(REPO, repo_type="dataset", exist_ok=True)
        remote = set(api.list_repo_files(REPO, repo_type="dataset"))
    except Exception as e:
        print(f"[rdm-cache] cannot reach {REPO}: {e}")
        return 1

    pushed = skipped = 0
    for f in local:
        task = f.parent.name
        session = f.name.replace("session_rdm_", "").replace(".npz", "")
        variant = variant_name(rdm_is_normalized(f))
        rp = remote_path(task, session, a.dataset, variant, roi_subdir)
        if rp in remote:
            print(f"[rdm-cache] have  {rp}")
            skipped += 1
            continue
        try:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=rp,
                            repo_id=REPO, repo_type="dataset",
                            commit_message=f"session RDM: {a.dataset}/{variant}/{roi_subdir or 'whole-brain'}/{task}/{session}")
            print(f"[rdm-cache] PUSH  {rp} ({f.stat().st_size/2**20:.1f} MB)")
            pushed += 1
        except Exception as e:
            print(f"[rdm-cache] push failed {rp}: {e}")
    print(f"[rdm-cache] sync done: {pushed} uploaded, {skipped} already cached")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pull", "push"):
        s = sub.add_parser(name)
        s.add_argument("--task", required=True)
        s.add_argument("--session", required=True)
        s.add_argument("--dir", required=True)
        s.add_argument("--dataset", default="ds003604",
                       help="neuro dataset accession (see configs/neuro_datasets.yaml)")
        s.add_argument("--variant", default=None,
                       choices=[VARIANT_RAW, VARIANT_WRN],
                       help="pull only: which variant to fetch (push reads it off the file)")
        s.add_argument("--roi-subdir", default=None,
                       help="e.g. 'roi-phonology' -- prepare_brain_rdms.sh's own ROI_SUBDIR value, "
                            "passed through as-is (not re-derived from ROI_SET here). Omit for "
                            "whole-brain, the default. A pull with this set NEVER falls back to "
                            "an unscoped (whole-brain) cache entry -- see the module docstring.")
    sy = sub.add_parser("sync", help="upload every local RDM not already cached")
    sy.add_argument("--root", required=True, help="tree to scan, e.g. data/processed/fmri_wrn/ds003604")
    sy.add_argument("--dataset", default="ds003604")
    sub.add_parser("list")
    a = ap.parse_args()
    return {"pull": cmd_pull, "push": cmd_push, "sync": cmd_sync, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
