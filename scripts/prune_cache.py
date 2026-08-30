#!/usr/bin/env python
"""Drop ONLY the checkpoints this sweep itself downloaded.

run_devai_grid.py already evicts each checkpoint after measuring it, so on a
clean run this finds nothing. It exists for the crash case: a family that dies
mid-loop leaves its last revision resident, and on a 99%-full disk a few of
those across a long sweep is the difference between finishing and filling the
volume other jobs are writing to.

SAFETY -- read before changing. An earlier version of this script selected
victims by repo-id PREFIX ("EleutherAI/", "Beetle-HumanScale/", ...). That is
wrong on a shared cache and it did real damage: those prefixes also match models
cached by the user's OTHER projects, and running it deleted 22 GB / 75 revisions
of someone else's cached weights out of /local/scratch/sas245/hf_cache/hub.
Prefixes describe who PUBLISHED a model, never who downloaded it.

So this version deletes only what is named in a manifest that the sweep writes
as it goes: scripts/sweep.sh records every ref it is about to run, and nothing
absent from that file is ever a candidate. A cache entry that predates the
manifest belongs to someone else by definition.

  MANIFEST: configs/downloaded_refs.txt   one "repo@revision" per line
"""
import os
import sys

from huggingface_hub import scan_cache_dir

ROOT = "/local/scratch/sas245/brainalign-evals"
MANIFEST = os.path.join(ROOT, "configs/downloaded_refs.txt")

if not os.path.exists(MANIFEST):
    print("[prune] no manifest -- nothing this sweep is responsible for; doing nothing")
    sys.exit(0)

wanted = set()
for line in open(MANIFEST):
    line = line.strip()
    if line and "@" in line:
        repo, rev = line.split("@", 1)
        wanted.add((repo, rev))
if not wanted:
    print("[prune] manifest empty; doing nothing")
    sys.exit(0)

# SWEEP-LOCAL CACHES ONLY. The shared caches (/local/scratch/sas245/hf_cache,
# huggingface_cache/, .hf/) are deliberately NOT listed and must never be added:
# other jobs and other people depend on them, and scanning them is what caused
# the incident in STATUS.md section 10. sweep.sh points both HF_HOME and the
# extractor's cache_dir at the two directories below, so everything this sweep
# downloads lands here and nothing it deletes can belong to anyone else.
PIPE_CACHE = os.path.join(ROOT, "pipeline/.cache/huggingface")
SWEEP_HF = os.path.join(ROOT, "hf_home")
CANDIDATE_DIRS = [PIPE_CACHE, os.path.join(PIPE_CACHE, "hub"),
                  SWEEP_HF, os.path.join(SWEEP_HF, "hub")]

FORBIDDEN = ("/local/scratch/sas245/hf_cache",
             "/local/scratch/sas245/huggingface_cache",
             "/local/scratch/sas245/.hf")
for _d in CANDIDATE_DIRS:
    assert not any(os.path.realpath(_d).startswith(f) for f in FORBIDDEN), \
        f"refusing to scan shared cache: {_d}"

total = 0
for cdir in CANDIDATE_DIRS:
    if not os.path.isdir(cdir):
        continue
    try:
        info = scan_cache_dir(cdir)
    except Exception:
        continue
    hashes = []
    for r in info.repos:
        if r.repo_type != "model":
            continue
        for rv in r.revisions:
            refs = set(rv.refs) | {rv.commit_hash}
            if any((r.repo_id, ref) in wanted for ref in refs):
                hashes.append(rv.commit_hash)
    if not hashes:
        continue
    strat = info.delete_revisions(*hashes)
    freed = strat.expected_freed_size_str
    strat.execute()
    total += len(hashes)
    print(f"[prune] {cdir}: freed {freed} across {len(hashes)} revisions "
          f"(all named in the manifest)")

if not total:
    print("[prune] nothing resident from this sweep's manifest")
