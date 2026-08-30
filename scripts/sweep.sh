#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BrainAlign neuro-alignment sweep on THIS box.
#
# Wraps the upstream pipeline (pipeline/scripts/run_devai_grid.py). It does NOT
# reimplement the metric -- the RSA estimator, the LM RDM extraction and the
# HF-cache eviction all come from that repo. What this adds is:
#
#   * brain RDMs are PULLED from BrainAlign/ds003604-session-rdms instead of
#     rebuilt from BOLD (hours of CPU + hundreds of transient GB we do not have);
#   * per-dataset --sessions and --tasks, which the upstream launcher does not
#     pass (see STATUS.md "Bug found" -- that omission silently cost ds002236
#     4 of its 6 cells and ds006239 all 8);
#   * resumability at (family x dataset) granularity;
#   * a hard disk floor and single-GPU pinning for a shared box.
#
# Usage:
#   bash scripts/sweep.sh                       # default family order
#   FAMILIES="pythia-70m-full pythia-160m-full" bash scripts/sweep.sh
#   DATASETS="ds003604" MAX_CKPT=8 bash scripts/sweep.sh
#
# Safe to re-run: a (family x dataset) cell whose alignment CSV already exists
# is skipped, so an interrupted sweep resumes where it stopped.
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT=/local/scratch/sas245/brainalign-evals
PIPE="$ROOT/pipeline"
PY="${PY:-/local/scratch/sas245/venvs/mergeability/bin/python}"
RDM_BASE="$ROOT/data/ds003604-session-rdms"

# --- hard constraints for this box ----------------------------------------
# Defaults to GPU 0, which is shared with a training job
# (neural.train, ~8GB) that must not be disturbed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # overridable; the retest arms need GPUs 1 and 2
# SWEEP-LOCAL cache. Deliberately NOT the shared /local/scratch/sas245/hf_cache:
# everything this sweep downloads must live somewhere only this sweep owns, so
# that eviction can never touch another job's weights (STATUS.md section 10).
export HF_HOME="${HF_HOME:-/local/scratch/sas245/brainalign-evals/hf_home}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# run_devai_grid.py evicts each checkpoint from the HF cache after measuring it.
# On a 99%-full disk that is not optional.
unset DEVAI_KEEP_HF_CACHE

DISK_FLOOR_GB="${DISK_FLOOR_GB:-150}"   # abort below this on /local/scratch
MAX_CKPT="${MAX_CKPT:-12}"              # log-subsampled checkpoints per family
BATCH_SIZE="${BATCH_SIZE:-16}"
GRID="${GRID:-$ROOT/grid}"
# RDM variant: the subdirectory of the per-dataset RDM tree to sweep. Defaults to
# the unmasked within-run-normalised set. The ROI-masked rebuilds land beside it as
# roi-language / roi-phonology / roi-all, and each needs its OWN grid dir so a masked
# sweep can never overwrite an unmasked one.
RDM_VARIANT="${RDM_VARIANT:-within-run-normalised}"
ZOO="${ZOO:-$ROOT/configs/model_zoo_extended.yaml}"

# ds006239 FIRST on purpose: its SemLocal cell is the only run x stimulus crossed
# cell in the collection (the confound cannot arise there) and no model has ever
# been scored against it. It is the highest-information cell we have.
DATASETS="${DATASETS:-ds006239 ds003604 ds002236}"

# Per-dataset sessions and RSA tasks, read off the RDM tree itself so a dataset
# gaining a cell needs no edit here.
sessions_for() { find "$RDM_BASE/$1/$RDM_VARIANT" -name 'session_rdm_*.npz' \
    -printf '%f\n' 2>/dev/null | sed -E 's/session_rdm_(.*)\.npz/\1/' | sort -u | tr '\n' ' '; }
tasks_for()    { find "$RDM_BASE/$1/$RDM_VARIANT" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' 2>/dev/null | sort | tr '\n' ' '; }

# --phenomena drives the localizer contrasts, which only exist for the four
# ds003604 phenomena. --tasks drives the brain RSA and is what varies by
# dataset. Keeping them separate is why Orth/SemLocal can be scored at all.
PHENOMENA="${PHENOMENA:-Sem Phon Gram Plaus}"

# Default family order = the prioritisation: scale ladder small->large first
# (the headline curve), then seed replicates for error bars, then PARC.
FAMILIES="${FAMILIES:-
  pythia-70m-full pythia-160m-full pythia-410m-full pythia-1b-full
  pythia-1.4b-full pythia-2.8b-full pythia-6.9b-full pythia-12b-full
  babylm-gpt2-3 babylm-gpt2-5 babylm-gpt2-7 babylm-gpt2
  polypythia-70m-seed1 polypythia-70m-seed2 polypythia-70m-seed3
  polypythia-160m-seed1 polypythia-160m-seed2 polypythia-160m-seed3
  polypythia-410m-seed1 polypythia-410m-seed2 polypythia-410m-seed3
  parc-pythia-seed0 parc-pythia-seed1 parc-pythia-seed2
  parc-mamba-seed0 parc-mamba-seed1 parc-mamba-seed2
  parc-rwkv-seed0 parc-rwkv-seed1 parc-rwkv-seed2
}"

mkdir -p "$ROOT/logs" "$ROOT/results" "$GRID"
LOG="$ROOT/logs/sweep.log"
log() { echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

free_gb() { df -BG --output=avail /local/scratch | tail -1 | tr -dc '0-9'; }

[ -f "$ZOO" ] || { log "no $ZOO -- run scripts/make_model_zoo.py first"; exit 1; }

cd "$PIPE"

log "START | gpu=$CUDA_VISIBLE_DEVICES | HF_HOME=$HF_HOME | free=$(free_gb)GB"
log "datasets: $DATASETS | max_ckpt=$MAX_CKPT | floor=${DISK_FLOOR_GB}GB"

for DS in $DATASETS; do
  RDM_ROOT="$RDM_BASE/$DS/$RDM_VARIANT"
  [ -d "$RDM_ROOT" ] || { log "$DS: no RDMs at $RDM_ROOT -- skipping"; continue; }
  SESSIONS="$(sessions_for "$DS")"
  TASKS="$(tasks_for "$DS")"
  [ -z "$SESSIONS" ] && { log "$DS: no sessions -- skipping"; continue; }
  OUTDIR="$GRID/$DS"; mkdir -p "$OUTDIR"
  log "=== $DS | tasks: $TASKS | sessions: $SESSIONS"

  for FAM in $FAMILIES; do
    CSV="$OUTDIR/alignment_${FAM}.csv"
    if [ -s "$CSV" ]; then
      log "  skip $FAM (done: $(wc -l < "$CSV") lines)"
      continue
    fi
    FREE=$(free_gb)
    if [ "$FREE" -lt "$DISK_FLOOR_GB" ]; then
      log "  ABORT: ${FREE}GB free < floor ${DISK_FLOOR_GB}GB"
      exit 2
    fi
    log "  run $FAM on $DS (free=${FREE}GB)"
    # Record every ref this family is about to pull BEFORE pulling it. This file
    # is the ONLY thing prune_cache.py is allowed to delete -- see the safety
    # note in that script. Without it the pruner is a no-op, which is the
    # correct failure direction on a cache shared with other projects.
    "$PY" - "$FAM" "$ZOO" "$MAX_CKPT" >>"$ROOT/configs/downloaded_refs.txt" <<'PYEOF'
import sys
sys.path.insert(0, "/local/scratch/sas245/brainalign-evals/pipeline")
from src.language_models.babylm_integration import ModelZoo
import numpy as np
fam, zoo_path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
cks = ModelZoo(zoo_path).resolve_checkpoints(fam)
if k and len(cks) > k:
    n = len(cks)
    idx = np.unique(np.round(np.geomspace(1, n, k)).astype(int) - 1)
    cks = [cks[i] for i in sorted(set(idx.tolist()) | {0, n - 1})]
for c in cks:
    print(c["ref"])
PYEOF

    timeout "${CELL_TIMEOUT:-21600}" \
    "$PY" scripts/run_devai_grid.py \
        --model "$FAM" --model-zoo "$ZOO" --dataset "$DS" \
        --contrast-dir contrasts \
        --phenomena $PHENOMENA \
        --tasks $TASKS \
        --sessions $SESSIONS \
        --brain-rdm-root "$RDM_ROOT" \
        --max-checkpoints "$MAX_CKPT" \
        --batch-size "$BATCH_SIZE" \
        --normalize \
        --no-encoding \
        --output-dir "$OUTDIR" \
        >>"$ROOT/logs/grid_${DS}_${FAM}.log" 2>&1
    rc=$?
    if [ -s "$CSV" ]; then
      log "  ok   $FAM/$DS rc=$rc rows=$(( $(wc -l < "$CSV") - 1 )) free=$(free_gb)GB"
    else
      log "  FAIL $FAM/$DS rc=$rc -- see logs/grid_${DS}_${FAM}.log"
    fi
    # Belt and braces: the grid evicts each revision as it goes, but a crash
    # mid-family can leave one behind. Never let the cache grow across families.
    "$PY" "$ROOT/scripts/prune_cache.py" >>"$LOG" 2>&1
  done
  log "=== $DS done"
done

"$PY" "$ROOT/scripts/collect_results.py" >>"$LOG" 2>&1 \
  && log "collected -> results/" || log "collect FAILED"
log "SWEEP COMPLETE | free=$(free_gb)GB"
