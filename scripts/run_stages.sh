#!/usr/bin/env bash
# Chained stage runner. Waits for any in-flight sweep, then runs the remaining
# stages in the coordinator-approved order:
#
#   (stage 1 already running: pythia 70m..1.4b)
#   PARC          -> the noise-seed NULL. Moved ahead of PolyPythias so every
#                    number from stages 1-4 can be referenced against a matched
#                    baseline instead of against zero. Without this the whole
#                    grid is uninterpretable (see STATUS.md section 2).
#   PolyPythias   -> seed error bars on the scaling curve
#   babylm-gpt2   -> BrainAlign's own in-domain child-scale models
#   2.8b, 6.9b    -> top of the ladder, fp32
#   410m bf16     -> precision A/B against the fp32 410m from stage 1
#   12b bf16      -> LAST and ALONE, bf16 only. Never fp32: 47.7 GB of weights
#                    against a 60 GB cap on a GPU carrying another user's job.
#
# Every stage is resumable and the disk floor is enforced between models.
set -uo pipefail
ROOT=/local/scratch/sas245/brainalign-evals
cd "$ROOT"
PY=/local/scratch/sas245/venvs/mergeability/bin/python
log() { echo "[stages $(date -u +%FT%TZ)] $*" | tee -a logs/stages.log; }

# Stage 1 runs INSIDE this script now. Previously it was a separate launch with
# a wait-loop here, and killing the orchestrator mid-wait let an orphaned child
# sweep start on its own -- one race is better than two processes.
while pgrep -f 'bash scripts/sweep.sh' >/dev/null; do sleep 30; done
log 'starting from stage 1'
log "in-flight sweep finished; starting chained stages"

run () {  # run <label> <extra-env> <families...>
  local label="$1"; shift
  local envs="$1"; shift
  log ">>> $label: $*"
  env $envs FAMILIES="$*" MAX_CKPT=12 bash scripts/sweep.sh \
      >>"logs/stage_${label}.log" 2>&1
  log "<<< $label rc=$? free=$(df -BG --output=avail /local/scratch|tail -1|tr -dc 0-9)GB"
  "$PY" scripts/collect_results.py >>"logs/stages.log" 2>&1
}

# --- stage 1: the scale ladder -------------------------------------------
run ladder "" pythia-70m-full pythia-160m-full pythia-410m-full \
  pythia-1b-full pythia-1.4b-full

# --- the null, first ------------------------------------------------------
run parc "" \
  parc-pythia-seed0 parc-pythia-seed1 parc-pythia-seed2 \
  parc-mamba-seed0 parc-mamba-seed1 parc-mamba-seed2 \
  parc-rwkv-seed0 parc-rwkv-seed1 parc-rwkv-seed2

# --- seed error bars ------------------------------------------------------
run polypythia "" \
  polypythia-70m-seed1 polypythia-70m-seed2 polypythia-70m-seed3 \
  polypythia-160m-seed1 polypythia-160m-seed2 polypythia-160m-seed3 \
  polypythia-410m-seed1 polypythia-410m-seed2 polypythia-410m-seed3

# --- BrainAlign's own models ---------------------------------------------
run babylm "" babylm-gpt2-3 babylm-gpt2-5 babylm-gpt2-7 babylm-gpt2

# --- top of the fp32 ladder ----------------------------------------------
run ladder_top "" pythia-2.8b-full pythia-6.9b-full

# --- precision A/B: same model, both dtypes, same cells -------------------
# Writes into a SEPARATE grid dir so it never collides with the fp32 410m rows.
log ">>> precision A/B: pythia-410m in bf16"
env DEVAI_DTYPE=bfloat16 GRID_SUFFIX=_bf16 FAMILIES="pythia-410m-full" MAX_CKPT=12 \
    GRID="$ROOT/grid_bf16" bash scripts/sweep.sh >>logs/stage_bf16ab.log 2>&1
"$PY" scripts/precision_ab.py >>logs/stages.log 2>&1

log "ALL STAGES BEFORE 12B COMPLETE"
