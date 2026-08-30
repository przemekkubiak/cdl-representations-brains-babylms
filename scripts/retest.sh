#!/usr/bin/env bash
# Instrument test-retest: how much does `rsa` move when nothing scientific changes?
#
# Every conclusion in this collection is a comparison BETWEEN families, and the
# between-family sd of family-mean rsa is only 0.005-0.016 per cell. If re-running
# the identical model on the identical stimuli moves rsa by a comparable amount,
# then a family ranking is not a measurement. The already-completed bf16/fp32 A/B
# suggests this is live: mean |delta| rises with training step to ~0.003.
#
# Four arms, same family, same cells, same checkpoints. Only nuisance parameters
# differ, none of which should change the mathematics.
set -uo pipefail
ROOT=/local/scratch/sas245/brainalign-evals
cd "$ROOT"
# HF_HOME deliberately unset: sweep.sh owns its own sweep-local cache.
log(){ echo "[retest $(date -u +%FT%TZ)] $*" | tee -a logs/retest.log; }

FAM="${FAM:-pythia-410m-full}"
log "start: family=$FAM"

run_arm () {  # run_arm <label> <gpu> <batch> [extra env]
  local label="$1" gpu="$2" bs="$3"; shift 3
  if [ -d "grid_retest_${label}" ] && [ -n "$(ls grid_retest_${label}/*/alignment_*.csv 2>/dev/null)" ]; then
    log "arm=$label already present, skipping"; return
  fi
  log ">>> arm=$label gpu=$gpu batch=$bs $*"
  env CUDA_VISIBLE_DEVICES="$gpu" BATCH_SIZE="$bs" MAX_CKPT=12 \
      FAMILIES="$FAM" GRID="$ROOT/grid_retest_${label}" "$@" \
      bash scripts/sweep.sh >>"logs/retest_${label}.log" 2>&1
  log "<<< arm=$label rc=$?"
}

# a: the reference re-run -- identical settings to the main sweep
run_arm a_batch16 1 16
# b: batch size down. A no-op mathematically; only kernel reduction order changes.
run_arm b_batch4  1 4
# c: batch size up.
run_arm c_batch32 1 32
# d: different GPU, same everything else. Pure hardware nondeterminism.
run_arm d_gpu2    2 16

log "RETEST ARMS COMPLETE"
/local/scratch/sas245/venvs/mergeability/bin/python scripts/retest_analysis.py \
  >>logs/retest.log 2>&1
log "RETEST DONE rc=$?"
