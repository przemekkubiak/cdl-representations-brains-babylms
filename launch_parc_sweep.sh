#!/usr/bin/env bash
# PARC sweep: architecture x seed, against the CORRECTED brain RDMs.
#
# https://huggingface.co/collections/jmichaelov/parc-models
# Pythia 160M / Mamba 130M / RWKV 169M, OpenWebText, 4000 steps, 6 seeds each,
# 73 shared checkpoints. All eighteen repos carry an identical checkpoint set.
#
# WHY THIS SUITE IS WORTH GPU TIME, given that our result is a null:
#
#   The SEED axis supplies the null distribution. Six seeds of the same
#   architecture at the same step differ only by initialisation, so the spread of
#   their alignments is what "no effect" looks like on this measurement. An
#   alignment only counts if it clears that spread. For a near-zero finding this
#   is far stronger than a p-value against zero, and it is the equivalence test
#   TODO.md section 2 asks for.
#
#   The ARCHITECTURE axis answers the reviewer's framing question directly --
#   transformer vs state-space vs RNN-like, with data, scale and steps held
#   fixed. Whether brain-LM correspondence resembles recurrence or signal
#   propagation becomes something we measure rather than assert.
#
# Kept SEPARATE from launch_full_sweep.sh on purpose: that script is running, and
# bash reads a script incrementally, so editing one mid-flight can corrupt it.
#
# Run this AFTER stage 1 of the main sweep has produced corrected RDMs -- it
# refuses to start otherwise, because scoring PARC against the confounded RDMs
# would reproduce exactly the mistake this whole effort is correcting.
#
# Usage:  bash launch_parc_sweep.sh            # all 18, GPUs 4-7
#         GPUS=4,5 MAX_CKPT=12 bash launch_parc_sweep.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
. "$ROOT/env_brainalign.sh"

GPUS="${GPUS:-4,5,6,7}"
MAX_CKPT="${MAX_CKPT:-20}"        # log-subsample the 73; 0 = all
WRN_ROOT="data/processed/fmri_wrn/ds003604"
PY="$ROOT/venv/bin/python"

export CUDA_VISIBLE_DEVICES="$GPUS"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

mkdir -p logs
log() { echo "[parc $(date -u +%FT%TZ)] $*" | tee -a logs/parc_sweep.log; }

# Two guards, both learned the hard way on 2026-08-25.
#
# CELL COUNT. A first attempt started when exactly ONE cell existed and would have
# spent GPU-hours producing a one-cell PARC result that had to be thrown away. The
# seed-null analysis needs the full grid, so require all cells up front.
REQUIRE_CELLS="${REQUIRE_CELLS:-12}"
N_RDM=$(find "$WRN_ROOT" -name "session_rdm_ses-*.npz" 2>/dev/null | wc -l)
if [ "$N_RDM" -lt "$REQUIRE_CELLS" ]; then
  log "REFUSING TO START: $N_RDM corrected cells under $WRN_ROOT, need $REQUIRE_CELLS."
  log "  Stage 1 of launch_full_sweep.sh is not finished. Scoring PARC against the"
  log "  confounded RDMs in data/processed/fmri/ would repeat the very error this"
  log "  effort exists to correct. Wait for stage 1, then re-run."
  log "  (override with REQUIRE_CELLS=n if a partial run is genuinely wanted)"
  exit 3
fi

# PROVENANCE. A raw, uncorrected RDM was found sitting in the corrected tree,
# left by an earlier pilot; because prepare_brain_rdms.sh skips cells that already
# exist, it would have been adopted silently as the corrected cell. Verify every
# file actually carries within_run_normalized=True before spending any GPU time.
if ! "$PY" scripts/verify_rdm_provenance.py --rdm-root "$WRN_ROOT"        --expect-within-run-normalized true --require-ceilings        --require-cells "$REQUIRE_CELLS" 2>&1 | tee -a logs/parc_sweep.log | tail -20; then
  log "REFUSING TO START: RDM provenance check failed (see above)."
  exit 4
fi
log "corrected RDMs available and verified: $N_RDM"

FAMILIES=()
for a in pythia mamba rwkv; do
  for s in 0 1 2 3 4 5; do FAMILIES+=("parc-$a-seed$s"); done
done
# Cache every corrected RDM before spending GPU time. launch_full_sweep.sh pins
# RDM_CACHE=0 for stage 1, so a full run otherwise finishes with twelve corrected
# RDMs on disk and none on the Hub -- and the next dataset pays for preprocessing
# all over again. Idempotent: skips what is already cached, deletes nothing.
log "syncing corrected RDMs to the Hub cache"
"$PY" scripts/rdm_cache_hf.py sync --root "$WRN_ROOT" --dataset ds003604 2>&1 \
  | grep -E "PUSH|sync done|failed" | sed "s/^/  /" | tee -a logs/parc_sweep.log

log "families: ${#FAMILIES[@]} | GPUs: $GPUS | max-ckpt: $MAX_CKPT"

pids=()
for i in "${!GPU_ARR[@]}"; do
  shard=()
  for j in "${!FAMILIES[@]}"; do
    [ $((j % NGPU)) -eq "$i" ] && shard+=("${FAMILIES[$j]}")
  done
  [ ${#shard[@]} -eq 0 ] && continue
  log "  gpu ${GPU_ARR[$i]} <- ${shard[*]}"
  env CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" \
      SKIP_BRAIN=1 ABLATE=0 MAX_CKPT="$MAX_CKPT" \
      DATASET=ds003604 \
      BRAIN_RDM_ROOT="$WRN_ROOT" \
      GRID_PARENT="data/processed/language_models/devai_grid_parc" \
      BACKUP=0 \
      bash slurm/run_devai_grid.sh "${shard[@]}" \
      >>"logs/parc_gpu${GPU_ARR[$i]}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
log "PARC grid done (fail=$fail)"

# The analysis this suite exists for.
"$PY" scripts/parc_seed_null.py \
    --grid-dir data/processed/language_models/devai_grid_parc/ds003604 \
    --out paper_results/parc >>logs/parc_sweep.log 2>&1 \
  && log "seed-null analysis -> paper_results/parc"

log "PARC SWEEP COMPLETE"
