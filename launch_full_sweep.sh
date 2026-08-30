#!/usr/bin/env bash
# Full corrected sweep: noise ceilings + within-run-normalised RDMs, then the model grid.
#
# WHAT CHANGED SINCE THE LAST RUN, and why this is not just a re-run:
#
#   1. Every session RDM is now built with WITHIN-RUN NORMALISATION. The previous
#      results are confounded by scanner run -- in ds003604 each stimulus appears
#      in exactly one run, so "different run" predicted brain dissimilarity at
#      rho +0.49..+0.87 while no stimulus property predicted anything. Measured on
#      a fresh 27-subject cohort, the correction takes that from +0.568 to -0.041.
#
#   2. Every session RDM now CARRIES ITS PER-SUBJECT RDMs, so the noise ceiling is
#      computed at build time and stays recomputable forever. The old release
#      stored only the group RDM, which is why no ceiling could be reported
#      without re-downloading the whole dataset -- the situation this run exists
#      to end. Ceilings are what make a near-zero alignment readable: measured so
#      far, ceiling 0.856 and best-layer alignment 0.027, i.e. 3.1% of achievable.
#
#   3. The Hub RDM cache is force-disabled whenever the correction is on. The
#      cached RDMs are the CONFOUNDED ones; serving one into this run would mix
#      corrected and uncorrected cells in a single results tree.
#
# Corrected RDMs land in data/processed/fmri_wrn/<dataset>/ -- a DIFFERENT tree
# from the confounded data/processed/fmri/, deliberately, so the two can never be
# confused or silently merged.
#
# Usage:  bash launch_full_sweep.sh            # all stages
#         STAGES="1" bash launch_full_sweep.sh # brain only
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
. "$ROOT/env_brainalign.sh"

STAGES="${STAGES:-1 2 3}"
GPUS="${GPUS:-0,1,2,3}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-350}"
JOBS="${JOBS:-32}"
MAX_SUBJECTS="${MAX_SUBJECTS:-0}"          # 0 = every subject
WRN_ROOT="data/processed/fmri_wrn"
LEDGER="logs/sweep_ledger.json"
PY="$ROOT/venv/bin/python"

export CUDA_VISIBLE_DEVICES="$GPUS"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

mkdir -p logs
log() { echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a logs/sweep.log; }
# -BG --output=avail is GNU-only; BSD/macOS df returns "" for it, which
# silently disables every disk-floor check that reads this (see
# prepare_brain_rdms.sh's copy of this fix, 2026-08-30).
free_gb() { df -Pk / | awk 'NR==2 {print int($4/1024/1024)}'; }

ledger_set() {  # ledger_set <stage> <key> <value>
  "$PY" - "$LEDGER" "$1" "$2" "$3" <<'PYEOF'
import json, sys, os
path, stage, key, val = sys.argv[1:5]
d = json.load(open(path)) if os.path.exists(path) else {}
d.setdefault(stage, {})[key] = val
json.dump(d, open(path, "w"), indent=2, sort_keys=True)
PYEOF
}

log "GPUs=$GPUS ($NGPU) jobs=$JOBS floor=${DISK_FLOOR_GB}GB free=$(free_gb)GB stages=$STAGES"
ledger_set driver started "$(date -u +%FT%TZ)"
ledger_set driver gpus "$GPUS"

# --------------------------------------------------------------- stage 1 ----
# Brain prep. CPU + disk bound -- GPUs are IDLE during this stage, and no amount
# of GPU allocation shortens it. This is the long pole: roughly 35-45 min per
# task x session cell at JOBS=32, so ~8-10 h for ds003604's twelve cells.
if [[ " $STAGES " == *" 1 "* ]]; then
  log "STAGE 1: within-run-normalised brain RDMs + noise ceilings (ds003604)"
  ledger_set stage1 status running
  env DATASET=ds003604 \
      PHENOMENA="Sem Phon Gram Plaus" \
      WITHIN_RUN_NORM=1 \
      RDM_CACHE=0 \
      KEEP_PATTERNS=0 \
      JOBS="$JOBS" \
      MAX_SUBJECTS="$MAX_SUBJECTS" \
      DISK_FLOOR_GB="$DISK_FLOOR_GB" \
      BRAIN_RDM_ROOT="$WRN_ROOT/ds003604" \
      bash prepare_brain_rdms.sh >>logs/sweep_stage1_ds003604.log 2>&1
  rc=$?
  N=$(find "$WRN_ROOT/ds003604" -name "session_rdm_ses-*.npz" | wc -l)
  log "STAGE 1 done rc=$rc -- $N corrected session RDMs, $(free_gb)GB free"
  ledger_set stage1 status "$([ "$N" -gt 0 ] && echo ok || echo failed)"
  ledger_set stage1 rdms "$N"

  # Ceiling table across every corrected cell, straight out of the saved RDMs.
  "$PY" scripts/collect_ceilings.py --rdm-root "$WRN_ROOT/ds003604" \
      --out paper_results/ceiling/ceilings_ds003604.csv \
      >>logs/sweep_stage1_ds003604.log 2>&1 \
      && log "ceiling table -> paper_results/ceiling/ceilings_ds003604.csv"
fi

# --------------------------------------------------------------- stage 2 ----
# The existing ten families against the CORRECTED RDMs. Families are sharded
# across GPUs; the grid itself is single-GPU per process.
run_grid_sharded() {   # run_grid_sharded <tag> <max_ckpt> <ablate> <families...>
  local tag="$1" maxckpt="$2" ablate="$3"; shift 3
  local fams=("$@") pids=()
  for i in "${!GPU_ARR[@]}"; do
    local shard=()
    for j in "${!fams[@]}"; do
      [ $((j % NGPU)) -eq "$i" ] && shard+=("${fams[$j]}")
    done
    [ ${#shard[@]} -eq 0 ] && continue
    log "  gpu ${GPU_ARR[$i]} <- ${shard[*]}"
    env CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" \
        SKIP_BRAIN=1 ABLATE="$ablate" MAX_CKPT="$maxckpt" \
        DATASET=ds003604 \
        BRAIN_RDM_ROOT="$WRN_ROOT/ds003604" \
        GRID_PARENT="data/processed/language_models/devai_grid_wrn" \
        BACKUP=0 \
        bash slurm/run_devai_grid.sh "${shard[@]}" \
        >>"logs/sweep_${tag}_gpu${GPU_ARR[$i]}.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  return $fail
}

if [[ " $STAGES " == *" 2 "* ]]; then
  log "STAGE 2: model grid vs CORRECTED RDMs (10 existing families)"
  ledger_set stage2 status running
  run_grid_sharded tier_existing 25 0 \
      pico-decoder-tiny pico-decoder-small pico-decoder-medium pico-decoder-large \
      beetle-humanscale-eng beetle-fineweb3-eng \
      babylm-gpt2-3 babylm-gpt2-5 babylm-gpt2-7 babylm-gpt2
  rc=$?
  log "STAGE 2 done rc=$rc"
  ledger_set stage2 status "$([ $rc -eq 0 ] && echo ok || echo partial)"
fi

# --------------------------------------------------------------- stage 3 ----
# The scale ladder. A null over 11M-1B models reads as "undertrained"; the Pythia
# ladder is the cheapest answer to that and is already defined in the model zoo,
# yet has never once appeared in the results.
if [[ " $STAGES " == *" 3 "* ]]; then
  log "STAGE 3: Pythia scale ladder vs CORRECTED RDMs"
  ledger_set stage3 status running
  run_grid_sharded scale 20 0 \
      pythia-70m-full pythia-160m-full pythia-410m-full pythia-1b-full pythia-1.4b-full
  rc=$?
  log "STAGE 3 done rc=$rc"
  ledger_set stage3 status "$([ $rc -eq 0 ] && echo ok || echo partial)"
fi

ledger_set driver ended "$(date -u +%FT%TZ)"
log "SWEEP COMPLETE -- ledger: $LEDGER"
