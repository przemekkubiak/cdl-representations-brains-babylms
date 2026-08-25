#!/usr/bin/env bash
# Start the PARC sweep the moment stage 1 has produced all corrected cells.
#
# Stage 1 is CPU/disk bound and leaves the GPUs idle for hours; stage 2/3 then use
# GPUs 0-3. PARC runs on 4-7, so it overlaps stage 2/3 rather than queueing behind
# them. launch_parc_sweep.sh re-checks cell count AND provenance itself, so this
# watcher cannot start a run against confounded or incomplete RDMs.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"

WRN_ROOT="data/processed/fmri_wrn/ds003604"
REQUIRE_CELLS="${REQUIRE_CELLS:-12}"
GPUS="${GPUS:-4,5,6,7}"
POLL="${POLL:-300}"
MAX_WAIT="${MAX_WAIT:-43200}"     # 12 h

log() { echo "[parc-watch $(date -u +%FT%TZ)] $*" | tee -a logs/parc_watcher.log; }

log "waiting for $REQUIRE_CELLS corrected cells under $WRN_ROOT (poll ${POLL}s, max ${MAX_WAIT}s)"
waited=0
while :; do
  n=$(find "$WRN_ROOT" -name "session_rdm_ses-*.npz" 2>/dev/null | wc -l)
  if [ "$n" -ge "$REQUIRE_CELLS" ]; then
    log "stage 1 complete ($n cells) -- checking GPUs $GPUS are still free"
    busy=0
    for g in ${GPUS//,/ }; do
      used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
      [ "${used:-0}" -gt 2000 ] && { log "  gpu $g in use (${used} MiB) -- not ours to take"; busy=1; }
    done
    if [ "$busy" -eq 1 ]; then
      log "ABORTING: another job appeared on the PARC GPUs. Re-run manually with a free set."
      exit 3
    fi
    log "launching PARC sweep on GPUs $GPUS"
    exec env GPUS="$GPUS" bash launch_parc_sweep.sh
  fi
  [ "$waited" -ge "$MAX_WAIT" ] && { log "TIMED OUT after ${waited}s with $n/$REQUIRE_CELLS cells"; exit 2; }
  sleep "$POLL"; waited=$((waited + POLL))
done
