#!/usr/bin/env bash
# Autonomous neuroalignment driver. Runs under tmux, survives logout, needs nobody.
#
# ds001894's brain prep was already in flight (orphaned from an earlier driver)
# when this was written, so this script runs the two finished datasets' grids
# first and only starts ds001894 once that in-flight prep has exited -- running
# prepare_brain_rdms.sh twice against the same tree at once would have two
# writers on one pattern directory.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
GPUS="${GPUS:-5,6}"; JOBS="${JOBS:-48}"; MAX_CKPT="${MAX_CKPT:-25}"
log() { echo "[neuro-auto $(date -u +%FT%TZ)] $*" | tee -a logs/neuro_autonomous.log; }

# Liveness that is zombie-aware. PID 1 does not reap on this box, so `kill -0`
# reports a defunct process as alive -- see PICKUP.md; this has produced a false
# ALIVE reading here before.
alive() { local st; st=$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' '); [ -n "$st" ] && [[ "$st" != Z* ]]; }

log "=== phase 1: grids for the datasets whose brain prep is done"
DATASETS="ds002236 ds006239" GPUS="$GPUS" JOBS="$JOBS" MAX_CKPT="$MAX_CKPT" \
  bash run_new_datasets.sh 2>&1 | tee -a logs/neuro_autonomous.log

INFLIGHT="$(pgrep -f 'bash prepare_brain_rdms.sh' | head -1)"
if [ -n "$INFLIGHT" ]; then
  log "=== waiting on in-flight ds001894 brain prep (pid $INFLIGHT)"
  while alive "$INFLIGHT"; do sleep 120; done
  log "=== in-flight brain prep exited"
fi

log "=== phase 2: ds001894 end to end"
DATASETS="ds001894" GPUS="$GPUS" JOBS="$JOBS" MAX_CKPT="$MAX_CKPT" \
  bash run_new_datasets.sh 2>&1 | tee -a logs/neuro_autonomous.log

log "=== ALL NEURO DATASETS COMPLETE"
git add -A && git -c user.name="suchirsalhan" -c user.email="suchirsalhan@gmail.com" \
  commit -q -m "Neuroalignment results: ds002236, ds006239, ds001894" 2>/dev/null
for r in origin przemek; do git push -q "$r" main 2>/dev/null && log "pushed $r"; done
