#!/usr/bin/env bash
# Heartbeat for run_brain_all.sh. Reports forward progress every 15 min and
# names a stage as STALLED if neither its log nor its output tree has changed
# for STALL_MIN minutes. It does not kill anything -- the point is that a stall
# becomes visible instead of silently consuming the 8h stage timeout.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
W="$ROOT/logs/brain_watchdog.log"
STALL_MIN="${STALL_MIN:-45}"
say() { echo "[watchdog $(date -u +%FT%TZ)] $*" | tee -a "$W"; }
newest_min() { local d="$1"; [ -e "$d" ] || { echo 99999; return; }
  local t; t=$(find "$d" -type f -newermt "-${STALL_MIN} minutes" 2>/dev/null | head -1)
  [ -n "$t" ] && echo 0 || echo "$STALL_MIN"; }

say "started; stall threshold ${STALL_MIN}m"
while tmux has-session -t brainwaves 2>/dev/null; do
  cur=$(grep -c "^\[waves.*START" logs/brain_waves.log 2>/dev/null || echo 0)
  last=$(grep -E "^\[waves.*(START|OK|FAIL|SKIP)" logs/brain_waves.log 2>/dev/null | tail -1)
  proc=$(find data/processed/fmri -type f -newermt "-${STALL_MIN} minutes" 2>/dev/null | wc -l)
  logs_moved=$(find logs -name "brain_all_*.log" -newermt "-${STALL_MIN} minutes" 2>/dev/null | wc -l)
  if [ "$proc" -eq 0 ] && [ "$logs_moved" -eq 0 ]; then
    say "STALLED: no new artifacts or log lines in ${STALL_MIN}m -- last: $last"
  else
    say "alive: $cur stage(s) started, $proc new artifacts / $logs_moved active logs in ${STALL_MIN}m -- last: $last"
  fi
  sleep 900
done
say "brainall session ended; final status:"
cat logs/brain_all_status.json 2>/dev/null | tee -a "$W"
