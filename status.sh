#!/bin/bash
# THE one status command.  bash status.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PID=$(cat logs/driver.pid 2>/dev/null || echo "")
echo "=== BrainAlign tier run: status @ $(date -u +%FT%TZ) ==="
# ZOMBIE-AWARE LIVENESS. `kill -0` returns success for a defunct (Z) process on this
# box, because PID 1 does not reap children -- that has already produced a false
# "ALIVE" reading here. The process state must exist AND not start with Z.
if [ -n "$PID" ]; then
  ST=$(ps -o stat= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -z "$ST" ];        then VERDICT="DEAD (no such pid)"
  elif [ "${ST:0:1}" = "Z" ]; then VERDICT="DEAD (zombie/defunct -- NOT running)"
  else VERDICT="ALIVE (state=$ST)"; fi
  echo "driver pid $PID : $VERDICT"
else
  echo "driver pid      : none recorded (logs/driver.pid missing)"
fi
echo
echo "--- ledger ---"; cat logs/tier_ledger.json 2>/dev/null || echo "(no ledger yet)"
echo
echo "--- disk (floor 350G; merge sweep aborts below 250G) ---"; df -h / | tail -1
echo
echo "--- OUR gpus 0,1,2 (3 reserved; 4-7 = other project's 96h merge sweep, DO NOT TOUCH) ---"
nvidia-smi --id=0,1,2 --query-gpu=index,memory.used,utilization.gpu --format=csv
echo
echo "--- last lines of driver log ---"; tail -15 logs/driver.log 2>/dev/null
for t in 1 2 3; do
  [ -f logs/tier$t.log ] && { echo; echo "--- tier$t log tail ---"; tail -6 logs/tier$t.log; }
done
