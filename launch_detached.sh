#!/bin/bash
# Launch the tier driver detached, so it survives the ssh session dropping.
# setsid puts it in a new session with no controlling terminal: SIGHUP on
# disconnect never reaches it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
mkdir -p logs
if [ -f logs/driver.pid ]; then
  OLD=$(cat logs/driver.pid)
  # zombie-aware: kill -0 succeeds on a defunct pid on this box (PID 1 does not reap)
  if [ -n "$OLD" ] && ps -o stat= -p "$OLD" 2>/dev/null | grep -qv '^Z'; then
    echo "driver already running as pid $OLD -- refusing to start a second one"; exit 1
  fi
fi
setsid nohup bash "$ROOT/run_tiers_detached.sh" >> "$ROOT/logs/driver.log" 2>&1 < /dev/null &
echo $! > "$ROOT/logs/driver.pid"
sleep 2
echo "driver launched, pid $(cat "$ROOT/logs/driver.pid") -> logs/driver.log"
