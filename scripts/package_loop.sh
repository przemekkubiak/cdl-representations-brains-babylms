#!/usr/bin/env bash
# Keep the two new HF dataset repos in step with the sweep.
#
# `run_stages.sh` adds families for hours (PARC -> PolyPythias -> babylm -> the
# fp32 ladder top -> the bf16 A/B -> 12b). Rebuilding only at the end would mean
# a sweep that dies at 03:00 leaves nothing on the Hub, so this rebuilds and
# repushes on a cycle: whatever has been measured is always published.
#
# Rebuild is cheap (pure pandas over CSVs) and the push is diffed by the Hub, so
# an idle cycle costs nothing. Exits once the stage runner is gone AND a cycle
# has produced no new families.
set -uo pipefail
ROOT=/local/scratch/sas245/brainalign-evals
cd "$ROOT"
PY=/local/scratch/sas245/venvs/mergeability/bin/python
export HF_TOKEN="${HF_TOKEN:-$(cat /local/scratch/sas245/hf_cache/token)}"
INTERVAL="${INTERVAL:-1800}"
mkdir -p logs
log() { echo "[pkg $(date -u +%FT%TZ)] $*" | tee -a logs/package_loop.log; }

fam_count() { ls grid/"$1"/alignment_*.csv 2>/dev/null | wc -l; }

PREV_A=-1; PREV_B=-1
while true; do
  A=$(fam_count ds006239); B=$(fam_count ds002236)
  RUNNING=0; pgrep -f 'bash scripts/run_stages.sh' >/dev/null && RUNNING=1

  if [ "$A" != "$PREV_A" ] || [ "$B" != "$PREV_B" ]; then
    log "families: ds006239=$A ds002236=$B (was $PREV_A/$PREV_B) -- rebuilding"
    for D in ds006239 ds002236; do
      # A fresh upstream stat pass per family, so claim tests cover new families.
      rm -rf "devai/$D"
      "$PY" scripts/build_devai_package.py --dataset "$D" --push \
        >>"logs/package_${D}.log" 2>&1
      log "  $D rc=$? -> $(grep -c . "logs/package_${D}.log" 2>/dev/null) log lines"
    done
    PREV_A=$A; PREV_B=$B
  else
    log "no new families (ds006239=$A ds002236=$B), stage runner running=$RUNNING"
    if [ "$RUNNING" = "0" ]; then
      log "stage runner gone and nothing new -- final push done, exiting"
      break
    fi
  fi
  sleep "$INTERVAL"
done
