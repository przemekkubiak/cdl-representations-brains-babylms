#!/bin/bash
# Multi-day, unattended driver: Tier 1 -> Tier 2 -> Tier 3, pushing after each tier.
# Launch with launch_detached.sh (setsid) so it outlives the ssh session.
#
# OPERATIONAL REALITY THIS ENCODES (see PICKUP.md):
#   * GPUs 0,1,2 are ours. GPU 3 is reserved. GPUs 4-7 run a 96-hour merge sweep for
#     another project -- touching them destroys days of work. Pin comes from env_brainalign.sh.
#   * Disk floor 350 GB. `/` is ONE 1.8T overlay shared with that merge sweep, which aborts
#     itself below 250 GB free. 350 leaves it its own margin. DO NOT LOWER IT.
#   * Every tier is recorded in a ledger so a restart resumes instead of redoing work.
#   * A failing tier is recorded and the driver moves on; one failure must not cost the rest.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
. "$ROOT/env_brainalign.sh"

LEDGER="$ROOT/logs/tier_ledger.json"
DISK_FLOOR_GB=350
PY="$ROOT/venv/bin/python"
mkdir -p logs paper_results

log() { echo "[driver $(date -u +%FT%TZ)] $*"; }

# ---- ledger ------------------------------------------------------------- #
[ -f "$LEDGER" ] || echo '{}' > "$LEDGER"
ledger_set() { # ledger_set <tier_id> <k=v json fragments...>
  "$PY" - "$LEDGER" "$@" <<'PY'
import json,sys
path,tier=sys.argv[1],sys.argv[2]
try: d=json.load(open(path))
except Exception: d={}
e=d.setdefault(tier,{})
for kv in sys.argv[3:]:
    k,v=kv.split("=",1)
    try: v=json.loads(v)
    except Exception: pass
    e[k]=v
json.dump(d,open(path,"w"),indent=2,sort_keys=True)
PY
}
ledger_get() { "$PY" -c "
import json,sys
try: d=json.load(open('$LEDGER'))
except Exception: d={}
print(d.get('$1',{}).get('$2',''))"; }

# ---- guards ------------------------------------------------------------- #
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

check_disk() { # check_disk <label>
  local f; f=$(free_gb)
  if [ "$f" -lt "$DISK_FLOOR_GB" ]; then
    log "ABORT($1): free disk ${f}GB < floor ${DISK_FLOOR_GB}GB (merge sweep needs its 250GB margin)"
    return 1
  fi
  log "disk ok ($1): ${f}GB free"; return 0
}

# GPU memory in MiB on OUR cards only (0,1,2) -- never queries 3-7.
our_gpu_mem() { nvidia-smi --id=0,1,2 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{printf "%d", s+0}'; }

verify_gpu_baseline() {
  local m; m=$(our_gpu_mem)
  if [ "${m:-0}" -gt 2000 ]; then
    log "WARN: GPUs 0-2 hold ${m}MiB with no live worker of ours -- checking for orphans"
    local pids; pids=$(nvidia-smi --id=0,1,2 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    for p in $pids; do
      if [ -n "$p" ] && ps -o comm= -p "$p" >/dev/null 2>&1; then
        log "killing orphan GPU pid $p"; kill -TERM "$p" 2>/dev/null; sleep 5; kill -KILL "$p" 2>/dev/null
      fi
    done
  else
    log "GPU baseline ok: ${m:-0}MiB used on 0-2"
  fi
}

# ---- signal handling: kill the child PROCESS GROUP, then exit ------------ #
CHILD_PGID=""
cleanup() {
  log "signal received -- terminating child process group ${CHILD_PGID:-none}"
  if [ -n "$CHILD_PGID" ]; then
    kill -TERM "-$CHILD_PGID" 2>/dev/null
    sleep 10
    kill -KILL "-$CHILD_PGID" 2>/dev/null
  fi
  ledger_set driver status=\"interrupted\" ended=\"$(date -u +%FT%TZ)\"
  exit 143
}
trap cleanup SIGTERM SIGINT

# ---- failure classification --------------------------------------------- #
classify() { # classify <logfile> <exit_code>
  local lg="$1" rc="$2"
  if [ "$rc" = "124" ]; then echo "timeout_wallclock"; return; fi
  if grep -qiE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED" "$lg" 2>/dev/null; then echo "cuda_oom"; return; fi
  if grep -qiE "device-side assert|CUDA error:" "$lg" 2>/dev/null; then echo "cuda_device_assert"; return; fi
  if grep -qiE "No space left on device" "$lg" 2>/dev/null; then echo "disk_full"; return; fi
  if grep -qiE "ConnectionError|Timeout|Temporary failure|502 Server|503 Server|Read timed out" "$lg" 2>/dev/null; then echo "transient_network"; return; fi
  if [ "$rc" = "0" ]; then echo "none"; else echo "code_or_data"; fi
}

# ---- did the tier actually produce anything? ---------------------------- #
# Exit 0 is necessary but NOT sufficient. slurm/run_devai_grid.sh swallows a
# per-family failure ("! grid failed for $FAM"; continue) and still exits 0, so a
# run in which every single family died looks identical to a good one from the exit
# code alone. A tier counts as ok only if it exited 0 AND wrote its declared outputs
# AND -- for the GPU tiers -- actually touched GPU memory on 0-2. A "GPU tier" that
# never put a byte on a card did not run, whatever it exited with.
GRID_DIR="$ROOT/data/processed/language_models/devai_grid/ds003604"
RDM_ROOT="$ROOT/data/processed/fmri/ds003604"

verify_tier_outputs() { # verify_tier_outputs <tier_id> <peak_mib> ; echo reason or "ok"
  local tier="$1" peak="$2" n
  if [ "$tier" = "0" ]; then
    n=$(find "$RDM_ROOT" -name "session_rdm_ses-*.npz" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo ok || echo "no_session_rdms"
    return
  fi
  n=$(find "$GRID_DIR" -name "mechanistic_*.csv" -size +1c 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then echo "no_grid_outputs"; return; fi
  if [ "${peak:-0}" -le 0 ]; then echo "never_used_gpu"; return; fi
  echo ok
}

# ---- run one tier ------------------------------------------------------- #
run_tier() { # run_tier <tier_id> <cap_seconds>
  local tier="$1" cap="$2" lg="$ROOT/logs/tier${1}.log" rc cls attempt=1
  local prev; prev=$(ledger_get "tier$tier" status)
  if [ "$prev" = "ok" ]; then log "tier$tier already ok in ledger -- resuming past it"; return 0; fi

  check_disk "pre-tier$tier" || { ledger_set "tier$tier" status=\"aborted_disk\" failure=\"disk_floor\"; return 1; }
  verify_gpu_baseline

  while : ; do
    log "=== TIER $tier START (attempt $attempt, cap ${cap}s, log $lg) ==="
    ledger_set "tier$tier" status=\"running\" attempt="$attempt" started=\"$(date -u +%FT%TZ)\" log=\"logs/tier${tier}.log\"
    local t0=$SECONDS

    # own process group so a signal kills the whole tree, not just the wrapper
    setsid timeout --signal=TERM --kill-after=120 "$cap" \
      bash "$ROOT/run_devai_bare.sh" --tier "$tier" >>"$lg" 2>&1 &
    local child=$!
    CHILD_PGID=$child

    # Watchdog, every 60s while the tier runs:
    #  (a) sample OUR gpus, so "was it really on GPU" is answerable after the fact;
    #  (b) enforce the disk floor DURING the tier, not just before it -- the ~150GB
    #      pico-full/Beetle download and the BOLD fetch both happen mid-tier, so a
    #      pre-tier check alone cannot protect the merge sweep's 250GB margin.
    ( peak=0
      while kill -0 "$child" 2>/dev/null; do
        m=$(our_gpu_mem); [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
        echo "$peak" > "$ROOT/logs/tier${tier}.gpupeak"
        fnow=$(free_gb)
        if [ "${fnow:-9999}" -lt "$DISK_FLOOR_GB" ]; then
          log "DISK FLOOR BREACHED mid-tier${tier}: ${fnow}GB < ${DISK_FLOOR_GB}GB -- terminating tier"
          echo "disk_floor_breach" > "$ROOT/logs/tier${tier}.diskabort"
          kill -TERM "-$child" 2>/dev/null; sleep 15; kill -KILL "-$child" 2>/dev/null
          break
        fi
        sleep 60
      done ) &
    local sampler=$!

    wait "$child"; rc=$?
    kill "$sampler" 2>/dev/null; CHILD_PGID=""
    local dur=$((SECONDS-t0))
    local peak; peak=$(cat "$ROOT/logs/tier${tier}.gpupeak" 2>/dev/null || echo 0)
    cls=$(classify "$lg" "$rc")
    if [ -f "$ROOT/logs/tier${tier}.diskabort" ]; then cls="aborted_disk_floor"; rm -f "$ROOT/logs/tier${tier}.diskabort"; fi
    log "=== TIER $tier END rc=$rc (${dur}s) class=$cls peak_gpu_mem_0_2=${peak}MiB ==="

    ledger_set "tier$tier" exit_code="$rc" duration_s="$dur" failure="\"$cls\"" \
        peak_gpu_mem_mib_gpu012="${peak:-0}" ended=\"$(date -u +%FT%TZ)\"

    local verdict; verdict=$(verify_tier_outputs "$tier" "$peak")
    ledger_set "tier$tier" outputs_check="\"$verdict\""
    if [ "$rc" = "0" ] && [ "$verdict" = "ok" ]; then
      ledger_set "tier$tier" status=\"ok\"; log "tier$tier verified ok"; break
    fi
    if [ "$rc" = "0" ]; then
      # exited clean but produced nothing usable -- that is a failure, not a success
      log "tier$tier exited 0 but FAILED verification: $verdict"
      cls="$verdict"
      ledger_set "tier$tier" failure="\"$verdict\""
    fi

    # NEVER retry a corrupt CUDA context; at most one retry for transient fs/network.
    if [ "$cls" = "transient_network" ] && [ "$attempt" -lt 2 ]; then
      log "transient failure -- one retry"; attempt=2; continue
    fi
    ledger_set "tier$tier" status=\"failed\"
    log "tier$tier failed ($cls) -- recording and moving to the next tier"
    break
  done

  verify_gpu_baseline
  # Only publish a tier that actually succeeded. paper_results is pushed to a
  # collaborator's repository as well as ours; a failed run must not land there.
  if [ "$(ledger_get "tier$tier" status)" = "ok" ]; then
    publish "tier$tier"
  else
    log "tier$tier not ok -- NOT publishing (ledger records the failure)"
  fi
  return 0
}

# ---- publish results after each tier ------------------------------------ #
publish() { # publish <label>
  log "publishing $1"
  cp -f "$LEDGER" "$ROOT/paper_results/tier_ledger.json" 2>/dev/null
  git add -A paper_results >/dev/null 2>&1
  if git diff --cached --quiet 2>/dev/null; then log "nothing new to commit for $1"; return 0; fi
  git -c user.name=suchirsalhan -c user.email=suchirsalhan@gmail.com \
      commit -q -m "Results: $1

Automated commit from run_tiers_detached.sh after $1 completed.
Ledger snapshot (status, timings, failure class, peak GPU memory on
GPUs 0-2) travels with the results in paper_results/tier_ledger.json." >/dev/null 2>&1
  if git push -q origin main 2>>"$ROOT/logs/push.log"; then log "pushed $1 -> origin"
  else log "PUSH FAILED to origin for $1 (see logs/push.log)"; fi
  # przemek is someone else's repo; this account may not have write access.
  # A rejection there must never abort the run.
  if git push -q przemek main 2>>"$ROOT/logs/push.log"; then log "pushed $1 -> przemek"
  else log "przemek push rejected/unavailable for $1 -- continuing with origin only"
       ledger_set driver przemek_push=\"rejected\"; fi
}

# ---- main --------------------------------------------------------------- #
log "driver starting, pid $$, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ledger_set driver status=\"running\" pid="$$" started=\"$(date -u +%FT%TZ)\" \
    gpus=\"0,1,2\" disk_floor_gb="$DISK_FLOOR_GB"

"$PY" -c "
import torch;assert torch.cuda.is_available(),'CUDA NOT AVAILABLE'
print('driver CUDA check: True |',torch.cuda.device_count(),'devices |',torch.cuda.get_device_name(0))" \
  || { log "FATAL: torch cannot see a GPU -- refusing to run tiers on CPU"; \
       ledger_set driver status=\"aborted_no_cuda\"; exit 1; }

# Stage 0 must come first: without brain session RDMs the grid emits no alignment
# rows at all (exactly what the smoke run shows). Given its own 24h cap so a slow
# ~578GiB streamed download cannot eat Tier 1's budget.
run_tier 0 86400    # 24h
run_tier 1 43200    # 12h
run_tier 2 64800    # 18h
run_tier 3 172800   # 48h

# Tier 3 second half (cross-dataset) is deliberately NOT run -- unresolved accession.
ledger_set tier3b status=\"skipped\" \
  reason="\"Runbook DATASET=ds00XXXX is a placeholder, not a real accession. scripts/batch_download_bold.py hardcodes every download URL to OpenNeuroDatasets/ds003604 and contrasts are built from ds003604 stimuli, so any other DATASET tag would re-download ds003604 and mislabel it as a second study. Not guessed, not fabricated -- needs a real accession plus a dataset-aware download path.\""

# Final state must distinguish "30 GPU-hours of results" from "everything failed in
# thirteen seconds". A bare DONE is exactly the line someone returning will look for,
# so it is never printed unless the tiers actually succeeded.
OK=0; BAD=0; DETAIL=""
for t in 0 1 2 3; do
  st=$(ledger_get "tier$t" status)
  [ -z "$st" ] && continue
  if [ "$st" = "ok" ]; then OK=$((OK+1)); else BAD=$((BAD+1))
    DETAIL="$DETAIL tier$t=$st($(ledger_get "tier$t" failure))"; fi
done
if [ "$BAD" -eq 0 ] && [ "$OK" -gt 0 ]; then
  FINAL="ALL $OK STAGES OK"; DSTATUS="finished_ok"
elif [ "$OK" -eq 0 ]; then
  FINAL="ALL $BAD STAGES FAILED --$DETAIL"; DSTATUS="finished_all_failed"
else
  FINAL="$BAD of $((OK+BAD)) STAGES FAILED --$DETAIL"; DSTATUS="finished_partial"
fi
ledger_set driver status="\"$DSTATUS\"" summary="\"$FINAL\"" ended=\"$(date -u +%FT%TZ)\"
# publish the ledger itself even on failure, so the record travels; publish() is a
# no-op for artefacts when nothing changed, and per-tier publishing already gated on ok.
publish "final ($FINAL)"
log "$FINAL"
