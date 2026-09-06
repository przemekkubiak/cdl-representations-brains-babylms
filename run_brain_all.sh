#!/usr/bin/env bash
# run_brain_all.sh -- every remaining ROI/dataset combination, cheapest first,
# with convergence checks between stages.
#
# WHY THE CHECKS. This pipeline has twice produced a job that looked successful
# and did nothing: a sweep that skipped models with existing CSVs and reported a
# 3-cell grid as if it were 12, and an ROI run whose EPI->T1->MNI registration
# failed silently and fell back to WHOLE BRAIN while still writing its output
# under roi-phonology. Neither raised an error. So after every stage this script
# asserts, from the artifacts rather than from the exit code:
#   1. the cell count for that (dataset, ROI) actually increased,
#   2. the masks it built are ROI-sized, not whole-brain-sized,
#   3. the stage is not still sitting at zero after its timeout.
# A stage that fails a check is recorded and SKIPPED, not retried forever, and
# the run continues to the next stage -- one bad dataset must not block the rest.
#
# ORDER IS BY COST. ds003604's images, registrations and masks are already on
# disk, so a new ROI there is compute only. Everything else needs a multi-GB
# streamed download first (ds002236 is 23.2 GB of broken git-annex symlinks
# today; ds006239 and ds001894 are not on disk at all).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/logs/brain_all.log"
LEDGER="$ROOT/logs/brain_all_status.json"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-28800}"      # 8h per stage
mkdir -p logs
log() { echo "[brainall $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

cells_for() {   # cells_for <dataset> <variant>
  "$PY" - "$1" "$2" <<'PYEOF'
import sys, pandas as pd, pathlib
f = pathlib.Path("/local/scratch/sas245/brainalign-evals/results/roi_by_cell.csv")
if not f.exists(): print(0); raise SystemExit
d = pd.read_csv(f)
print(int(((d.dataset == sys.argv[1]) & (d.variant == sys.argv[2])).sum()))
PYEOF
}

mask_ok() {     # mask_ok <dataset> <roi>  -- ROI-sized, not whole-brain fallback
  "$PY" - "$1" "$2" <<'PYEOF'
import sys, pandas as pd, pathlib
ds, roi = sys.argv[1], sys.argv[2]
f = pathlib.Path(f"/local/scratch/sas245/brainalign-evals/pipeline/data/processed/fmri/{ds}/_masks/roi_mask_status.csv")
if not f.exists(): print("NOMASKS"); raise SystemExit
d = pd.read_csv(f); g = d[d.roi_set == roi]
if not len(g): print("NOROWS"); raise SystemExit
ok = int((g.status == "ok").sum())
big = int((g.n_roi_voxels > 60000).sum()) if "n_roi_voxels" in g else 0
print(f"{'FALLBACK' if big else 'OK'} ok={ok}/{len(g)} med_voxels={g.n_roi_voxels.median():.0f}"
      if "n_roi_voxels" in g else f"OK ok={ok}/{len(g)}")
PYEOF
}

record() { "$PY" - "$LEDGER" "$1" "$2" <<'PYEOF'
import json, os, sys
p, k, v = sys.argv[1:4]
d = json.load(open(p)) if os.path.exists(p) else {}
d[k] = v
json.dump(d, open(p, "w"), indent=2, sort_keys=True)
PYEOF
}

stage() {       # stage <dataset> <roi_set>   ("" roi_set = whole brain)
  local ds="$1" roi="${2:-}" key variant before after
  key="${ds}/${roi:-wholebrain}"
  variant="${roi:+roi-$roi}"; variant="${variant:-within-run-normalised}"
  before="$(cells_for "$ds" "$variant")"
  if [ "$before" -gt 0 ]; then
    log "SKIP $key -- already has $before cells"; record "$key" "skip:$before cells"; return 0
  fi
  log "START $key (timeout ${STAGE_TIMEOUT}s)"
  local t0=$SECONDS
  if [ -n "$roi" ]; then
    ROI_SET="$roi" DATASETS="$ds" RDM_CACHE=1 \
      timeout "$STAGE_TIMEOUT" bash run_new_datasets.sh >> "logs/brain_all_${ds}_${roi}.log" 2>&1
  else
    DATASETS="$ds" RDM_CACHE=1 \
      timeout "$STAGE_TIMEOUT" bash run_new_datasets.sh >> "logs/brain_all_${ds}_wb.log" 2>&1
  fi
  local rc=$? mins=$(( (SECONDS - t0) / 60 ))
  # Re-aggregate before checking: the cell table is what downstream reads.
  bash "$ROOT/../refresh_results.sh" >/dev/null 2>&1 || true
  after="$(cells_for "$ds" "$variant")"
  local mk="n/a"; [ -n "$roi" ] && mk="$(mask_ok "$ds" "$roi")"
  if [ "$after" -le "$before" ]; then
    log "FAIL  $key rc=$rc ${mins}m -- cells still $after; masks: $mk"
    record "$key" "FAIL rc=$rc ${mins}m cells=$after masks=$mk"
  elif [[ "$mk" == FALLBACK* ]]; then
    log "FAIL  $key -- masks are whole-brain sized ($mk); NOT a valid ROI result"
    record "$key" "FAIL wholebrain-fallback $mk"
  else
    log "OK    $key rc=$rc ${mins}m cells $before->$after; masks: $mk"
    record "$key" "ok ${mins}m cells=$after masks=$mk"
  fi
}

log "=== brain sweep starting; GPUs $(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"

# 1. cheapest: ds003604 images/registrations are already local, so a new ROI
#    there costs compute only.
stage ds003604 language

# 2. datasets that need their imaging downloaded first. Whole-brain already
#    exists for ds002236/ds006239, so those stages will skip themselves.
for ds in ds002236 ds006239 ds001894; do
  stage "$ds" ""
  for roi in phonology auditory motor language; do
    stage "$ds" "$roi"
  done
done

"$PY" scripts/coverage_matrix.py | tee -a "$LOG"
log "=== BRAIN SWEEP COMPLETE"
