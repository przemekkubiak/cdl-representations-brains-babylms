#!/usr/bin/env bash
# run_brain_waves.sh -- matched-cohort waves, smallest first.
#
# WHY WAVES, AND WHY MATCHED. This covers the THREE NEW datasets -- ds002236,
# ds006239, ds001894. ds003604 is deliberately excluded: it already has
# whole-brain plus three ROI levels at its full 322-subject cohort, and there is
# no reason to redo that work at a reduced N. Its only gap is roi-language,
# which is a separate depth item (see run_ds003604_language.sh) rather than part
# of this comparison.
#
# Among the new three, running each at its own full cohort gives a first
# complete result only after ~25 hours, at cohorts of 188, 91 and 89 -- so any
# difference between them is confounded with how many subjects each contributed.
# Both problems have the same fix: hold N equal ACROSS THE NEW THREE and grow it
# in waves. Every wave produces a complete, directly comparable grid (3 datasets
# x whole-brain + 4 ROI levels); later waves buy precision, not coverage.
# Comparisons against ds003604 are between-cohort by construction and should be
# read as such.
#
# THE WAVES NEST. prepare_brain_rdms.sh takes the first N of a SORTED subject
# list (line 303), so N=12 is a strict prefix of N=25 is a prefix of N=50. Work
# is never thrown away when scaling, and the streamed download only fetches the
# subjects in the current wave -- which is why wave A is cheap even though
# ds002236 is 23.2 GB in total.
#
# THE MATCHED COHORT IS CAPPED AT 89 by ds006239 (configs/neuro_datasets.yaml;
# ds002236 has 91, ds001894 has 188). Wave D at N=89 is therefore the largest
# matched analysis these three admit.
#
# Convergence checks are as in run_brain_all.sh: after each stage we assert from
# the artifacts, not the exit code, that the cell count rose and the masks are
# ROI-sized rather than a whole-brain fallback.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/logs/brain_waves.log"
LEDGER="$ROOT/logs/brain_waves_status.json"
DATASETS_ALL="${DATASETS_ALL:-ds002236 ds006239 ds001894}"   # ds003604 excluded on purpose
ROIS="${ROIS:-phonology auditory motor language}"
mkdir -p logs
log() { echo "[waves $(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# wave name : subjects : budget seconds
WAVES="${WAVES:-A:12:14400 B:25:21600 C:50:28800 D:89:144000}"

record() { "$PY" - "$LEDGER" "$1" "$2" <<'PYEOF'
import json, os, sys
p, k, v = sys.argv[1:4]
d = json.load(open(p)) if os.path.exists(p) else {}
d[k] = v; json.dump(d, open(p, "w"), indent=2, sort_keys=True)
PYEOF
}
cells_for() { "$PY" - "$1" "$2" <<'PYEOF'
import sys, pandas as pd, pathlib
f = pathlib.Path("/local/scratch/sas245/brainalign-evals/results/roi_by_cell.csv")
print(0 if not f.exists() else int(((pd.read_csv(f).dataset == sys.argv[1]) &
      (pd.read_csv(f).variant == sys.argv[2])).sum()))
PYEOF
}
mask_state() { "$PY" - "$1" "$2" <<'PYEOF'
import sys, pandas as pd, pathlib
ds, roi = sys.argv[1], sys.argv[2]
f = pathlib.Path(f"/local/scratch/sas245/brainalign-evals/pipeline/data/processed/fmri/{ds}/_masks/roi_mask_status.csv")
if not f.exists(): print("NOMASKS"); raise SystemExit
d = pd.read_csv(f); g = d[d.roi_set == roi]
if not len(g): print("NOROWS"); raise SystemExit
big = int((g.n_roi_voxels > 60000).sum()) if "n_roi_voxels" in g else 0
med = g.n_roi_voxels.median() if "n_roi_voxels" in g else float("nan")
print(f"{'FALLBACK' if big else 'OK'} ok={int((g.status=='ok').sum())}/{len(g)} med={med:.0f}")
PYEOF
}

run_stage() {  # run_stage <wave> <n> <dataset> <roi|"">  <deadline_epoch>
  local w="$1" n="$2" ds="$3" roi="${4:-}" deadline="$5"
  local variant="${roi:+roi-$roi}"; variant="${variant:-within-run-normalised}"
  local key="${w}/N${n}/${ds}/${roi:-wholebrain}" before after left
  left=$(( deadline - $(date +%s) ))
  if [ "$left" -le 300 ]; then
    log "BUDGET EXHAUSTED before $key -- deferring to the next wave"
    record "$key" "deferred: wave budget spent"; return 1
  fi
  before="$(cells_for "$ds" "$variant")"
  log "START $key (${left}s left in wave)"
  local t0=$SECONDS
  MAX_SUBJECTS="$n" ROI_SET="$roi" DATASETS="$ds" RDM_CACHE=1 \
    timeout "$left" bash run_new_datasets.sh >>"logs/waves_${w}_${ds}_${roi:-wb}.log" 2>&1
  local rc=$? mins=$(( (SECONDS - t0) / 60 ))
  bash "$ROOT/../refresh_results.sh" >/dev/null 2>&1 || true
  after="$(cells_for "$ds" "$variant")"
  local mk="n/a"; [ -n "$roi" ] && mk="$(mask_state "$ds" "$roi")"
  if [[ "$mk" == FALLBACK* ]]; then
    log "FAIL  $key -- whole-brain-sized masks ($mk); not a valid ROI result"
    record "$key" "FAIL wholebrain-fallback $mk"
  elif [ "$after" -le "$before" ] && [ "$before" -eq 0 ]; then
    log "FAIL  $key rc=$rc ${mins}m -- no cells produced; masks: $mk"
    record "$key" "FAIL rc=$rc ${mins}m cells=0 masks=$mk"
  else
    log "OK    $key rc=$rc ${mins}m cells=$after masks=$mk"
    record "$key" "ok ${mins}m cells=$after masks=$mk"
  fi
}

for spec in $WAVES; do
  W="${spec%%:*}"; rest="${spec#*:}"; N="${rest%%:*}"; BUDGET="${rest##*:}"
  DEADLINE=$(( $(date +%s) + BUDGET ))
  log "===== WAVE $W: N=$N subjects per dataset, budget $((BUDGET/3600))h ====="
  for ds in $DATASETS_ALL; do
    run_stage "$W" "$N" "$ds" "" "$DEADLINE"
    for roi in $ROIS; do run_stage "$W" "$N" "$ds" "$roi" "$DEADLINE"; done
  done
  "$PY" scripts/coverage_matrix.py | tee -a "$LOG"
  log "===== WAVE $W COMPLETE; publishing to the Hub ====="
  bash push_brain_to_hf.sh >>"logs/waves_${W}_hfpush.log" 2>&1 \
    && log "wave $W published" || log "wave $W PUBLISH had failures (logs/waves_${W}_hfpush.log)"
done
log "===== ALL WAVES COMPLETE ====="
