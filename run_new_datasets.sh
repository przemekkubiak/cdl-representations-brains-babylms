#!/usr/bin/env bash
# Full replication on the three neuro datasets that have never been run:
# ds001894 (Lytle 2019), ds006239 (Wang 2025), ds002236 (Lytle 2020).
#
# Per dataset, in order:
#   1. brain prep   -- streamed download -> GLM betas -> within-run-normalised
#                      session RDMs + noise ceilings, raw BOLD deleted per subject
#   2. gate         -- positive control + RDM dimensionality on THOSE RDMs
#   3. model grid   -- all 15 families x every checkpoint against those RDMs
#   4. summary      -- ceiling-normalised tables, seed-null comparison, figures
#   5. publish      -- its OWN HuggingFace dataset, with a README that states the
#                      gate result before any alignment number
#
# WHY THE GATE MATTERS HERE. On ds003604 the positive control failed: nothing
# stimulus-driven correlates with those RDMs because the per-stimulus betas are
# near-degenerate (rank ~3 of 40-48 stimuli), so the language-model null there is
# a property of the measurement rather than of the models. The estimator is
# shared, so the same failure may reproduce on these datasets. Running the
# control per dataset means each published result says which case it is instead
# of presenting an uninterpretable null as a finding.
#
# Safe to re-run: every stage is idempotent and skips finished work.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
. "$ROOT/env_brainalign.sh"

PY="$ROOT/venv/bin/python"
DATASETS="${DATASETS:-ds001894 ds006239 ds002236}"
JOBS="${JOBS:-32}"
MAX_CKPT="${MAX_CKPT:-25}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-350}"
GPUS="${GPUS:-0,1,2}"
LEDGER="${LEDGER:-logs/new_datasets_ledger.json}"
GIT_NAME="suchirsalhan"
GIT_EMAIL="suchirsalhan@gmail.com"

FAMILIES=(pico-decoder-tiny pico-decoder-small pico-decoder-medium pico-decoder-large
          beetle-humanscale-eng beetle-fineweb3-eng
          babylm-gpt2-3 babylm-gpt2-5 babylm-gpt2-7 babylm-gpt2
          pythia-70m-full pythia-160m-full pythia-410m-full pythia-1b-full
          pythia-1.4b-full)

IFS=',' read -ra GPU_ARR <<< "$GPUS"; NGPU=${#GPU_ARR[@]}

mkdir -p logs paper_results
log() { echo "[newds${LOGTAG:-} $(date -u +%FT%TZ)] $*" | tee -a logs/new_datasets.log; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

ledger_set() {  # ledger_set <dataset> <key> <value>
  "$PY" - "$LEDGER" "$1" "$2" "$3" <<'PYEOF'
import json, sys, os
path, ds, key, val = sys.argv[1:5]
d = {}
if os.path.exists(path):
    try: d = json.load(open(path))
    except Exception: d = {}
d.setdefault(ds, {})[key] = val
json.dump(d, open(path, "w"), indent=2, sort_keys=True)
PYEOF
}

tasks_for() {   # PHENOMENON keys, not BIDS task labels -- see comment below.
  # prepare_brain_rdms.sh's PHENOMENA is consumed as FMRIPreprocessor(task=...),
  # which przemek's _resolve_real_tasks() maps through the registry's
  # `phenomena:` block to the real BIDS task(s), and which src/contrast_spec.py
  # CONTRAST_SPECS is keyed by. Handing it raw BIDS task labels (AAWord,
  # AudRhyme, ReadPhon...) finds BOLD runs -- _resolve_real_tasks falls back to
  # the literal name -- but has no contrast spec, so the condition>control
  # contrast is undefined and the RDM is not the one we claim to be building.
  # ds003604 is unaffected either way: its phenomena map to their own names.
  "$PY" - "$1" <<'PYEOF'
import sys, glob, os, re
sys.path.insert(0, ".")
ds = sys.argv[1]
phenomena = []
try:
    from src.datasets import get_dataset
    from src.contrast_spec import CONTRAST_SPECS
    spec = get_dataset(ds)
    have = set(CONTRAST_SPECS.get(spec.contrast_spec, {}))
    # only phenomena that BOTH the registry declares and a contrast spec defines
    phenomena = [p for p in (spec.phenomena or {}) if p in have]
except Exception:
    pass
print(" ".join(phenomena))
PYEOF
}

# ---------------------------------------------------------------- per dataset
for DS in $DATASETS; do
  RDM_ROOT="data/processed/fmri_wrn/$DS"
  GRID_PARENT="data/processed/language_models/devai_grid_wrn"
  OUTDIR="paper_results/$DS"
  mkdir -p "$OUTDIR"

  TASKS=$(tasks_for "$DS")
  [ -z "$TASKS" ] && { log "$DS: no tasks resolvable -- SKIPPING"; ledger_set "$DS" status no_tasks; continue; }
  log "=== $DS START -- tasks: $TASKS"
  ledger_set "$DS" started "$(date -u +%FT%TZ)"
  ledger_set "$DS" tasks "$TASKS"

  # ---- 1. brain prep -------------------------------------------------------
  if [ "$(find "$RDM_ROOT" -name 'session_rdm_*.npz' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "$DS: session RDMs already present -- skipping brain prep"
  else
    log "$DS: brain prep (streamed; floor ${DISK_FLOOR_GB}GB, $(free_gb)GB free)"
    ledger_set "$DS" stage1 running
    DATASET="$DS" PHENOMENA="$TASKS" WITHIN_RUN_NORM=1 RDM_CACHE=0 KEEP_PATTERNS=0 \
      JOBS="$JOBS" DISK_FLOOR_GB="$DISK_FLOOR_GB" BRAIN_RDM_ROOT="$RDM_ROOT" \
      bash prepare_brain_rdms.sh >>"logs/newds_stage1_$DS.log" 2>&1
    rc=$?
    N=$(find "$RDM_ROOT" -name 'session_rdm_*.npz' 2>/dev/null | wc -l)
    log "$DS: brain prep rc=$rc -- $N session RDMs, $(free_gb)GB free"
    ledger_set "$DS" stage1 "$([ "$N" -gt 0 ] && echo ok || echo failed)"
    ledger_set "$DS" n_rdms "$N"
    [ "$N" -eq 0 ] && { log "$DS: no RDMs -- SKIPPING the rest of this dataset"; continue; }
  fi

  # ---- 1b. stimulus texts --------------------------------------------------
  # run_devai_grid.py feeds `stimulus_texts` to the LM; a cell whose texts are
  # empty contributes no alignment row at all. RDMs built before the
  # pair-filename derivation landed in src/rsa/semantic_metadata.py (and any
  # pulled from the Hub cache) have an all-empty column, so annotate in place --
  # the RDM itself is untouched. Idempotent, and a no-op for ds003604.
  "$PY" scripts/backfill_rdm_texts.py --roots "$RDM_ROOT" \
      --characteristics-dir "data/brain/$DS/stimuli/Stimulus_Characteristics" \
      >>"logs/newds_stage1_$DS.log" 2>&1 \
    && log "$DS: stimulus texts present on every RDM" \
    || log "$DS: WARNING -- some RDMs still have no stimulus text (see log); those cells cannot produce alignment rows"

  # ---- 2. ceilings ---------------------------------------------------------
  "$PY" scripts/collect_ceilings.py --rdm-root "$RDM_ROOT" \
      --out "$OUTDIR/ceilings_$DS.csv" >>"logs/newds_stage1_$DS.log" 2>&1 \
    && log "$DS: ceiling table -> $OUTDIR/ceilings_$DS.csv"

  # ---- 3. the gate ---------------------------------------------------------
  log "$DS: positive control (the gate)"
  "$PY" scripts/positive_control.py --rdm-root "$RDM_ROOT" \
      --stimuli "data/brain/$DS/stimuli" --sessions "$(
        find "$RDM_ROOT" -name 'session_rdm_*.npz' -printf '%f\n' 2>/dev/null \
          | sed -E 's/session_rdm_(.*)\.npz/\1/' | sort -u | paste -sd, -)" \
      --compare-root "data/processed/fmri/$DS" \
      --lm-cells "$OUTDIR/alignment_by_cell.csv" \
      --out "$OUTDIR/control" >>"logs/newds_control_$DS.log" 2>&1
  "$PY" scripts/rdm_dimensionality.py --rdm-root "$RDM_ROOT" \
      --sessions "$(find "$RDM_ROOT" -name 'session_rdm_*.npz' -printf '%f\n' 2>/dev/null \
          | sed -E 's/session_rdm_(.*)\.npz/\1/' | sort -u | paste -sd, -)" \
      --out "$OUTDIR/control" >>"logs/newds_control_$DS.log" 2>&1
  GATE=$("$PY" - "$OUTDIR/control/summary.json" <<'PYEOF'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p): print("unknown"); raise SystemExit
d = json.load(open(p))
print("pass" if d.get("n_significant_holm", 0) > 0 else "fail")
PYEOF
)
  log "$DS: GATE = $GATE"
  ledger_set "$DS" gate "$GATE"

  # ---- 4. model grid -------------------------------------------------------
  log "$DS: model grid, ${#FAMILIES[@]} families across GPUs $GPUS"
  ledger_set "$DS" stage_grid running
  pids=()
  for i in "${!GPU_ARR[@]}"; do
    shard=()
    for j in "${!FAMILIES[@]}"; do
      [ $((j % NGPU)) -eq "$i" ] && shard+=("${FAMILIES[$j]}")
    done
    [ ${#shard[@]} -eq 0 ] && continue
    # PHENOMENA must be passed: slurm/run_devai_grid.sh defaults to ds003604's
    # "Sem Phon Gram Plaus", so without this the grid looked for RDMs under
    # Gram/ and Plaus/ (which do not exist here) and missed Orth/ and SemLocal/
    # entirely -- it printed "Tasks: Sem Phon Gram Plaus", found nothing, and
    # reported "(no rows for alignment)" while exiting 0.
    env CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" SKIP_BRAIN=1 ABLATE=0 \
        MAX_CKPT="$MAX_CKPT" DATASET="$DS" BRAIN_RDM_ROOT="$RDM_ROOT" \
        PHENOMENA="$TASKS" GRID_PARENT="$GRID_PARENT" BACKUP=0 \
        bash slurm/run_devai_grid.sh "${shard[@]}" \
        >>"logs/newds_grid_${DS}_gpu${GPU_ARR[$i]}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  NROWS=$(find "$GRID_PARENT/$DS" -name 'alignment_*.csv' 2>/dev/null | wc -l)
  log "$DS: grid done -- $NROWS alignment files"
  ledger_set "$DS" stage_grid "$([ "$NROWS" -gt 0 ] && echo ok || echo failed)"
  ledger_set "$DS" n_alignment_files "$NROWS"

  # ---- 5. summary ----------------------------------------------------------
  if [ "$NROWS" -gt 0 ]; then
    "$PY" scripts/corrected_sweep_summary.py \
        --grid-dir "$GRID_PARENT/$DS" \
        --ceilings "$OUTDIR/ceilings_$DS.csv" \
        --out "$OUTDIR" >>"logs/newds_summary_$DS.log" 2>&1 \
      && log "$DS: summary -> $OUTDIR"
  fi

  # ---- 6. publish to its own HF dataset ------------------------------------
  log "$DS: publishing to its own HuggingFace dataset"
  "$PY" scripts/publish_dataset_results.py --dataset "$DS" \
      --results "$OUTDIR" --gate "$GATE" >>"logs/newds_publish_$DS.log" 2>&1 \
    && log "$DS: published" || log "$DS: PUBLISH FAILED (see logs/newds_publish_$DS.log)"

  ledger_set "$DS" ended "$(date -u +%FT%TZ)"
  ledger_set "$DS" status done
  log "=== $DS DONE"

  git add -A
  if ! git diff --cached --quiet; then
    git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
        commit -q -m "Results for $DS: $NROWS alignment files, control gate: $GATE" \
      && for r in origin przemek; do git push -q "$r" main 2>/dev/null \
           && log "$DS: pushed $r" || log "$DS: push failed $r"; done
  fi
done

log "ALL DATASETS COMPLETE -- ledger: $LEDGER"
