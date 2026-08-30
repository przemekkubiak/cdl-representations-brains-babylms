#!/usr/bin/env bash
# Full replication on the three neuro datasets that have never been run:
# ds001894 (Lytle 2019), ds006239 (Wang 2025), ds002236 (Lytle 2020).
#
# ds003604 (the flagship dataset) is also supported -- pass DATASETS=ds003604
# explicitly (it is NOT in the default DATASETS list, on purpose: this script's
# name and its default are about the three NEW datasets, and ds003604 is not
# new -- an unqualified invocation should not risk starting a many-hour
# ds003604 run for someone who only wanted the others). Everything below
# (tasks_for, the per-dataset loop, ROI_SET scoping) is dataset-generic and
# was verified against ds003604's real registry entry/local checkout
# 2026-08-30 -- no special-casing needed. Two things ARE ds003604-specific in
# practice, not in code:
#   - SCALE. ds003604 is ~575 subject-sessions vs. ~90 for each of the other
#     three. A first ROI_SET run should set MAX_SUBJECTS (e.g. 20-30) to see
#     real numbers in a reasonable time before committing to the full cohort
#     -- see DATASETS.md section 9's local-run example. Whole-brain
#     (ROI_SET unset) is a different story: its within-run-normalised RDMs
#     are already Hub-cached (BrainAlign/ds003604-session-rdms), so
#     RDM_CACHE=1 (this script's default) should make that path a cache pull,
#     not a full recompute -- MAX_SUBJECTS is irrelevant there.
#   - The RDM Hub cache is namespaced by ROI_SET (scripts/rdm_cache_hf.py,
#     fixed 2026-08-30) specifically BECAUSE ds003604 already has whole-brain
#     RDMs cached -- without that fix, an ROI_SET run here would have
#     silently pulled the whole-brain RDM and mislabeled it as
#     ROI-restricted. Already fixed; noted here so nobody "fixes" it again
#     by disabling the cache instead.
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
#
# ROI_SET (env var: "phonology" | "language" | "all", see
# src/preprocessing/roi_atlas.py) -- the three-level masking standard,
# DATASETS.md section 10. Unset (default) = whole-brain, byte-identical to
# this script's behaviour before ROI_SET existed here. This script runs ONE
# level per invocation; run it three times for the full standard:
#   for ROI in phonology language all; do
#     ROI_SET=$ROI DATASETS=ds002236 bash run_new_datasets.sh
#   done
# Every path (RDM root, grid output, published results, logs, ledger keys)
# is scoped by ROI_SET automatically, so the three levels and the
# whole-brain default never collide -- see the per-dataset loop below.
set -uo pipefail

# Same guard as prepare_brain_rdms.sh, and for the same reason: this script
# needs bash >=4 (mapfile, used below for the T1-download subject list) and
# macOS ships 3.2. Re-exec with Homebrew's bash rather than failing deep
# inside with a cryptic error partway through a run.
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
  if [ -x /opt/homebrew/bin/bash ]; then
    exec /opt/homebrew/bin/bash "$0" "$@"
  fi
  echo "run_new_datasets.sh needs bash >=4 (mapfile) -- this is $BASH_VERSION." >&2
  echo "Install with: brew install bash coreutils -- see DATASETS.md section 9." >&2
  exit 1
fi

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
# See prepare_brain_rdms.sh's copy of this function for why -- GNU-only
# flags silently returned "" on macOS, disabling every disk-floor check
# that reads this, not just the log line here.
free_gb() { df -Pk / | awk 'NR==2 {print int($4/1024/1024)}'; }

# ds003604 is ~575 subject-sessions vs. ~90 for each of the other three --
# see the header comment for why this is a warning, not a block: whole-brain
# should be a cheap Hub-cache pull regardless of MAX_SUBJECTS, but an
# ROI_SET run has never been cached for ds003604 at any scale, so it pays
# the full cost. A one-line heads-up here costs nothing and has already
# saved a multi-hour surprise once this session (locally, at a much smaller
# scale, before this dataset was even in scope for this script).
case " $DATASETS " in
  *" ds003604 "*)
    if [ -n "${ROI_SET:-}" ] && [ -z "${MAX_SUBJECTS:-}" ]; then
      log "WARNING: DATASETS includes ds003604 with ROI_SET=$ROI_SET set and no MAX_SUBJECTS -- this will attempt the FULL ~575-subject-session cohort from scratch (never cached at any ROI level). Consider MAX_SUBJECTS=20-30 for a first look; see DATASETS.md section 9."
    fi
    ;;
esac

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

sessions_for() {  # sessions_for <rdm_root> -- comma-separated session labels
  # with a session_rdm_*.npz present anywhere under rdm_root. GNU find's
  # `-printf '%f\n'` is not portable -- BSD/macOS find (no findutils
  # installed) rejects it outright ("illegal option"/"unknown primary"),
  # which lands on stderr and is invisible inside a `2>/dev/null` command
  # substitution: the substitution silently returns "", --sessions gets
  # passed an empty string, and positive_control.py's per-session loop then
  # runs zero times -- not a fail, no rows at all, printing only "no results
  # -- check the RDM root and the stimulus directory" with nothing upstream
  # explaining why. Caught 2026-08-30 on ds003604's first local (macOS) run:
  # the gate logged as "unknown", not "fail", and the real signal (the
  # dimensionality report the very same log ran successfully afterwards --
  # different script, no find in its path) was sitting right next to it.
  # `-exec basename {} \;` is the portable equivalent, identical behaviour
  # on GNU and BSD find.
  find "$1" -name 'session_rdm_*.npz' -exec basename {} \; 2>/dev/null \
    | sed -E 's/session_rdm_(.*)\.npz/\1/' | sort -u | paste -sd, -
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
# ROI_SET (env var: "phonology" | "language" | "all" | "auditory,motor" | ...,
# see src/preprocessing/roi_atlas.py) -- unset (default) = whole-brain,
# BYTE-IDENTICAL to this script's behaviour before ROI_SET existed here.
# When set, every path below (RDM root, grid output, published results,
# logs, ledger keys) gets its own "roi-<set>" subdirectory/suffix, mirroring
# prepare_brain_rdms.sh's OWN internal ROI_SUBDIR convention EXACTLY -- a
# mismatch here would make every step after brain-prep silently look at the
# wrong (or a nonexistent) directory. See DATASETS.md section 10 for the
# three-level standard this exists to run (phonology/language/all), and run
# this script three times, once per ROI_SET value, for the full standard --
# it does not loop over levels itself, so a whole-brain-only invocation
# (ROI_SET unset) is unaffected either way.
ROI_SUBDIR=""; ROI_LABEL="whole-brain"; ROI_TAG=""
if [ -n "${ROI_SET:-}" ]; then
  ROI_SUBDIR="/roi-${ROI_SET//,/+}"
  ROI_LABEL="roi-${ROI_SET//,/+}"
  ROI_TAG="_${ROI_LABEL}"
fi

for DS in $DATASETS; do
  RDM_ROOT_BASE="data/processed/fmri_wrn/$DS"   # unscoped -- for prepare_brain_rdms.sh's
                                                  # BRAIN_RDM_ROOT only; it does its own
                                                  # roi- nesting internally (below).
  RDM_ROOT="$RDM_ROOT_BASE$ROI_SUBDIR"           # what every OTHER step in this script reads/writes.
  GRID_PARENT="data/processed/language_models/devai_grid_wrn$ROI_SUBDIR"
  OUTDIR="paper_results/$DS$ROI_SUBDIR"
  LKEY="$DS${ROI_TAG}"                           # ledger key -- ROI-scoped so a phonology/
                                                  # language/all run never overwrites another
                                                  # level's (or whole-brain's) ledger entry.
  mkdir -p "$OUTDIR"

  TASKS=$(tasks_for "$DS")
  [ -z "$TASKS" ] && { log "$DS ($ROI_LABEL): no tasks resolvable -- SKIPPING"; ledger_set "$LKEY" status no_tasks; continue; }
  log "=== $DS ($ROI_LABEL) START -- tasks: $TASKS"
  ledger_set "$LKEY" started "$(date -u +%FT%TZ)"
  ledger_set "$LKEY" tasks "$TASKS"

  # T1 anatomicals -- ROI masking needs one per subject-session for
  # registration (MASKING.md), and nothing in stage 1 below ever resolves
  # one (batch_download_bold.py only touches func/). Skipped for whole-brain
  # runs, which don't register anything and don't need this.
  if [ -n "${ROI_SET:-}" ]; then
    # Bound the download to (approximately) the subjects prepare_brain_rdms.sh
    # will actually use when MAX_SUBJECTS is set. Without this, download_anat.py
    # (no --subjects passed) fetches T1s for EVERY subject in the dataset,
    # completely ignoring MAX_SUBJECTS -- exactly backwards for a small first
    # look. Caught 2026-08-30: a MAX_SUBJECTS=3 ds003604 run downloaded T1s for
    # all 322 subjects (827 files, 7GB, ~1hr) before prepare_brain_rdms.sh ever
    # looked at MAX_SUBJECTS at all. This is the same "first N sorted subject
    # IDs" heuristic prepare_brain_rdms.sh's own SUBS uses, not a guaranteed
    # match to its per-task, BOLD-driven selection (that's discovered later,
    # per task/session, and T1 download has to happen before any of that runs)
    # -- close enough to bound the cost, and a subject prepare_brain_rdms.sh
    # picks that wasn't pre-downloaded here just falls back to whole-brain
    # masking for that one subject (logged, not raised; see roi_mask_status.csv).
    ANAT_SUBJECTS_ARGS=()
    if [ "${MAX_SUBJECTS:-0}" -gt 0 ]; then
      mapfile -t ANAT_SUBS < <(ls -d "data/brain/$DS"/sub-* 2>/dev/null \
          | xargs -n1 basename | sort -u | head -n "$MAX_SUBJECTS")
      [ "${#ANAT_SUBS[@]}" -gt 0 ] && ANAT_SUBJECTS_ARGS=(--subjects "${ANAT_SUBS[@]}")
    fi
    ANAT_SCOPE_MSG=""
    [ ${#ANAT_SUBJECTS_ARGS[@]} -gt 0 ] && ANAT_SCOPE_MSG=" (bounded to MAX_SUBJECTS=$MAX_SUBJECTS)"
    log "$DS ($ROI_LABEL): downloading T1 anatomicals$ANAT_SCOPE_MSG"
    "$PY" scripts/download_anat.py --dataset "$DS" "${ANAT_SUBJECTS_ARGS[@]}" >>"logs/newds_stage1_${DS}${ROI_TAG}.log" 2>&1 \
      && log "$DS ($ROI_LABEL): T1 anatomicals present" \
      || log "$DS ($ROI_LABEL): WARNING -- T1 download had failures (see log); affected subjects will fall back to whole-brain masking (see roi_mask_status.csv)"
  fi

  # ---- 1. brain prep -------------------------------------------------------
  if [ "$(find "$RDM_ROOT" -name 'session_rdm_*.npz' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "$DS ($ROI_LABEL): session RDMs already present -- skipping brain prep"
  else
    log "$DS ($ROI_LABEL): brain prep (streamed; floor ${DISK_FLOOR_GB}GB, $(free_gb)GB free)"
    ledger_set "$LKEY" stage1 running
    # RDM_CACHE=1: push each session RDM to the Hub as it is built, and pull
    # instead of rebuilding on any later run. Turning fMRI into RDMs costs hours
    # of CPU and hundreds of transient GB per dataset, and nothing about that
    # work is machine-specific -- paying it again on a fresh checkout is pure
    # waste. This was 0, so every one of these datasets would have had to be
    # reprocessed from BOLD by anyone who wanted the RDMs.
    #
    # BRAIN_RDM_ROOT is the UNSCOPED base -- prepare_brain_rdms.sh applies its
    # own "roi-<set>/" nesting internally when ROI_SET is set, landing RDMs at
    # exactly $RDM_ROOT (the scoped path everything else here reads). Passing
    # the already-scoped $RDM_ROOT here would double-nest it.
    # ROI_SET is forwarded as-is (empty string when unset -- prepare_brain_rdms.sh
    # treats that identically to the variable never being exported).
    DATASET="$DS" PHENOMENA="$TASKS" WITHIN_RUN_NORM=1 RDM_CACHE=1 KEEP_PATTERNS=0 \
      JOBS="$JOBS" DISK_FLOOR_GB="$DISK_FLOOR_GB" BRAIN_RDM_ROOT="$RDM_ROOT_BASE" \
      ROI_SET="${ROI_SET:-}" \
      bash prepare_brain_rdms.sh >>"logs/newds_stage1_${DS}${ROI_TAG}.log" 2>&1
    rc=$?
    N=$(find "$RDM_ROOT" -name 'session_rdm_*.npz' 2>/dev/null | wc -l)
    log "$DS ($ROI_LABEL): brain prep rc=$rc -- $N session RDMs, $(free_gb)GB free"
    ledger_set "$LKEY" stage1 "$([ "$N" -gt 0 ] && echo ok || echo failed)"
    ledger_set "$LKEY" n_rdms "$N"
    [ "$N" -eq 0 ] && { log "$DS ($ROI_LABEL): no RDMs -- SKIPPING the rest of this dataset"; continue; }
  fi

  # ---- 1b. stimulus texts --------------------------------------------------
  # run_devai_grid.py feeds `stimulus_texts` to the LM; a cell whose texts are
  # empty contributes no alignment row at all. RDMs built before the
  # pair-filename derivation landed in src/rsa/semantic_metadata.py (and any
  # pulled from the Hub cache) have an all-empty column, so annotate in place --
  # the RDM itself is untouched. Idempotent, and a no-op for ds003604.
  "$PY" scripts/backfill_rdm_texts.py --roots "$RDM_ROOT" \
      --characteristics-dir "data/brain/$DS/stimuli/Stimulus_Characteristics" \
      >>"logs/newds_stage1_${DS}${ROI_TAG}.log" 2>&1 \
    && log "$DS: stimulus texts present on every RDM" \
    || log "$DS: WARNING -- some RDMs still have no stimulus text (see log); those cells cannot produce alignment rows"

  # Sync every RDM this dataset has on disk, including ones built before
  # RDM_CACHE was turned on. `sync` skips what the Hub already holds.
  "$PY" scripts/rdm_cache_hf.py sync --root "$RDM_ROOT" --dataset "$DS" \
      >>"logs/newds_stage1_${DS}${ROI_TAG}.log" 2>&1 \
    && log "$DS: session RDMs synced to the Hub cache" \
    || log "$DS: WARNING -- RDM Hub sync failed (results unaffected; reprocessing will be needed elsewhere)"

  # ---- 2. ceilings ---------------------------------------------------------
  "$PY" scripts/collect_ceilings.py --rdm-root "$RDM_ROOT" \
      --out "$OUTDIR/ceilings_$DS.csv" >>"logs/newds_stage1_${DS}${ROI_TAG}.log" 2>&1 \
    && log "$DS: ceiling table -> $OUTDIR/ceilings_$DS.csv"

  # ---- 3. the gate ---------------------------------------------------------
  # Two prerequisites the gate silently ran without before 2026-08-29, which is
  # why ds002236's "0/6 significant" was really "1 of ~9 possible controls had
  # any data at all" -- see scripts/backfill_rdm_conditions.py's docstring and
  # PICKUP.md.
  #
  # (a) Real trial_types/semantic_categories for THESE datasets' already-built
  # RDMs. session_based_rsa.py now writes them correctly on a fresh build
  # (--dataset is threaded through as of this commit), but the RDMs already on
  # disk from before that fix still carry the "unknown" placeholder, which
  # made the `condition` control permanently degenerate. Idempotent -- a no-op
  # once an RDM has real labels, and correctly a no-op for ds003604 (which was
  # never affected -- see the script's docstring).
  "$PY" scripts/backfill_rdm_conditions.py --roots "$RDM_ROOT" --dataset "$DS" \
      >>"logs/newds_stage1_${DS}${ROI_TAG}.log" 2>&1 \
    && log "$DS: real condition labels present on every RDM" \
    || log "$DS: WARNING -- condition-label backfill failed for some cells (see log); the 'condition' control will stay degenerate for those"

  # (b) The stimulus audio/images themselves. positive_control.py's acoustic
  # and visual controls need the actual media bytes, not the git-annex
  # pointers a metadata-only checkout leaves behind -- nothing else in this
  # pipeline ever resolves them, so without this the acoustic control (the
  # STRONGEST low-level control for an auditory design) silently has no data
  # to build from, the same silent-omission failure mode as (a). Small: ~100s
  # of files, ~100-300MB per dataset (ds003604's own precedent: 352 files,
  # ~120MB) against the >>1GB BOLD floor this run already clears.
  "$PY" scripts/download_stimuli.py --dataset "$DS" \
      >>"logs/newds_stage1_${DS}${ROI_TAG}.log" 2>&1 \
    && log "$DS: stimulus media downloaded" \
    || log "$DS: WARNING -- stimulus media download failed (see log); the acoustic/visual control will have no data"

  log "$DS: positive control (the gate)"
  SESSIONS="$(sessions_for "$RDM_ROOT")"
  "$PY" scripts/positive_control.py --rdm-root "$RDM_ROOT" \
      --stimuli "data/brain/$DS/stimuli" --sessions "$SESSIONS" \
      --compare-root "data/processed/fmri/$DS" \
      --lm-cells "$OUTDIR/alignment_by_cell.csv" --dataset "$DS" \
      --out "$OUTDIR/control" >>"logs/newds_control_${DS}${ROI_TAG}.log" 2>&1
  "$PY" scripts/rdm_dimensionality.py --rdm-root "$RDM_ROOT" \
      --sessions "$SESSIONS" \
      --out "$OUTDIR/control" >>"logs/newds_control_${DS}${ROI_TAG}.log" 2>&1
  GATE=$("$PY" - "$OUTDIR/control/summary.json" <<'PYEOF'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p): print("unknown"); raise SystemExit
d = json.load(open(p))
print("pass" if d.get("n_significant_holm", 0) > 0 else "fail")
PYEOF
)
  log "$DS: GATE = $GATE"
  ledger_set "$LKEY" gate "$GATE"

  # ---- 4. model grid -------------------------------------------------------
  log "$DS: model grid, ${#FAMILIES[@]} families across GPUs $GPUS"
  ledger_set "$LKEY" stage_grid running
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
        >>"logs/newds_grid_${DS}${ROI_TAG}_gpu${GPU_ARR[$i]}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  NROWS=$(find "$GRID_PARENT/$DS" -name 'alignment_*.csv' 2>/dev/null | wc -l)
  log "$DS: grid done -- $NROWS alignment files"
  ledger_set "$LKEY" stage_grid "$([ "$NROWS" -gt 0 ] && echo ok || echo failed)"
  ledger_set "$LKEY" n_alignment_files "$NROWS"

  # ---- 5. summary ----------------------------------------------------------
  if [ "$NROWS" -gt 0 ]; then
    "$PY" scripts/corrected_sweep_summary.py \
        --grid-dir "$GRID_PARENT/$DS" \
        --ceilings "$OUTDIR/ceilings_$DS.csv" \
        --out "$OUTDIR" >>"logs/newds_summary_${DS}${ROI_TAG}.log" 2>&1 \
      && log "$DS: summary -> $OUTDIR"
  fi

  # ---- 6. publish to its own HF dataset ------------------------------------
  log "$DS ($ROI_LABEL): publishing to its own HuggingFace dataset"
  # --roi-set nests this level under its own "roi-<set>/" path in the SAME
  # per-dataset repo (not a separate repo per level) -- omitted entirely for
  # whole-brain, which publishes at the repo root exactly as before this
  # existed. See publish_dataset_results.py's --roi-set help for why the
  # same repo, not four.
  PUBLISH_ROI_ARGS=(); [ -n "${ROI_SET:-}" ] && PUBLISH_ROI_ARGS=(--roi-set "$ROI_SET")
  "$PY" scripts/publish_dataset_results.py --dataset "$DS" \
      --results "$OUTDIR" --gate "$GATE" "${PUBLISH_ROI_ARGS[@]}" \
      >>"logs/newds_publish_${DS}${ROI_TAG}.log" 2>&1 \
    && log "$DS ($ROI_LABEL): published" || log "$DS ($ROI_LABEL): PUBLISH FAILED (see logs/newds_publish_${DS}${ROI_TAG}.log)"

  ledger_set "$LKEY" ended "$(date -u +%FT%TZ)"
  ledger_set "$LKEY" status done
  log "=== $DS ($ROI_LABEL) DONE"

  git add -A
  if ! git diff --cached --quiet; then
    git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
        commit -q -m "Results for $DS ($ROI_LABEL): $NROWS alignment files, control gate: $GATE" \
      && for r in origin przemek; do git push -q "$r" main 2>/dev/null \
           && log "$DS ($ROI_LABEL): pushed $r" || log "$DS ($ROI_LABEL): push failed $r"; done
  fi
done

log "ALL DATASETS COMPLETE -- ledger: $LEDGER"
