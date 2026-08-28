#!/bin/bash
# Stage 0: build brain session RDMs for ds003604 without ever holding the dataset whole.
#
# WHY THIS EXISTS AND WHY IT STREAMS
#   slurm/run_devai_grid.sh does download -> preprocess -> session-RDM inline per task and
#   never reclaims anything. ds003604 is 3851 BOLD runs at ~154 MiB = ~578 GiB, and each run
#   then yields ~176 MiB of voxel patterns (~660 GiB more). This box has ~759 GB free with a
#   hard 350 GB floor -- the merge sweep on GPUs 4-7 aborts itself below 250 GB -- so only
#   ~409 GB is spendable. Doing it the inline way WOULD BREACH THE FLOOR and take the other
#   project down. Raw BOLD and per-run patterns are both intermediates; the products are the
#   per-session RDM .npz files, which are small.
#
#   So: parallel over subjects, each subject's BOLD deleted the moment it is preprocessed,
#   and the whole task's patterns deleted once its session RDMs exist. Peak disk is a few
#   subjects of BOLD plus one task of patterns, not 1.2 TB. Same inputs, same science.
#
#   Also: batch_download_bold.py does NOT create the dataset checkout, it requires one. On a
#   clean clone the grid therefore dies instantly with "No subjects found in
#   data/brain/ds003604" before touching a GPU. That bootstrap is done here.
#
# Cost, measured on this box: ~154 MiB/BOLD file, ~60 s/run to preprocess. Sequentially that
# is ~64 hours; at JOBS=24 it is a few hours. Set MAX_SUBJECTS to build a smaller cohort.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
. "$ROOT/env_brainalign.sh"

export DATASET="${DATASET:-ds003604}"
export DATA_DIR="${DATA_DIR:-data/brain/$DATASET}"
RDM_ROOT="${BRAIN_RDM_ROOT:-data/processed/fmri/$DATASET}"
PHENOMENA=(${PHENOMENA:-Sem Phon Gram Plaus})
export DISK_FLOOR_GB="${DISK_FLOOR_GB:-350}"
JOBS="${JOBS:-16}"
MAX_SUBJECTS="${MAX_SUBJECTS:-0}"      # 0 = all subjects
KEEP_PATTERNS="${KEEP_PATTERNS:-0}"    # 1 = keep per-run patterns (needs ~660 GiB; do not)
# WITHIN_RUN_NORM=1 z-scores each voxel within run before aggregating across runs.
# REQUIRED for every dataset whose run/stimulus structure is nested -- which,
# measured, is ds003604 (100% of stimuli in one run), ds001894 (98-99%), and
# ds006239's Read* tasks (96-97%). Without it the session RDM encodes scanner run:
# "different run" predicts dissimilarity at rho +0.49..+0.87, versus ~0 for every
# stimulus property. See configs/neuro_datasets.yaml and hf_results_staging/README.md.
WITHIN_RUN_NORM="${WITHIN_RUN_NORM:-0}"
WRN_FLAG=(); [ "$WITHIN_RUN_NORM" = "1" ] && WRN_FLAG=(--within-run-normalize)
# ROI_SET (comma-separated: language,auditory,motor -- see
# src/preprocessing/roi_atlas.py) restricts every subject's mask to the named
# region(s), via real per-subject registration to MNI152 (MASKING.md). Unset
# (default) = whole-brain mask only -- which still gets the mask_strategy=
# 'epi' fix (see fmri_preprocessing.py) regardless of this variable; that fix
# is not optional and applies either way.
# MASK_CACHE_DIR is exported here, ONCE, above the per-task loop below, so
# every task's call to brainprep_subject.sh shares the same cache and
# registration is computed once per subject-session rather than once per
# task. Do not set this per-task.
export ROI_SET="${ROI_SET:-}"
export MASK_CACHE_DIR="${MASK_CACHE_DIR:-$RDM_ROOT/_masks}"
# SAVE_NATIVE_MAPS=1 turns on real anatomical brain maps: each subject-session's
# whole-brain mask is saved during preprocessing (brainprep_subject.sh, only
# when ROI_SET is unset -- see the comment there) and run_brain_localization.py
# then reconstructs + warps each condition t-map to MNI space into MNI_MAPS_DIR.
# Off by default -- it adds a per-subject registration cost that most runs
# (which only need the scalar Gini/entropy table) don't need to pay.
export SAVE_NATIVE_MAPS="${SAVE_NATIVE_MAPS:-0}"
MNI_MAPS_DIR="${MNI_MAPS_DIR:-$RDM_ROOT/_mni_maps}"
LOCALIZE_MAP_ARGS=()
[ "$SAVE_NATIVE_MAPS" = "1" ] && LOCALIZE_MAP_ARGS=(--mask-cache-dir "$MASK_CACHE_DIR" --mni-maps-dir "$MNI_MAPS_DIR" --data-dir "$DATA_DIR")
PY="$ROOT/venv/bin/python"

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
log() { echo "[brainprep $(date -u +%FT%TZ)] $*"; }

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -d "$DATA_DIR"/sub-* 2>/dev/null | head -1)" ]; then
  log "bootstrapping dataset checkout -> $DATA_DIR (metadata only, ~98 MB)"
  mkdir -p "$(dirname "$DATA_DIR")"
  git clone --depth 1 --filter=blob:none --single-branch --branch main \
      "https://github.com/OpenNeuroDatasets/$DATASET.git" "$DATA_DIR" \
      || { log "FATAL: could not clone $DATASET metadata"; exit 1; }
fi

[ -f contrasts/Sem.csv ] || "$PY" scripts/build_contrasts.py --source github --out-dir contrasts

for T in "${PHENOMENA[@]}"; do
  # $T is a PHENOMENON (Sem/Phon/Orth/SemLocal), which is what the Python side
  # wants: FMRIPreprocessor resolves it through the registry's `phenomena:`
  # mapping (_resolve_real_tasks), and src/contrast_spec.py CONTRAST_SPECS is
  # keyed by it. But the BOLD files on disk are named after the real BIDS task
  # -- ds002236's Phon lives in task-AudRhyme_*, ds001894's spans six tasks --
  # so every *filename glob* in this script has to use the resolved labels, not
  # $T. Globbing "*task-Phon_*" against ds002236 matched nothing and the run
  # exited with "no runs found, skipping" and rc=0. Resolve once here; the
  # arrays below are what the finds use. For ds003604 each phenomenon maps to
  # its own name, so REAL_TASKS=($T) and every find is byte-identical to before.
  mapfile -t REAL_TASKS < <("$PY" - "$DATASET" "$T" <<'RESOLVE'
import sys
sys.path.insert(0, ".")
ds, phen = sys.argv[1], sys.argv[2]
try:
    from src.datasets import get_dataset
    tasks = get_dataset(ds).phenomena.get(phen) or [phen]
except Exception:
    tasks = [phen]
print("\n".join(tasks))
RESOLVE
)
  [ "${#REAL_TASKS[@]}" -eq 0 ] && REAL_TASKS=("$T")
  # find predicates: -name "*task-X_*bold.nii.gz" OR'd across REAL_TASKS.
  TASK_FIND=(); for _rt in "${REAL_TASKS[@]}"; do
    [ ${#TASK_FIND[@]} -gt 0 ] && TASK_FIND+=(-o)
    TASK_FIND+=(-name "*task-${_rt}_*bold.nii.gz")
  done
  TASK_FIND=(\( "${TASK_FIND[@]}" \))
  [ "${REAL_TASKS[*]}" != "$T" ] && log "$T -> BIDS task(s): ${REAL_TASKS[*]}"
  # ROI_SET changes what's IN a pattern file (which voxels), so language/
  # phonology/all runs must not share a path with each other or with the
  # whole-brain default -- otherwise the second grouping's patterns and RDMs
  # would silently overwrite the first's. Only add a subdirectory when
  # ROI_SET is actually set, so the default (unset) path is byte-identical to
  # before this existed -- an already-completed ds003604 run's cache/output
  # location does not move.
  ROI_SUBDIR=""
  [ -n "${ROI_SET:-}" ] && ROI_SUBDIR="roi-${ROI_SET//,/+}/"
  export OUT="$RDM_ROOT/${ROI_SUBDIR}$T"
  # Cross-sectional datasets (everything except ds003604) need patterns
  # relabeled by real per-subject age-group bin before an RDM is built --
  # see scripts/regroup_patterns_by_age.py and configs/age_groups.yaml for
  # why: a BIDS session there does not correspond to one developmental
  # timepoint the way ds003604's ses-5/7/9 do. ds003604 is untouched --
  # everything below behaves exactly as it did before this existed.
  NEEDS_AGE_REGROUP=1
  [ "$DATASET" = "ds003604" ] && NEEDS_AGE_REGROUP=0
  # NO task-level skip here. This used to be
  #     if ls "$OUT"/session_rdm_ses-*.npz; then continue; fi
  # which globs across sessions, so ONE finished session skipped the WHOLE task.
  # Observed 2026-08-25: Sem/ses-5 existed, and Sem/ses-7 and Sem/ses-9 were
  # silently dropped from the run -- two of twelve cells missing with a log line
  # that read like success. The per-SESSION skip further down is the correct one
  # and already handles resumption.
  mkdir -p "$OUT"
  if [ "$(free_gb)" -lt "$((DISK_FLOOR_GB + 40))" ]; then
    log "ABORT $T: $(free_gb)GB free, too close to the ${DISK_FLOOR_GB}GB floor"; exit 3
  fi

  # ------------------------------------------------------------------ per-SESSION batching
  # The first version of this loop preprocessed EVERY subject in the task, then computed RDMs,
  # then reclaimed. That makes peak disk the whole task's patterns. On 2026-08-18 the Sem task
  # alone reached 636 pattern files / 201 GB and drove free space onto the 350 GB floor, which
  # aborted brain prep and all three tiers with it (`aborted_disk_floor`), while the neighbouring
  # 96-hour merge sweep sat 101 GB above its own abort threshold. Nothing was lost -- no session
  # RDMs had been produced, so the 201 GB were pure orphaned intermediate -- but the run died.
  #
  # A session RDM aggregates across subjects WITHIN one session, so one session's patterns is the
  # smallest working set that can produce any output; there is no way to go finer without changing
  # the science. So batch by session: prep it, reduce it, reclaim it, move on. Peak disk becomes
  # one session instead of one task, and `session_based_rsa.py --sessions` already supports it.
  # Session labels are NOT always ses-<digits>. ds001894 uses ses-T1/ses-T2, and
  # ds002236 has no session entity at all (sub-XX/func/...). The old pattern was
  # `ses-[0-9]+` with a non-printing sed, so for those two datasets every path
  # passed through unchanged and SESSIONS filled up with full file paths --
  # silently producing garbage globs rather than an error. Match any session
  # token (letters/digits/+, for the age-group "11+" bin further down), print
  # only real matches (sed -n ... p, not a blanket substitution), and fall back
  # to a single pseudo-session labeled to match fmri_preprocessing.py's own
  # SESSIONLESS_LABEL ("ses-all") -- both sides of the pipeline must agree on
  # this name, since it ends up in the actual pattern filenames FMRIPreprocessor
  # writes, not just here.
  mapfile -t SESSIONS < <(find "$DATA_DIR" "${TASK_FIND[@]}" \
      | sed -nE 's|.*_(ses-[A-Za-z0-9+]+)_.*|\1|p' | sort -u)
  if [ "${#SESSIONS[@]}" -eq 0 ]; then
    if [ -n "$(find "$DATA_DIR" "${TASK_FIND[@]}" -print -quit)" ]; then
      SESSIONS=(ses-all)
      log "$T: no ses-* entity found in filenames -- treating as one session (ses-all)"
    else
      log "$T: no runs found, skipping"; continue
    fi
  fi

  # ONLY_SESSIONS / SKIP_SESSIONS -- run a subset of sessions.
  # A session RDM aggregates across every subject in that session, so one session's patterns is an
  # atom: it cannot be split without changing the science. That makes the LARGE sessions the whole
  # disk problem. Measured on 2026-08-18: ses-5 (91 subjects) peaked around 60 GB and completed
  # cleanly, while ses-7 (217 subjects) had already reached 529 pattern files / 166 GB when it hit
  # the floor, and needs roughly 250 GB to finish. On a box sharing one filesystem with a 20-hour
  # merge sweep that aborts below 250 GB free, ses-7 is not affordable and ses-5/ses-9 are. This
  # lets the affordable sessions run now and ses-7 be picked up later with the disk to itself,
  # rather than the all-or-nothing choice that has now cost two aborted runs.
  if [ -n "${ONLY_SESSIONS:-}" ]; then
    KEEP=(); for S in "${SESSIONS[@]}"; do
      case " ${ONLY_SESSIONS//,/ } " in *" $S "*) KEEP+=("$S");; esac
    done
    SESSIONS=("${KEEP[@]}")
    [ "${#SESSIONS[@]}" -eq 0 ] && { log "$T: no sessions match ONLY_SESSIONS=$ONLY_SESSIONS, skipping"; continue; }
  fi
  if [ -n "${SKIP_SESSIONS:-}" ]; then
    KEEP=(); for S in "${SESSIONS[@]}"; do
      case " ${SKIP_SESSIONS//,/ } " in *" $S "*) ;; *) KEEP+=("$S");; esac
    done
    SESSIONS=("${KEEP[@]}")
  fi
  log "$T: ${#SESSIONS[@]} session(s) to run: ${SESSIONS[*]}"

  for S in "${SESSIONS[@]}"; do
    if [ "$NEEDS_AGE_REGROUP" = "1" ]; then
      # No per-on-disk-session skip/cache-pull here (unlike the ds003604
      # branch below): which on-disk sessions feed which age-group bin is an
      # M:N relationship (ds001894's ses-T1 alone spans 4 bins), so "this
      # session's RDM already exists" isn't a coherent question to ask per S.
      # Resumption still works, just at a coarser grain: the age-group block
      # after this loop checks per-BIN whether an RDM (or a Hub-cached one)
      # already exists before doing any RSA work, so a re-run after an
      # interruption redoes at most this task's preprocessing (bounded, cheap
      # relative to the GPU-hours a fresh sweep costs) rather than any
      # already-finished RDM computation.
      :
    else
      if ls "$OUT"/session_rdm_${S}.npz >/dev/null 2>&1; then
        log "$T/$S: RDM already present -- skip"; continue
      fi
      # Try the Hub cache before doing hours of CPU. Session RDMs are deterministic given the
      # dataset, so a run that has already produced one anywhere never needs to produce it again;
      # `rdm_cache_hf.py pull` exits 0 only if it actually placed the file.
      # Cache paths are namespaced by dataset AND correction variant, so a pull can
      # only ever return an RDM built the same way this run is building them. Before
      # that namespacing existed the cache was a hazard here -- it held the original
      # confounded RDMs under bare "{task}/session_rdm_{session}.npz" and would have
      # served one into a corrected run. It is now safe, and worth using: a session
      # RDM is deterministic given the dataset, so preprocessing is paid for once
      # ever rather than once per run.
      CACHE_VARIANT=raw
      [ "$WITHIN_RUN_NORM" = "1" ] && CACHE_VARIANT=within-run-normalised
      if [ "${RDM_CACHE:-1}" = "1" ] && "$PY" "$ROOT/scripts/rdm_cache_hf.py" \
           pull --task "$T" --session "$S" --dir "$OUT" \
           --dataset "$DATASET" --variant "$CACHE_VARIANT" 2>&1 | sed "s/^/  /"; then
        if ls "$OUT"/session_rdm_${S}.npz >/dev/null 2>&1; then
          log "$T/$S: pulled from Hub cache -- preprocessing skipped"; continue
        fi
      fi
    fi
    # Re-check the floor per session, not just per task: a session batch is the unit that can now
    # actually move the needle, so it is the unit that must be gated.
    if [ "$(free_gb)" -lt "$((DISK_FLOOR_GB + 40))" ]; then
      log "ABORT $T/$S: $(free_gb)GB free, too close to the ${DISK_FLOOR_GB}GB floor"; exit 3
    fi

    # "ses-all" is the pseudo-session used when the dataset has no session
    # entity at all (ds002236) -- it must match fmri_preprocessing.py's
    # SESSIONLESS_LABEL, since that is the label baked into the pattern
    # filenames. This test read "ses-none" until 2026-08-28, a label nothing in
    # the pipeline produces any more, so a session-less dataset fell to the
    # else-branch and globbed "*ses-all_task-..." -- which matches nothing,
    # because those filenames carry no session token. Zero subjects, skipped,
    # rc=0.
    if [ "$S" = "ses-all" ]; then
      mapfile -t SUBS < <(find "$DATA_DIR" "${TASK_FIND[@]}" \
          | sed -E 's|.*/(sub-[^/]+)/.*|\1|' | sort -u)
    else
      SES_TASK_FIND=(); for _rt in "${REAL_TASKS[@]}"; do
        [ ${#SES_TASK_FIND[@]} -gt 0 ] && SES_TASK_FIND+=(-o)
        SES_TASK_FIND+=(-name "*${S}_task-${_rt}_*bold.nii.gz")
      done
      mapfile -t SUBS < <(find "$DATA_DIR" \( "${SES_TASK_FIND[@]}" \) \
          | sed -E 's|.*/(sub-[^/]+)/.*|\1|' | sort -u)
    fi
    [ "$MAX_SUBJECTS" -gt 0 ] && SUBS=("${SUBS[@]:0:$MAX_SUBJECTS}")
    [ "${#SUBS[@]}" -eq 0 ] && { log "$T/$S: no subjects, skipping"; continue; }
    log "$T/$S: ${#SUBS[@]} subjects, JOBS=$JOBS, $(free_gb)GB free"

    printf '%s\n' "${SUBS[@]}" | xargs -P "$JOBS" -I{} \
        bash "$ROOT/scripts/brainprep_subject.sh" {} "$T" 2>&1 | grep -vE "^\[.*\] ok$"

    # Put the git-annex SYMLINKS back. brainprep_subject.sh drops each subject's raw BOLD once it
    # is preprocessed, which is what bounds disk -- but the downloaded blob REPLACES the symlink,
    # so deleting it removes the path entirely rather than reverting it to a pointer. The run then
    # silently loses the ability to re-derive that subject: after the 2026-08-18 abort, Sem had
    # dropped from 255 findable subjects to 34, and with the orphaned patterns reclaimed those
    # runs were simply gone from the experiment. `git checkout` restores every dropped pointer
    # from the metadata checkout for ~0 bytes. Done once per batch, not per subject, because 16
    # parallel workers would contend on .git/index.lock.
    if git -C "$DATA_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git -C "$DATA_DIR" checkout -- . 2>/dev/null \
        && log "$T/$S: restored dropped BOLD symlinks ($(find "$DATA_DIR" -name '*bold.nii.gz' | wc -l) runs referencable)"
    fi

    NP=$(ls "$OUT"/*${S}*_patterns.npz 2>/dev/null | wc -l)
    [ "$NP" -eq 0 ] && { log "$T/$S: no patterns produced, skipping"; continue; }

    if [ "$NEEDS_AGE_REGROUP" = "1" ]; then
      # Patterns stay on disk -- the age-group block below needs every
      # on-disk session's patterns present at once to relabel by real age.
      # This is the one place this dataset class departs from the per-
      # session reclaim discipline the comment at the top of this file
      # describes; see MASKING.md-adjacent notes / the disk-floor checks
      # both here and in the age-group block for how that's bounded.
      log "$T/$S: $NP pattern files -- deferring RDM build until every on-disk "
      log "  session for this task is preprocessed (age-group regrouping)"
      continue
    fi

    log "$T/$S: $NP pattern files, $(free_gb)GB free -- computing session RDM"
    # --task is REQUIRED. It used to be omitted and session_based_rsa.py defaulted it to
    # "Sem", so Phon/Gram/Plaus stimuli were matched against Sem's stimulus list, matched
    # nothing, and no session RDM was ever produced for three of the four tasks.
    "$PY" src/rsa/session_based_rsa.py --pattern-dir "$OUT" --output-dir "$OUT" \
        --task "$T" --sessions "$S" --metric correlation --aggregation hyperalignment \
        "${WRN_FLAG[@]}" \
        --characteristics-dir "$DATA_DIR/stimuli/Stimulus_Characteristics" \
        || { log "$T/$S: RSA failed"; continue; }

    if ls "$OUT"/session_rdm_${S}.npz >/dev/null 2>&1; then
      # MUST run before reclaim, not after: brain_specialization() needs the
      # actual pattern files, which the very next block deletes. Calling this
      # once at the end of the whole script -- what this used to do -- found
      # nothing for ANY task/session, ever, because every one of them had
      # already been reclaimed by the time it ran. See the module docstring
      # in scripts/run_brain_localization.py for the full story.
      "$PY" scripts/run_brain_localization.py --dataset "$DATASET" \
          --pattern-dir "$OUT" --sessions "$S" --append \
          --output-dir "$RDM_ROOT/localization" \
          --characteristics-dir "$DATA_DIR/stimuli/Stimulus_Characteristics" \
          "${LOCALIZE_MAP_ARGS[@]}" \
          2>&1 | sed "s/^/  [localize] /" \
          || log "$T/$S: brain localization failed (non-fatal, continuing)"
      if [ "$KEEP_PATTERNS" != "1" ]; then
        log "$T/$S: RDM built -- reclaiming $NP pattern files"
        find "$OUT" -name "*${S}*_patterns.npz" -type f -delete
      fi
      # Push regardless of the variant: the pushed path is derived from the
      # file's own within_run_normalized flag, so it is self-labelling and cannot
      # be filed under the wrong variant.
      [ "${RDM_CACHE:-1}" = "1" ] && "$PY" "$ROOT/scripts/rdm_cache_hf.py" \
          push --task "$T" --session "$S" --dir "$OUT" \
          --dataset "$DATASET" 2>&1 | sed "s/^/  /"
      log "$T/$S: done, $(free_gb)GB free"
    else
      log "$T/$S: NO session RDM produced -- keeping patterns for diagnosis"
    fi
  done

  # ---------------------------------------------------------- age-group RDMs
  # For cross-sectional datasets: every on-disk session for this task has now
  # been preprocessed (patterns for ALL of them still on disk, deliberately --
  # see the "deferring" log line above). Relabel by real per-subject age, then
  # build one RDM per age-group bin that actually has patterns, using the
  # SAME session_based_rsa.py call the ds003604 path uses -- the only thing
  # that differs is which session label is passed.
  if [ "$NEEDS_AGE_REGROUP" = "1" ]; then
    if [ "$(free_gb)" -lt "$((DISK_FLOOR_GB + 40))" ]; then
      log "ABORT $T: $(free_gb)GB free, too close to the ${DISK_FLOOR_GB}GB floor before age regrouping"
      exit 3
    fi
    log "$T: regrouping patterns by real per-subject age (configs/age_groups.yaml)"
    "$PY" scripts/regroup_patterns_by_age.py --dataset "$DATASET" --pattern-dir "$OUT" --mode copy \
      || log "$T: age regrouping failed (see above) -- no age-group RDMs will be built"

    for BIN in 5 7 9 11 "11+"; do
      AS="ses-$BIN"
      NP=$(ls "$OUT"/*"_${AS}_"*_patterns.npz 2>/dev/null | wc -l)
      [ "$NP" -eq 0 ] && continue
      if ls "$OUT"/session_rdm_${AS}.npz >/dev/null 2>&1; then
        log "$T/$AS: RDM already present -- skip"; continue
      fi
      CACHE_VARIANT=raw
      [ "$WITHIN_RUN_NORM" = "1" ] && CACHE_VARIANT=within-run-normalised
      if [ "${RDM_CACHE:-1}" = "1" ] && "$PY" "$ROOT/scripts/rdm_cache_hf.py" \
           pull --task "$T" --session "$AS" --dir "$OUT" \
           --dataset "$DATASET" --variant "$CACHE_VARIANT" 2>&1 | sed "s/^/  /"; then
        if ls "$OUT"/session_rdm_${AS}.npz >/dev/null 2>&1; then
          log "$T/$AS: pulled from Hub cache -- skipping"; continue
        fi
      fi
      if [ "$(free_gb)" -lt "$((DISK_FLOOR_GB + 40))" ]; then
        log "ABORT $T/$AS: $(free_gb)GB free, too close to the ${DISK_FLOOR_GB}GB floor"; exit 3
      fi
      log "$T/$AS: $NP pattern files (age group), $(free_gb)GB free -- computing session RDM"
      "$PY" src/rsa/session_based_rsa.py --pattern-dir "$OUT" --output-dir "$OUT" \
          --task "$T" --sessions "$AS" --metric correlation --aggregation hyperalignment \
          "${WRN_FLAG[@]}" \
          --characteristics-dir "$DATA_DIR/stimuli/Stimulus_Characteristics" \
          || { log "$T/$AS: RSA failed"; continue; }
      if ls "$OUT"/session_rdm_${AS}.npz >/dev/null 2>&1; then
        # Before reclaim, same reasoning as the ds003604 branch above -- this
        # is the ONLY point in this branch where the age-group-labeled
        # patterns for $AS are guaranteed to still be on disk (reclaim below
        # happens once, after every bin in this loop is done).
        "$PY" scripts/run_brain_localization.py --dataset "$DATASET" \
            --pattern-dir "$OUT" --sessions "$AS" --append \
            --output-dir "$RDM_ROOT/localization" \
            "${LOCALIZE_MAP_ARGS[@]}" \
            2>&1 | sed "s/^/  [localize] /" \
            || log "$T/$AS: brain localization failed (non-fatal, continuing)"
        [ "${RDM_CACHE:-1}" = "1" ] && "$PY" "$ROOT/scripts/rdm_cache_hf.py" \
            push --task "$T" --session "$AS" --dir "$OUT" \
            --dataset "$DATASET" 2>&1 | sed "s/^/  /"
        log "$T/$AS: done, $(free_gb)GB free"
      else
        log "$T/$AS: NO session RDM produced -- keeping patterns for diagnosis"
      fi
    done

    if [ "$KEEP_PATTERNS" != "1" ]; then
      NP_ALL=$(ls "$OUT"/*_patterns.npz 2>/dev/null | wc -l)
      log "$T: reclaiming all $NP_ALL pattern files (on-disk-session + age-group copies)"
      find "$OUT" -name "*_patterns.npz" -type f -delete
    fi
  fi
done

# Finalize the brain localizer: collapse the table accumulated by every
# --append call above (across every task/session/bin, whichever branch built
# it) into onsets + a figure. Does NOT re-scan any pattern directory -- by
# this point in the script every one of them has been reclaimed, which is
# exactly the bug this two-phase append/finalize design exists to route
# around (see scripts/run_brain_localization.py's module docstring).
LOC_OUT="$RDM_ROOT/localization"
if [ -f "$LOC_OUT/brain_localization_by_session.csv" ]; then
  log "brain isolation localizer: finalizing accumulated table -> $LOC_OUT"
  "$PY" scripts/run_brain_localization.py --finalize-only --output-dir "$LOC_OUT" \
      || log "brain localization finalize failed (non-fatal)"
else
  log "brain isolation localizer: no accumulated table at $LOC_OUT -- nothing to finalize "
  log "  (every --append call above must have failed or found no patterns; check the "
  log "  [localize] lines further up this log)"
fi

N=$(find "$RDM_ROOT" -name "session_rdm_ses-*.npz" | wc -l)
log "brain prep complete: $N session RDM files, $(free_gb)GB free"
[ "$N" -gt 0 ] || exit 4
