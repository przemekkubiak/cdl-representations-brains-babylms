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
  export OUT="$RDM_ROOT/$T"
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
  # token, print only real matches, and fall back to a single pseudo-session.
  mapfile -t SESSIONS < <(find "$DATA_DIR" -name "*task-${T}_*bold.nii.gz" \
      | sed -nE 's|.*_(ses-[A-Za-z0-9]+)_.*|\1|p' | sort -u)
  if [ "${#SESSIONS[@]}" -eq 0 ]; then
    if [ -n "$(find "$DATA_DIR" -name "*task-${T}_*bold.nii.gz" -print -quit)" ]; then
      SESSIONS=(ses-none)
      log "$T: no session entity in filenames -- single pseudo-session ses-none"
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
    # Re-check the floor per session, not just per task: a session batch is the unit that can now
    # actually move the needle, so it is the unit that must be gated.
    if [ "$(free_gb)" -lt "$((DISK_FLOOR_GB + 40))" ]; then
      log "ABORT $T/$S: $(free_gb)GB free, too close to the ${DISK_FLOOR_GB}GB floor"; exit 3
    fi

    if [ "$S" = "ses-none" ]; then
      mapfile -t SUBS < <(find "$DATA_DIR" -name "*task-${T}_*bold.nii.gz" \
          | sed -E 's|.*/(sub-[^/]+)/.*|\1|' | sort -u)
    else
      mapfile -t SUBS < <(find "$DATA_DIR" -name "*${S}_task-${T}_*bold.nii.gz" \
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
    log "$T/$S: $NP pattern files, $(free_gb)GB free -- computing session RDM"
    [ "$NP" -eq 0 ] && { log "$T/$S: no patterns produced, skipping RSA"; continue; }

    # --task is REQUIRED. It used to be omitted and session_based_rsa.py defaulted it to
    # "Sem", so Phon/Gram/Plaus stimuli were matched against Sem's stimulus list, matched
    # nothing, and no session RDM was ever produced for three of the four tasks.
    "$PY" src/rsa/session_based_rsa.py --pattern-dir "$OUT" --output-dir "$OUT" \
        --task "$T" --sessions "$S" --metric correlation --aggregation hyperalignment \
        "${WRN_FLAG[@]}" \
        --characteristics-dir "$DATA_DIR/stimuli/Stimulus_Characteristics" \
        || { log "$T/$S: RSA failed"; continue; }

    if ls "$OUT"/session_rdm_${S}.npz >/dev/null 2>&1; then
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
done

SPEC="$RDM_ROOT/localization/brain_specialization.csv"
if [ ! -f "$SPEC" ]; then
  log "brain isolation localizer"
  "$PY" scripts/run_brain_localization.py --pattern-dir "$RDM_ROOT" \
      --characteristics-dir "$DATA_DIR/stimuli/Stimulus_Characteristics" \
      || log "brain localization skipped (patterns/characteristics missing)"
fi

N=$(find "$RDM_ROOT" -name "session_rdm_ses-*.npz" | wc -l)
log "brain prep complete: $N session RDM files, $(free_gb)GB free"
[ "$N" -gt 0 ] || exit 4
