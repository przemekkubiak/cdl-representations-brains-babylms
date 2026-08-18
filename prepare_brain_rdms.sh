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

DATASET="${DATASET:-ds003604}"
export DATA_DIR="${DATA_DIR:-data/brain/$DATASET}"
RDM_ROOT="${BRAIN_RDM_ROOT:-data/processed/fmri/$DATASET}"
PHENOMENA=(${PHENOMENA:-Sem Phon Gram Plaus})
export DISK_FLOOR_GB="${DISK_FLOOR_GB:-350}"
JOBS="${JOBS:-16}"
MAX_SUBJECTS="${MAX_SUBJECTS:-0}"      # 0 = all subjects
KEEP_PATTERNS="${KEEP_PATTERNS:-0}"    # 1 = keep per-run patterns (needs ~660 GiB; do not)
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
  if ls "$OUT"/session_rdm_ses-*.npz >/dev/null 2>&1; then
    log "$T: session RDMs already present -- skip"; continue
  fi
  mkdir -p "$OUT"
  if [ "$(free_gb)" -lt "$((DISK_FLOOR_GB + 40))" ]; then
    log "ABORT $T: $(free_gb)GB free, too close to the ${DISK_FLOOR_GB}GB floor"; exit 3
  fi

  # subjects that actually have this task, straight from the checkout's symlinks
  mapfile -t SUBS < <(find "$DATA_DIR" -name "*task-${T}_*bold.nii.gz" \
      | sed -E 's|.*/(sub-[^/]+)/.*|\1|' | sort -u)
  [ "$MAX_SUBJECTS" -gt 0 ] && SUBS=("${SUBS[@]:0:$MAX_SUBJECTS}")
  log "$T: ${#SUBS[@]} subjects, JOBS=$JOBS, $(free_gb)GB free"

  printf '%s\n' "${SUBS[@]}" | xargs -P "$JOBS" -I{} \
      bash "$ROOT/scripts/brainprep_subject.sh" {} "$T" 2>&1 | grep -vE "^\[.*\] ok$"

  NP=$(ls "$OUT"/*_patterns.npz 2>/dev/null | wc -l)
  log "$T: $NP pattern files, $(free_gb)GB free -- computing session RDMs"
  [ "$NP" -eq 0 ] && { log "$T: no patterns produced, skipping RSA"; continue; }

  "$PY" src/rsa/session_based_rsa.py --pattern-dir "$OUT" --output-dir "$OUT" \
      --metric correlation --aggregation hyperalignment || { log "$T: RSA failed"; continue; }

  if ls "$OUT"/session_rdm_ses-*.npz >/dev/null 2>&1; then
    if [ "$KEEP_PATTERNS" != "1" ]; then
      log "$T: RDMs built -- reclaiming $NP pattern files"
      find "$OUT" -name "*_patterns.npz" -type f -delete
    fi
    log "$T: done, $(free_gb)GB free"
  else
    log "$T: NO session RDMs produced -- keeping patterns for diagnosis"
  fi
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
