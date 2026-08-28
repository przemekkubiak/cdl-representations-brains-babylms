#!/bin/bash
# Download -> preprocess -> drop raw BOLD, for ONE subject and ONE task.
# Invoked in parallel by prepare_brain_rdms.sh. Keeping the delete inside the
# per-subject unit is what bounds peak disk: raw BOLD for a subject exists only
# between its download and its preprocessing, so the dataset never lands whole.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
SUB="$1"; T="$2"
DATA_DIR="${DATA_DIR:-data/brain/ds003604}"
OUT="${OUT:-data/processed/fmri/ds003604/$T}"
FLOOR="${DISK_FLOOR_GB:-350}"
PYBIN="$ROOT/venv/bin/python"

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
drop_bold() { find "$DATA_DIR/$SUB" -name "*task-${T}_*bold.nii.gz" -type f -delete 2>/dev/null; }

# already preprocessed?
if ls "$OUT/${SUB}_"*_patterns.npz >/dev/null 2>&1; then drop_bold; exit 0; fi

if [ "$(free_gb)" -lt "$FLOOR" ]; then echo "[$SUB/$T] SKIP: below disk floor"; exit 9; fi

"$PYBIN" scripts/batch_download_bold.py --data-dir "$DATA_DIR" --task "$T" \
    --dataset "${DATASET:-ds003604}" \
    --subjects "$SUB" --workers 2 >/dev/null 2>&1 || { echo "[$SUB/$T] download failed"; drop_bold; exit 1; }

# Both BIDS layouts: sub-XX/ses-YY/func/ (ds003604, ds001894, ds006239) and
# sub-XX/func/ for datasets with no session entity (ds002236). The glob used to
# require the session level, so every session-less dataset reported "no BOLD for
# this task" for every subject and exited 0 -- a silent total loss.
if ! find "$DATA_DIR/$SUB" -name "*task-${T}_*bold.nii.gz" -print -quit \
     | grep -q . ; then
  echo "[$SUB/$T] no BOLD for this task"; exit 0
fi

# ROI_SET (e.g. "auditory,motor") triggers per-subject registration to MNI152
# instead of a plain whole-brain mask -- see src/preprocessing/roi_atlas.py
# and MASKING.md. MASK_CACHE_DIR MUST be the same across every task for this
# dataset (set once in prepare_brain_rdms.sh, above the per-task loop) so
# registration is computed once per subject-session and reused, not redone
# for each of Sem/Phon/Gram/Plaus.
ROI_ARGS=()
if [ -n "${ROI_SET:-}" ]; then
  # OUT is "$RDM_ROOT/$T" (task-scoped); its parent is the dataset-level
  # RDM_ROOT, which is the right default cache location -- shared across
  # every task's invocation of this script for the same dataset.
  ROI_ARGS=(--roi-set "$ROI_SET" --mask-cache-dir "${MASK_CACHE_DIR:-$(dirname "$OUT")/_masks}")
elif [ "${SAVE_NATIVE_MAPS:-0}" = "1" ]; then
  # SAVE_NATIVE_MAPS is the whole-brain counterpart of ROI_SET above: it also
  # needs --mask-cache-dir (same shared cache, same reasoning), but is
  # mutually exclusive with --roi-set at the FMRIPreprocessor level (an
  # ROI-intersected pattern can't be unmasked back against the whole-brain
  # mask this saves -- see the ValueError in fmri_preprocessing.py), so it
  # only applies in the branch where ROI_SET is unset.
  ROI_ARGS=(--mask-cache-dir "${MASK_CACHE_DIR:-$(dirname "$OUT")/_masks}" --save-native-maps)
fi

"$PYBIN" src/preprocessing/batch_preprocessing.py --data-dir "$DATA_DIR" \
    --output-dir "$OUT" --task "$T" --dataset "${DATASET:-ds003604}" --subjects "$SUB" \
    --smoothing-fwhm "${SMOOTHING:-6.0}" --high-pass "${HIGHPASS:-0.01}" \
    "${ROI_ARGS[@]}" >/dev/null 2>&1
rc=$?
drop_bold                      # raw BOLD is an intermediate; RDMs are the product
[ $rc -eq 0 ] || { echo "[$SUB/$T] preprocess failed"; exit 1; }
echo "[$SUB/$T] ok"
