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
    --subjects "$SUB" --workers 2 >/dev/null 2>&1 || { echo "[$SUB/$T] download failed"; drop_bold; exit 1; }

if ! ls "$DATA_DIR/$SUB"/*/func/*task-${T}_*bold.nii.gz >/dev/null 2>&1; then
  echo "[$SUB/$T] no BOLD for this task"; exit 0
fi

"$PYBIN" src/preprocessing/batch_preprocessing.py --data-dir "$DATA_DIR" \
    --output-dir "$OUT" --task "$T" --subjects "$SUB" \
    --smoothing-fwhm "${SMOOTHING:-6.0}" --high-pass "${HIGHPASS:-0.01}" >/dev/null 2>&1
rc=$?
drop_bold                      # raw BOLD is an intermediate; RDMs are the product
[ $rc -eq 0 ] || { echo "[$SUB/$T] preprocess failed"; exit 1; }
echo "[$SUB/$T] ok"
