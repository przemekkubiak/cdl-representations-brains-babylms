#!/bin/bash
# Parallel stage-0 worker for ONE task.
#
# WHY A SEPARATE DATASET CHECKOUT
#   prepare_brain_rdms.sh runs `git -C "$DATA_DIR" checkout -- .` after each session batch to
#   restore the git-annex BOLD symlinks it deleted. If two instances shared one checkout, that
#   checkout would revert the OTHER instance's freshly downloaded, not-yet-preprocessed BOLD
#   back to a pointer -- the "vanished BOLD" bug that silently cut Sem from 255 subjects to 34.
#   So each parallel worker gets its own metadata-only clone (~98 MB) and they share nothing
#   but the output directory tree, where they touch different tasks.
#
# Usage:  bash scripts/par_stage0_task.sh <Task> <DataDirSuffix> [JOBS] [MAX_SUBJECTS]
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK="${1:?usage: par_stage0_task.sh <Task> <suffix> [jobs] [max_subjects]}"
SUFFIX="${2:?}"
export JOBS="${3:-12}"
export MAX_SUBJECTS="${4:-40}"
export DATA_DIR="data/brain/ds003604_${SUFFIX}"
export PHENOMENA="$TASK"
export DISK_FLOOR_GB=350
export KEEP_PATTERNS="${KEEP_PATTERNS:-0}"

if [ ! -d "$DATA_DIR" ]; then
  git clone --quiet --depth 1 --filter=blob:none --single-branch --branch main \
      https://github.com/OpenNeuroDatasets/ds003604.git "$DATA_DIR" || exit 1
fi

exec bash prepare_brain_rdms.sh
