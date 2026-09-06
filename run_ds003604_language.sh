#!/usr/bin/env bash
# The one gap in ds003604: roi-language at its full 322-subject cohort.
# Kept separate from run_brain_waves.sh because it is a DEPTH item on a dataset
# that is already complete at three ROI levels, not part of the matched
# comparison across the three new datasets. Budgeted at 8h on the evidence of
# the auditory wave, which took 4h45m over the same cohort with masks already
# built.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"

# REQUIRED. env_brainalign.sh defaults HF_HOME to /root/hf_cache_brainalign,
# which is not writable here, and /home/sas245 is not readable either -- see
# ENVIRONMENT.md. Leaving HOME unset makes every model load fail with a
# PermissionError on /root, and the grid then reports "0 alignment files" and
# exits 0, which is indistinguishable from a real empty result. That has already
# been committed as a finding once (env_brainalign.sh line 17). Set both.
export HOME=/local/scratch/sas245
export HF_HOME="$HOME/hf_cache"
export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
export TOKENIZERS_PARALLELISM=false
log() { echo "[ds3604lang $(date -u +%FT%TZ)] $*" | tee -a logs/ds003604_language.log; }
log "START roi-language, full cohort"
ROI_SET=language DATASETS=ds003604 RDM_CACHE=1 \
  timeout 28800 bash run_new_datasets.sh >>logs/ds003604_language_run.log 2>&1
log "finished rc=$? -- reconciling coverage and Hub"
bash "$ROOT/../refresh_results.sh" >/dev/null 2>&1 || true
"$ROOT/venv/bin/python" scripts/coverage_matrix.py | tee -a logs/ds003604_language.log
bash push_brain_to_hf.sh >>logs/ds003604_language_hf.log 2>&1 && log "published" || log "PUBLISH had failures"
log "DONE"
