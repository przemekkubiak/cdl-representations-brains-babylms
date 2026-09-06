#!/usr/bin/env bash
# run_brain_par.sh -- one of the three new datasets, all waves, all ROI levels.
# Launched once per dataset so the three run CONCURRENTLY.
#
# WHAT CHANGED AND WHY. The first version ran one (dataset, ROI) stage at a time.
# On a box with 128 cores and three A100s that leaves nearly everything idle: the
# GLM/registration stage is CPU-parallel across subjects, and the three datasets
# are completely independent -- different images, different masks, different
# output trees. So they now run at the same time, one tmux session each, with a
# GPU pinned per dataset for the model-grid stage and the CPU pool split three
# ways.
#
# Cohorts stay MATCHED across the three and NESTED across waves, so a later wave
# extends an earlier one rather than redoing it: prepare_brain_rdms.sh takes the
# first N of a sorted subject list (line 303). Waves 6, 12, 25, 50, 89 -- the
# first deliberately tiny so there is a complete, publishable grid early, and 89
# is the cap, set by ds006239's cohort.
#
# ds003604 is not here. It already has whole-brain plus three ROI levels at its
# full 322-subject cohort; only roi-language is missing, and that is a depth item
# on one dataset rather than part of this comparison.
#
# Each wave publishes to the Hub before the next begins, so an interruption
# always leaves a complete published tier behind.
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
DS="${1:?usage: run_brain_par.sh <dataset> <gpu>}"
GPU="${2:?usage: run_brain_par.sh <dataset> <gpu>}"
WAVES="${WAVES:-6 12 25 50 89}"
ROIS="${ROIS:-phonology auditory motor language}"
JOBS="${JOBS:-36}"
PY="$ROOT/venv/bin/python"
L="$ROOT/logs/par_${DS}.log"
log() { echo "[$DS $(date -u +%FT%TZ)] $*" | tee -a "$L"; }

log "start; GPU $GPU, JOBS $JOBS, waves: $WAVES"
for N in $WAVES; do
  log "===== wave N=$N ====="
  for roi in "" $ROIS; do
    tag="${roi:-wholebrain}"
    t0=$SECONDS
    MAX_SUBJECTS="$N" ROI_SET="$roi" DATASETS="$DS" RDM_CACHE=1 \
      JOBS="$JOBS" GPUS="$GPU" \
      timeout 21600 bash run_new_datasets.sh >>"logs/par_${DS}_N${N}_${tag}.log" 2>&1
    rc=$?
    # "grid done -- 0 alignment files" means every model load failed and the
    # stage still exited 0. Treat it as a failure so it is never published or
    # counted as coverage.
    if grep -q "grid done -- 0 alignment files" "logs/par_${DS}_N${N}_${tag}.log" 2>/dev/null; then
      log "N=$N $tag FAILED: grid produced 0 alignment files (check HF_HOME/token)"
    else
      log "N=$N $tag rc=$rc $(( (SECONDS - t0) / 60 ))m"
    fi
  done
  log "wave N=$N complete; publishing"
  bash push_brain_to_hf.sh >>"logs/par_${DS}_N${N}_hf.log" 2>&1 \
    && log "N=$N published" || log "N=$N publish had failures"
done
log "ALL WAVES COMPLETE"
