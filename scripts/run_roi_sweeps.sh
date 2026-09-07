#!/usr/bin/env bash
# Sweep and publish the ROI-masked RDM rebuilds.
#
# The published datasets were built from WHOLE-BRAIN, unmasked RDMs -- anatomical
# masking was never applied on the cluster that produced them. That is the single
# largest open question about every number in them: whether the near-zero alignment
# is a property of language models or of averaging over 917k voxels of which most
# are not language-responsive.
#
# `prepare_brain_rdms.sh` (run off-cluster) writes the masked trees beside the
# unmasked one:
#     data/ds003604-session-rdms/<accession>/roi-language/<Task>/session_rdm_<ses>.npz
#     .../roi-phonology/...      auditory + motor
#     .../roi-all/...            language + auditory + motor
#
# Each variant gets its OWN grid dir and its OWN HF repo, so a masked result can
# never overwrite the unmasked one and the two stay directly comparable.
#
# Usage:  bash scripts/run_roi_sweeps.sh [variant ...]      (default: all three)
set -uo pipefail
ROOT=/local/scratch/sas245/brainalign-evals
cd "$ROOT"
PY=/local/scratch/sas245/venvs/mergeability/bin/python
RDM_BASE="$ROOT/data/ds003604-session-rdms"
export HF_TOKEN="${HF_TOKEN:-$(cat /local/scratch/sas245/hf_cache/token)}"
GPU="${GPU:-1}"
DATASETS="${DATASETS:-ds002236 ds006239 ds003604}"
VARIANTS="${*:-roi-language roi-phonology roi-all}"
mkdir -p logs
log(){ echo "[roi $(date -u +%FT%TZ)] $*" | tee -a logs/roi_sweeps.log; }

for V in $VARIANTS; do
  # Only sweep datasets whose masked tree actually exists -- a missing variant is
  # "not built yet", never an empty result.
  PRESENT=""
  for DS in $DATASETS; do
    [ -d "$RDM_BASE/$DS/$V" ] && PRESENT="$PRESENT $DS"
  done
  PRESENT="$(echo "$PRESENT" | xargs)"
  if [ -z "$PRESENT" ]; then
    log "SKIP $V -- no masked RDM tree found under $RDM_BASE/*/$V"
    continue
  fi
  log ">>> $V: sweeping [$PRESENT] on GPU $GPU"
  env CUDA_VISIBLE_DEVICES="$GPU" RDM_VARIANT="$V" DATASETS="$PRESENT" \
      GRID="$ROOT/grid_$V" MAX_CKPT="${MAX_CKPT:-12}" \
      bash scripts/sweep.sh >>"logs/roi_sweep_${V}.log" 2>&1
  log "<<< $V sweep rc=$?"

  for DS in $PRESENT; do
    "$PY" scripts/build_devai_package.py --dataset "$DS" --rdm-variant "$V" \
        --grid-dir "grid_$V/$DS" --push >>"logs/roi_package_${V}.log" 2>&1
    log "    packaged+pushed $DS/$V rc=$?"
  done
done

# Side-by-side masked vs unmasked, which is the whole point of running these.
"$PY" scripts/roi_comparison.py >>logs/roi_sweeps.log 2>&1
log "ROI SWEEPS COMPLETE rc=$?"
