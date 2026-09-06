#!/bin/bash
# ROI_SET=motor -- the non-linguistic control for the auditory (phonology) arm.
#
# WHY MOTOR MATTERS. roi_atlas.py defines "phonology" as auditory UNION motor,
# but its own docstring designates motor (precentral gyrus) as the CONTROL
# region: "a useful comparison/control area -- is any residual alignment
# specific to language regions, or generic to any cortex?". The phonology mask
# therefore contains its own control, and roughly half its voxels are motor
# (13,847 total vs 6,876 for auditory alone). Splitting them is the only way to
# read the result:
#     auditory > motor  -> alignment is language-specific
#     auditory ~ motor  -> alignment is generic to cortex
#
# WHY CHAINED, NOT CONCURRENT. Each preprocessing worker holds ~4.7 GB. The
# auditory build already runs 40 of them; at launch time MemAvailable was 118 GB
# with swap at 14/15 GB. Starting a second 40-worker build would OOM-kill both.
# So this waits for the auditory build to exit, then takes the freed capacity.
# Registrations are cached per (subject, session) and are ROI-independent, so
# the motor run reuses all 1,015 of them and pays only for BOLD + GLM.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_HOME=/local/scratch/sas245/hf_cache_brainalign
export HF_TOKEN="$(cat /local/scratch/sas245/.cache/huggingface/token)"
export DATASET=ds003604
export ROI_SET=motor
export WITHIN_RUN_NORM=1
export AGGREGATION=hyperalignment
export MAX_SUBJECTS=0
JOBS="${JOBS:-40}"
STATUS=data/processed/fmri/$DATASET/_masks/roi_mask_status.csv
say(){ echo "[motor $(date -u +%FT%TZ)] $*"; }

say "waiting for the auditory build to finish before claiming memory"
while pgrep -f run_masked_waves_auditory.sh >/dev/null 2>&1; do sleep 120; done
say "auditory build finished; starting motor (JOBS=$JOBS)"

PHENOMENA="Sem Phon Gram Plaus" JOBS="$JOBS" bash prepare_brain_rdms.sh
say "motor build rc=$?"

# Verification: a non-ok registration means that subject fell back to the
# WHOLE-BRAIN mask and its data is not motor-masked, whatever directory it is in.
if [ -s "$STATUS" ]; then
  say "registrations by status:"
  awk -F, 'NR>1 && $3=="motor"{print "    "$4}' "$STATUS" | sort | uniq -c
  awk -F, 'NR>1 && $3=="motor" && $4=="ok"{n++;d+=$8;v+=$9} END{if(n)printf "[motor] n=%d mean_dice=%.3f mean_voxels=%.0f\n",n,d/n,v/n}' "$STATUS"
fi
say "session RDMs: $(find data/processed/fmri/$DATASET/roi-motor -name 'session_rdm_*.npz' 2>/dev/null | wc -l)/12"
say "MOTOR BUILD COMPLETE"
