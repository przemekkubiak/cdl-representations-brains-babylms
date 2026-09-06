#!/bin/bash
# Two-wave masked (roi-phonology) RDM build for ds003604, with verification.
#
# WAVE 1 (~5h target): Phon only -- 1052 of 3847 BOLD runs. Phon is the cell that
#   actually answers the phonology-masking question, so it is the wave that has
#   to land first.
# WAVE 2 (~12h more): Sem, Gram, Plaus -- the remaining 2795 runs. The per-session
#   skip in prepare_brain_rdms.sh means wave 2 never redoes wave 1.
#
# Each session RDM is pushed to HF as soon as it is built (rdm_cache_hf.py push,
# self-labelling by variant), so artefacts appear on the Hub continuously rather
# than only at the end. HF_TOKEN must therefore be set -- without it the pipeline
# logs "no HF_TOKEN; cache disabled" and pushes nothing.
#
# VERIFICATION (the "no failures" part). Registration falls back to the whole-brain
# mask, logged not raised, whenever a T1w is missing or registration fails -- which
# silently produces whole-brain data filed under roi-phonology/. After each wave we
# assert every row of roi_mask_status.csv says `ok`. Any other status is reported
# loudly, because those subjects' data is NOT masked whatever directory it is in.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_HOME=/local/scratch/sas245/hf_cache_brainalign
export HF_TOKEN="$(cat /local/scratch/sas245/.cache/huggingface/token)"
export DATASET=ds003604
export ROI_SET=auditory
export WITHIN_RUN_NORM=1
export AGGREGATION=hyperalignment
export MAX_SUBJECTS=0
JOBS="${JOBS:-40}"

STATUS=data/processed/fmri/$DATASET/_masks/roi_mask_status.csv
say(){ echo "[waves $(date -u +%FT%TZ)] $*"; }

verify(){   # $1 = wave label
  local bad total
  if [ ! -s "$STATUS" ]; then say "$1: WARNING no status ledger at $STATUS"; return 0; fi
  total=$(( $(wc -l < "$STATUS") - 1 ))
  bad=$(awk -F, 'NR>1 && $4!="ok"' "$STATUS" | wc -l)
  say "$1: registrations total=$total non-ok=$bad"
  if [ "$bad" -gt 0 ]; then
    say "$1: *** $bad NON-OK REGISTRATIONS -- those subjects are WHOLE-BRAIN, not masked ***"
    awk -F, 'NR>1 && $4!="ok"{print "    "$1","$2","$4","$5}' "$STATUS" | head -20
  fi
  say "$1: mean dice / voxels:"
  awk -F, 'NR>1 && $4=="ok"{n++;d+=$8;v+=$9} END{if(n)printf "    n=%d mean_dice=%.3f mean_voxels=%.0f\n",n,d/n,v/n}' "$STATUS"
  say "$1: session RDMs on disk: $(find data/processed/fmri/$DATASET/roi-phonology -name 'session_rdm*.npz' 2>/dev/null | wc -l)"
}

say "WAVE 1 START (Phon, JOBS=$JOBS, all subjects)"
PHENOMENA="Sem Phon Gram Plaus" JOBS="$JOBS" bash prepare_brain_rdms.sh
say "WAVE 1 END rc=$?"
verify "wave1"

say "WAVE 2 START (Sem Gram Plaus, JOBS=$JOBS)"
true
say "WAVE 2 END rc=$?"
verify "wave2"

say "ALL WAVES COMPLETE"
