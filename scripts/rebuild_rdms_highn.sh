#!/bin/bash
# Rebuild every session RDM at a LARGER subject cohort, on idle CPU, while a GPU tier runs.
#
# WHY. Every RDM in the first pass was capped at MAX_SUBJECTS=40. That cap was chosen to fit
# a disk budget that turned out to be a phantom: 361 GB of the "full" disk was a nilearn
# joblib cache that is never re-read. With that gone there is ~600 GB free, and 40-subject
# cohorts are the leading suspect for why the first Tier-1 pass reads null (RSA magnitudes
# ~0.01-0.09, no consistent trend across families).
#
# Measured availability per (task, session), counted from the dataset checkout:
#     ses-5   84-122 subjects | ses-7  217-256 | ses-9   88-101
# Measured cost: 1.48 GB per subject (Phon: 40 subjects -> 59 GB across all their sessions).
# So 90 subjects is ~133 GB peak per task, which fits with room to spare; the ses-7 cohorts
# are capped rather than run whole, because 256 subjects would be ~379 GB and that is not
# affordable next to a running tier.
#
# SAFETY.
#   * Own dataset checkout, so `git checkout -- .` cannot revert another run's downloads.
#   * Own output root, so the RDMs the running tier is reading are never modified underneath
#     it. Swapping RDMs mid-tier would make early and late model families incomparable.
#   * RDM_CACHE=0: does not push to the Hub. The 40-subject RDMs there are verified and good;
#     they get overwritten only after these are checked, and only by an explicit push.
#   * DISK_FLOOR_GB defaults to 420, well above the driver's 350, so this job aborts itself
#     long before it could threaten the tier or the neighbouring merge sweep.
#   * One task at a time.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MAX_SUBJECTS="${MAX_SUBJECTS:-90}"
export JOBS="${JOBS:-24}"
export DISK_FLOOR_GB="${DISK_FLOOR_GB:-420}"
export RDM_CACHE=0
export KEEP_PATTERNS=0
export BRAIN_RDM_ROOT="${BRAIN_RDM_ROOT:-data/processed/fmri_full/ds003604}"
export DATA_DIR="${DATA_DIR:-data/brain/ds003604_full}"

if [ ! -d "$DATA_DIR" ]; then
  git clone --quiet --depth 1 --filter=blob:none --single-branch --branch main \
      https://github.com/OpenNeuroDatasets/ds003604.git "$DATA_DIR" || exit 1
fi

for T in "${@:-Sem Phon Gram Plaus}"; do
  free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if [ "$free" -lt "$((DISK_FLOOR_GB + 60))" ]; then
    echo "[rebuild] STOP before $T: ${free}GB free, too close to the ${DISK_FLOOR_GB}GB floor"
    exit 3
  fi
  echo "[rebuild] === $T at MAX_SUBJECTS=$MAX_SUBJECTS, ${free}GB free ==="
  PHENOMENA="$T" bash prepare_brain_rdms.sh
done
echo "[rebuild] ALL DONE $(date -u +%FT%TZ)"
