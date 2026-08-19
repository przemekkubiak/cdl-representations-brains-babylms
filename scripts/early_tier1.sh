#!/bin/bash
# Early Tier-1 pass over the FAST families, into a SEPARATE output dir.
#
# Purpose: the GPUs sit idle while stage 0 finishes on CPU, and the driver's Tier 1 is
# the authoritative run. This is strictly additive -- it banks real alignment rows during
# a window that would otherwise be wasted, without touching the directory the driver
# verifies against. It must finish before the driver starts Tier 1, because
# ActivationExtractor always lands on cuda:0 and two concurrent grids would contend for
# the same card.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. env_brainalign.sh
. venv/bin/activate

OUT="${EARLY_OUT:-data/processed/language_models/devai_grid_early/ds003604}"
mkdir -p "$OUT"

# Yield to the driver. ActivationExtractor always lands on cuda:0, so an early pass and
# the driver's Tier 1 would share one card; pico-decoder-large is ~49 GB in fp32 and two
# of those do not fit in 80 GB. The driver creates logs/tier1.log when it starts Tier 1,
# so check before every family and stop rather than contend.
driver_tier1_started() { [ -s logs/tier1.log ]; }

for FAM in "$@"; do
  if driver_tier1_started; then
    echo "EARLY TIER1 YIELDING: driver Tier 1 has started (logs/tier1.log is non-empty)"
    break
  fi
  echo ""; echo "######## EARLY GRID: $FAM ########"
  python scripts/run_devai_grid.py --model "$FAM" --dataset ds003604 \
      --contrast-dir contrasts --phenomena Sem Phon Gram Plaus \
      --brain-rdm-root data/processed/fmri/ds003604 \
      --max-checkpoints 25 --batch-size 16 --normalize --output-dir "$OUT" \
      || echo "  ! early grid failed for $FAM"
done
echo "EARLY TIER1 DONE $(date -u +%FT%TZ)"
