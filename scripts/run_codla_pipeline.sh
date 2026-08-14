#!/usr/bin/env bash
# End-to-end CoDLA pipeline for one model-zoo family (checkpoints auto-download
# from HuggingFace on first use and are cached under $HF_HOME).
#
# Usage:
#   scripts/run_codla_pipeline.sh <family> [percentage] [--ablate]
# Examples:
#   scripts/run_codla_pipeline.sh pythia-160m 1.0 --ablate
#   scripts/run_codla_pipeline.sh babylm-gpt2 1.0
#
# Families are defined in configs/model_zoo.yaml. Runs both headline families
# by default if none is given.
set -euo pipefail

FAMILY="${1:-}"
PCT="${2:-1.0}"
ABLATE_FLAG=""
for arg in "$@"; do [ "$arg" = "--ablate" ] && ABLATE_FLAG="--ablate"; done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

CONTRASTS="contrasts"
LOC_DIR="data/processed/language_models/circuit_localization"
ALIGN_DIR="data/processed/language_models/checkpoint_trajectory"
PHENOMENA=(Sem Phon Gram Plaus)

run_family () {
  local family="$1"
  echo "########################################################"
  echo "# CoDLA pipeline: $family (pct=$PCT ${ABLATE_FLAG:-})"
  echo "########################################################"

  # 0. contrasts from the real ds003604 stimuli (idempotent)
  [ -f "$CONTRASTS/Sem.csv" ] || python scripts/build_contrasts.py --source github --out-dir "$CONTRASTS"

  # 1. LM circuit localization + differentiation (+ optional causal ablation)
  python scripts/run_circuit_localization.py --model "$family" \
      --contrast-dir "$CONTRASTS" --percentage "$PCT" $ABLATE_FLAG \
      --output-dir "$LOC_DIR"

  # 2. brain-LM RSA alignment trajectory, per phenomenon (existing engine)
  local align_args=()
  for T in "${PHENOMENA[@]}"; do
    python scripts/checkpoint_alignment_trajectory.py --model "$family" --task "$T" \
        --output-dir "$ALIGN_DIR/$T" --normalize || true
    # locate the produced CSV (name carries a model_variant suffix)
    local csv
    csv="$(ls -t "$ALIGN_DIR/$T"/checkpoint_alignment_trajectory*.csv 2>/dev/null | head -1 || true)"
    [ -n "$csv" ] && align_args+=("$T=$csv")
  done

  # 3. join + test C1-C4 (brain specialization optional; add if computed)
  local brain_arg=()
  [ -f data/processed/fmri/localization/brain_specialization.csv ] && \
      brain_arg=(--brain-specialization data/processed/fmri/localization/brain_specialization.csv)

  python scripts/codla_compare.py --family "$family" \
      --localization "$LOC_DIR/localization_trajectory_${family}.csv" \
      --alignment "${align_args[@]}" \
      "${brain_arg[@]}"
}

if [ -z "$FAMILY" ]; then
  for fam in pythia-160m babylm-gpt2; do run_family "$fam"; done
else
  run_family "$FAMILY"
fi

echo "Done. See data/processed/language_models/codla/ for summaries + figures."
