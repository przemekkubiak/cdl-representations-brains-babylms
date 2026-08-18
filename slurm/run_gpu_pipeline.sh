#!/bin/bash
#SBATCH --job-name=codla_gpu
#SBATCH --output=logs/codla_gpu_%j.out
#SBATCH --error=logs/codla_gpu_%j.err
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#
# GPU pipeline: interpretability (circuit localization) + brain-LM RSA trajectory
# + CoDLA join, for one or more model-zoo families.
#
# The GPU-bound steps are the two that push LM checkpoints through forward passes:
#   1. scripts/run_circuit_localization.py     (interp: per-unit selectivity, ablation)
#   2. scripts/checkpoint_alignment_trajectory.py  (brain-LM RSA vs training step)
# Both auto-select CUDA when a GPU is visible. The CoDLA join (correlations/stats)
# is CPU and runs in seconds.
#
# Usage:
#   sbatch slurm/run_gpu_pipeline.sh [family ...]        # SLURM
#   bash   slurm/run_gpu_pipeline.sh [family ...]        # bare machine (no SLURM)
# Examples:
#   sbatch slurm/run_gpu_pipeline.sh pythia-160m
#   sbatch slurm/run_gpu_pipeline.sh pythia-160m babylm-gpt2
#   bash   slurm/run_gpu_pipeline.sh                     # both headline families
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- config (override via env) -------------------------------------------- #
PCT="${PCT:-1.0}"                    # top-% units in the circuit
PHENOMENA=(${PHENOMENA:-Sem Phon Gram Plaus})
ABLATE="${ABLATE:-1}"                # 1 => causal ablation in the interp step
NORMALIZE="${NORMALIZE:-1}"          # 1 => z-normalize RDMs in the RSA step
BOOTSTRAP="${BOOTSTRAP:-1}"          # 1 => bootstrap CIs on RSA correlations
CONTRASTS="${CONTRASTS:-contrasts}"
BRAIN_RDM_ROOT="${BRAIN_RDM_ROOT:-data/processed/fmri}"   # per-task brain RDMs live under $BRAIN_RDM_ROOT/<Task>
LOC_DIR="data/processed/language_models/circuit_localization"
ALIGN_DIR="data/processed/language_models/checkpoint_trajectory"
FAMILIES=("$@")
# FULL-PAPER default: dense Pythia + scale covariate + babylm data-efficiency series.
[ ${#FAMILIES[@]} -eq 0 ] && FAMILIES=(
  pythia-160m-full pythia-410m-full
  babylm-gpt2-3 babylm-gpt2-5 babylm-gpt2-7 babylm-gpt2
)

# --- environment ----------------------------------------------------------- #
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs "$LOC_DIR" "$ALIGN_DIR"

# activate an env if present (venv or conda); harmless if neither exists
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
if [ -n "${CONDA_ENV:-}" ]; then source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"; fi

echo "=========================================="
echo "CoDLA GPU pipeline"
echo "Job: ${SLURM_JOB_ID:-local}   Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Families: ${FAMILIES[*]}   pct=$PCT   ablate=$ABLATE   normalize=$NORMALIZE"
echo "Start: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=========================================="

abl_flag=();      [ "$ABLATE"    = "1" ] && abl_flag=(--ablate)
norm_flag=();     [ "$NORMALIZE" = "1" ] && norm_flag=(--normalize)
boot_flag=();     [ "$BOOTSTRAP" = "1" ] && boot_flag=(--bootstrap-ci)

# contrasts from the real ds003604 stimuli (idempotent, CPU, cheap)
[ -f "$CONTRASTS/Sem.csv" ] || python scripts/build_contrasts.py --source github --out-dir "$CONTRASTS"

run_family () {
  local family="$1"
  echo ""; echo "######## $family ########"

  # 1. INTERP: circuit localization + differentiation (+ causal ablation) [GPU]
  echo "--- [1/3] circuit localization (interp): $family"
  python scripts/run_circuit_localization.py \
      --model "$family" \
      --contrast-dir "$CONTRASTS" \
      --phenomena "${PHENOMENA[@]}" \
      --percentage "$PCT" \
      --batch-size "${BATCH_SIZE:-16}" \
      "${abl_flag[@]}" \
      --output-dir "$LOC_DIR"

  # 2. RSA: brain-LM alignment trajectory per phenomenon [GPU]
  #    (needs brain RDMs in data/processed/fmri — produced by the CPU pipeline)
  local align_args=()
  for T in "${PHENOMENA[@]}"; do
    echo "--- [2/3] RSA trajectory: $family / $T"
    if ! ls "$BRAIN_RDM_ROOT/$T"/session_rdm_ses-*.npz >/dev/null 2>&1; then
      echo "  (skip RSA $T — no brain RDMs in $BRAIN_RDM_ROOT/$T; run the brain pipeline for $T first)"
      continue
    fi
    python scripts/checkpoint_alignment_trajectory.py \
        --model "$family" --task "$T" \
        --brain-rdm-dir "$BRAIN_RDM_ROOT/$T" \
        --output-dir "$ALIGN_DIR/$T" \
        "${norm_flag[@]}" "${boot_flag[@]}" || { echo "  (RSA $T failed)"; continue; }
    local csv
    csv="$(ls -t "$ALIGN_DIR/$T"/checkpoint_alignment_trajectory*.csv 2>/dev/null | head -1 || true)"
    [ -n "$csv" ] && align_args+=("$T=$csv")
  done

  # 3. CoDLA join + claims C1-C4 [CPU, fast]
  echo "--- [3/3] CoDLA join: $family"
  local brain_arg=()
  [ -f data/processed/fmri/localization/brain_specialization.csv ] && \
      brain_arg=(--brain-specialization data/processed/fmri/localization/brain_specialization.csv)
  python scripts/codla_compare.py --family "$family" \
      --localization "$LOC_DIR/localization_trajectory_${family}.csv" \
      ${align_args:+--alignment "${align_args[@]}"} \
      "${brain_arg[@]}" || echo "  (codla_compare skipped/failed for $family)"
}

for fam in "${FAMILIES[@]}"; do run_family "$fam"; done

echo ""
echo "=========================================="
echo "DONE  $(date)"
echo "Interp:    $LOC_DIR/localization_trajectory_<family>.csv"
echo "RSA:       $ALIGN_DIR/<Phen>/checkpoint_alignment_trajectory*.csv"
echo "CoDLA:     data/processed/language_models/codla/codla_summary_<family>.csv"
echo "=========================================="
