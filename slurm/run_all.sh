#!/bin/bash
#SBATCH --job-name=codla_all
#SBATCH --output=logs/codla_all_%j.out
#SBATCH --error=logs/codla_all_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#
# ONE script, end to end, in a single allocation:
#   A. CPU brain pipeline, per task:  download BOLD -> preprocess -> session RDMs
#   B. GPU:  interp (circuit localization) + brain-LM RSA trajectory
#   C. CPU:  CoDLA join + claims C1-C4
#
# Brain RDMs are task-agnostic on disk (session_rdm_<ses>.npz), so each task is
# preprocessed into its OWN dir ($BRAIN_RDM_ROOT/<Task>) and the RSA step points
# --brain-rdm-dir at the matching one. The GPU sits idle during (A); that is the
# cost of keeping this to a single submit.
#
# Usage:
#   sbatch slurm/run_all.sh [family ...]        # SLURM (A100)
#   bash   slurm/run_all.sh [family ...]        # bare machine
# Examples:
#   sbatch slurm/run_all.sh pythia-160m
#   sbatch slurm/run_all.sh pythia-160m babylm-gpt2
#   SKIP_BRAIN=1 bash slurm/run_all.sh pythia-160m   # brain RDMs already built
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- config (override via env) -------------------------------------------- #
PHENOMENA=(${PHENOMENA:-Sem Phon Gram Plaus})
FAMILIES=("$@"); [ ${#FAMILIES[@]} -eq 0 ] && FAMILIES=(pythia-160m babylm-gpt2)
DATA_DIR="${DATA_DIR:-data/brain/ds003604}"
BRAIN_RDM_ROOT="${BRAIN_RDM_ROOT:-data/processed/fmri}"
SKIP_BRAIN="${SKIP_BRAIN:-0}"          # 1 => brain RDMs already exist, jump to GPU
DL_WORKERS="${DL_WORKERS:-4}"
SMOOTHING="${SMOOTHING:-6.0}"
HIGHPASS="${HIGHPASS:-0.01}"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NILEARN_CACHE="${NILEARN_CACHE:-nilearn_cache}"
mkdir -p logs

if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
if [ -n "${CONDA_ENV:-}" ]; then source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"; fi

echo "=========================================="
echo "CoDLA end-to-end (single job)"
echo "Job: ${SLURM_JOB_ID:-local}   Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Families: ${FAMILIES[*]}   Tasks: ${PHENOMENA[*]}   skip_brain=$SKIP_BRAIN"
echo "Start: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=========================================="

# ========================================================================== #
# A. CPU BRAIN PIPELINE (per task, into its own dir)
# ========================================================================== #
if [ "$SKIP_BRAIN" != "1" ]; then
  for T in "${PHENOMENA[@]}"; do
    OUT="$BRAIN_RDM_ROOT/$T"
    if ls "$OUT"/session_rdm_ses-*.npz >/dev/null 2>&1; then
      echo ">> brain RDMs for $T already present in $OUT — skipping"; continue
    fi
    echo ""; echo "######## BRAIN: $T ########"
    mkdir -p "$OUT"

    echo "--- download BOLD ($T)"
    python scripts/batch_download_bold.py --data-dir "$DATA_DIR" --task "$T" --workers "$DL_WORKERS"

    echo "--- preprocess ($T)"
    python src/preprocessing/batch_preprocessing.py \
        --data-dir "$DATA_DIR" --output-dir "$OUT" --task "$T" \
        --smoothing-fwhm "$SMOOTHING" --high-pass "$HIGHPASS"

    echo "--- session RDMs ($T)"
    python src/rsa/session_based_rsa.py \
        --pattern-dir "$OUT" --output-dir "$OUT" \
        --metric correlation --aggregation hyperalignment
  done
else
  echo ">> SKIP_BRAIN=1 — using existing brain RDMs under $BRAIN_RDM_ROOT/<Task>"
fi

# ========================================================================== #
# B + C. GPU interp + RSA trajectory + CoDLA join
# ========================================================================== #
echo ""; echo "######## GPU: interp + RSA + CoDLA ########"
BRAIN_RDM_ROOT="$BRAIN_RDM_ROOT" PHENOMENA="${PHENOMENA[*]}" \
    bash slurm/run_gpu_pipeline.sh "${FAMILIES[@]}"

echo ""
echo "=========================================="
echo "ALL DONE  $(date)"
echo "Brain RDMs:  $BRAIN_RDM_ROOT/<Task>/session_rdm_ses-*.npz"
echo "Interp:      data/processed/language_models/circuit_localization/localization_trajectory_<family>.csv"
echo "RSA:         data/processed/language_models/checkpoint_trajectory/<Task>/checkpoint_alignment_trajectory*.csv"
echo "CoDLA:       data/processed/language_models/codla/codla_summary_<family>.csv"
echo "=========================================="
