#!/bin/bash
#SBATCH --job-name=devai_grid
#SBATCH --output=logs/devai_grid_%j.out
#SBATCH --error=logs/devai_grid_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#
# ONE-SCRIPT DevAI experimental grid (single A100 allocation):
#
#   A. CPU brain (per task): download BOLD -> preprocess -> session RDMs   [reused]
#   B. CPU brain isolation : per-voxel condition>control localizer                (optional)
#   C. GPU per family      : run_devai_grid.py  -> alignment + isolation + mechanistic
#   D. CPU join            : mechanistic_brain_analysis.py -> R1/R2/R3/R5 summary
#
# Every model here is PicoDecoderHF or GPT-2/NeoX and runs through the hook-based
# extractor, so pico + Beetle + babylm + pythia families all work on one code path.
#
# Usage:
#   sbatch slurm/run_devai_grid.sh [family ...]
#   bash   slurm/run_devai_grid.sh [family ...]
#   SKIP_BRAIN=1 bash slurm/run_devai_grid.sh pico-decoder-small   # brain RDMs already built
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- config (override via env) -------------------------------------------- #
PHENOMENA=(${PHENOMENA:-Sem Phon Gram Plaus})
DATA_DIR="${DATA_DIR:-data/brain/ds003604}"
BRAIN_RDM_ROOT="${BRAIN_RDM_ROOT:-data/processed/fmri}"
CONTRASTS="${CONTRASTS:-contrasts}"
GRID_DIR="${GRID_DIR:-data/processed/language_models/devai_grid}"
DEVAI_DIR="${DEVAI_DIR:-data/processed/language_models/devai}"
BRAIN_SPEC="$BRAIN_RDM_ROOT/localization/brain_specialization.csv"
SKIP_BRAIN="${SKIP_BRAIN:-0}"
MAX_CKPT="${MAX_CKPT:-25}"          # log-subsample dense pico trajectories (0 = all 126)
BATCH_SIZE="${BATCH_SIZE:-16}"

# DevAI full-paper default grid: pico scale ladder + Beetle English data-budget axis.
FAMILIES=("$@")
[ ${#FAMILIES[@]} -eq 0 ] && FAMILIES=(
  pico-decoder-tiny pico-decoder-small pico-decoder-medium pico-decoder-large
  beetle-humanscale-eng beetle-fineweb3-eng
)

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NILEARN_CACHE="${NILEARN_CACHE:-nilearn_cache}"
mkdir -p logs "$GRID_DIR" "$DEVAI_DIR"

if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
if [ -n "${CONDA_ENV:-}" ]; then source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"; fi

echo "=========================================="
echo "DevAI grid | Job ${SLURM_JOB_ID:-local} | Node ${SLURM_NODELIST:-$(hostname)}"
echo "Families: ${FAMILIES[*]}"
echo "Tasks: ${PHENOMENA[*]} | max_ckpt=$MAX_CKPT | skip_brain=$SKIP_BRAIN"
echo "Start: $(date)"
python -c "import torch;print('CUDA:',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=========================================="

# contrasts from the real ds003604 stimuli (idempotent, cheap)
[ -f "$CONTRASTS/Sem.csv" ] || python scripts/build_contrasts.py --source github --out-dir "$CONTRASTS"

# ---- A. CPU brain RDMs, per task (into isolated dirs) --------------------- #
if [ "$SKIP_BRAIN" != "1" ]; then
  for T in "${PHENOMENA[@]}"; do
    OUT="$BRAIN_RDM_ROOT/$T"
    if ls "$OUT"/session_rdm_ses-*.npz >/dev/null 2>&1; then
      echo ">> brain RDMs for $T present — skip"; continue
    fi
    echo ""; echo "######## BRAIN: $T ########"; mkdir -p "$OUT"
    python scripts/batch_download_bold.py --data-dir "$DATA_DIR" --task "$T" --workers "${DL_WORKERS:-4}"
    python src/preprocessing/batch_preprocessing.py --data-dir "$DATA_DIR" \
        --output-dir "$OUT" --task "$T" --smoothing-fwhm "${SMOOTHING:-6.0}" --high-pass "${HIGHPASS:-0.01}"
    python src/rsa/session_based_rsa.py --pattern-dir "$OUT" --output-dir "$OUT" \
        --metric correlation --aggregation hyperalignment
  done
  # ---- B. brain isolation (per-voxel localizer) — optional --------------- #
  if [ ! -f "$BRAIN_SPEC" ]; then
    echo ">> brain isolation"
    python scripts/run_brain_localization.py \
        --pattern-dir "$BRAIN_RDM_ROOT" \
        --characteristics-dir "$DATA_DIR/stimuli/Stimulus_Characteristics" || \
        echo "  (brain localization skipped — patterns/characteristics missing)"
  fi
else
  echo ">> SKIP_BRAIN=1"
fi

# ---- C+D. per-family GPU grid + join ------------------------------------- #
BRAIN_SPEC_ARG=()
[ -f "$BRAIN_SPEC" ] && BRAIN_SPEC_ARG=(--brain-specialization "$BRAIN_SPEC")

for FAM in "${FAMILIES[@]}"; do
  echo ""; echo "######## GRID: $FAM ########"
  python scripts/run_devai_grid.py --model "$FAM" \
      --contrast-dir "$CONTRASTS" --phenomena "${PHENOMENA[@]}" \
      --brain-rdm-root "$BRAIN_RDM_ROOT" --max-checkpoints "$MAX_CKPT" \
      --batch-size "$BATCH_SIZE" --normalize --output-dir "$GRID_DIR" \
      || { echo "  ! grid failed for $FAM"; continue; }
  python scripts/mechanistic_brain_analysis.py --family "$FAM" \
      --grid-dir "$GRID_DIR" --output-dir "$DEVAI_DIR" "${BRAIN_SPEC_ARG[@]}" \
      || echo "  ! join failed for $FAM"
done

echo ""
echo "=========================================="
echo "DONE  $(date)"
echo "Per-checkpoint: $GRID_DIR/{alignment,isolation,mechanistic}_<family>.csv"
echo "Claims summary: $DEVAI_DIR/devai_summary_<family>.csv"
echo "Isolation cmp:  $DEVAI_DIR/isolation_comparison_<family>.csv"
echo "=========================================="
