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
DATASET="${DATASET:-ds003604}"          # neuro dataset tag (Tier-3 cross-dataset axis)
DATA_DIR="${DATA_DIR:-data/brain/$DATASET}"
BRAIN_RDM_ROOT="${BRAIN_RDM_ROOT:-data/processed/fmri/$DATASET}"
CONTRASTS="${CONTRASTS:-contrasts}"
GRID_PARENT="${GRID_PARENT:-data/processed/language_models/devai_grid}"
GRID_DIR="$GRID_PARENT/$DATASET"        # per-dataset so multiple datasets don't collide
DEVAI_DIR="${DEVAI_DIR:-data/processed/language_models/devai}/$DATASET"
BRAIN_SPEC="$BRAIN_RDM_ROOT/localization/brain_specialization.csv"
SKIP_BRAIN="${SKIP_BRAIN:-0}"
MAX_CKPT="${MAX_CKPT:-25}"          # log-subsample dense pico trajectories (0 = all 126)
BATCH_SIZE="${BATCH_SIZE:-16}"

# SMOKE=1: ~2-min sanity run — one small family, 2 checkpoints, no brain, no RSA.
# Confirms pico/Beetle actually load + extract on THIS cluster before the full sweep.
if [ "${SMOKE:-0}" = "1" ]; then
  SKIP_BRAIN=1; MAX_CKPT=2
  [ $# -eq 0 ] && set -- pico-decoder-tiny
  echo ">> SMOKE MODE: family=$* max_ckpt=$MAX_CKPT (isolation+mechanistic only; RSA skipped without brain RDMs)"
fi

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

# Tier-2 toggles: ABLATE=1 (causal, T2.1), BEHAVIOUR default on (T2.2), BOOTSTRAP>0 (T2.5)
ABLATE_FLAG=(); [ "${ABLATE:-1}" = "1" ] && ABLATE_FLAG=(--ablate)
BOOT_FLAG=(); [ "${BOOTSTRAP:-0}" != "0" ] && BOOT_FLAG=(--bootstrap "$BOOTSTRAP")

for FAM in "${FAMILIES[@]}"; do
  echo ""; echo "######## GRID: $FAM ########"
  python scripts/run_devai_grid.py --model "$FAM" --dataset "$DATASET" \
      --contrast-dir "$CONTRASTS" --phenomena "${PHENOMENA[@]}" \
      --brain-rdm-root "$BRAIN_RDM_ROOT" --max-checkpoints "$MAX_CKPT" \
      --batch-size "$BATCH_SIZE" --normalize --output-dir "$GRID_DIR" \
      "${ABLATE_FLAG[@]}" "${BOOT_FLAG[@]}" \
      || { echo "  ! grid failed for $FAM"; continue; }
  python scripts/mechanistic_brain_analysis.py --family "$FAM" \
      --grid-dir "$GRID_DIR" --output-dir "$DEVAI_DIR" "${BRAIN_SPEC_ARG[@]}" \
      || echo "  ! join failed for $FAM"
done

# T2.4: held-out cross-family predictive validation
python scripts/heldout_predictor.py --families "${FAMILIES[@]}" \
    --grid-dir "$GRID_DIR" --out "$DEVAI_DIR" || echo "  ! held-out CV skipped"

# ---- E. publication figures + LaTeX tables ------------------------------- #
echo ""; echo "######## FIGURES ########"
# cross-dataset figure (Fig 10) aggregates every per-dataset grid dir present
GRID_DIRS=("$GRID_PARENT"/*/); [ ${#GRID_DIRS[@]} -eq 0 ] && GRID_DIRS=("$GRID_DIR")
python scripts/make_figures.py --families "${FAMILIES[@]}" \
    --grid-dir "$GRID_DIR" --grid-dirs "${GRID_DIRS[@]}" \
    --devai-dir "$DEVAI_DIR" --out "${FIG_DIR:-figures}/$DATASET" \
    || echo "  ! figure generation failed"

echo ""
echo "=========================================="
echo "DONE  $(date)"
echo "Per-checkpoint: $GRID_DIR/{alignment,isolation,mechanistic,mechanistic_layer}_<family>.csv"
echo "Claims summary: $DEVAI_DIR/devai_summary_<family>.csv"
echo "Isolation cmp:  $DEVAI_DIR/isolation_comparison_<family>.csv"
echo "Figures:        ${FIG_DIR:-figures}/fig{1..6}_*.pdf  + table{1,2}_*.tex"
echo "=========================================="
