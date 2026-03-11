#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=batch

# Full pipeline: download -> preprocess -> RSA
# This job does not require GPU

set -e

echo "=========================================="
echo "FULL PIPELINE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules (adjust for your cluster)
# module load python/3.9

# Activate virtual environment
source venv/bin/activate

# Create directories
mkdir -p logs
mkdir -p data/processed/fmri

# Set environment variables
export NILEARN_CACHE=nilearn_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Step 1: Download data
echo ""
echo "=========================================="
echo "STEP 1: Downloading BOLD data"
echo "=========================================="
python scripts/batch_download_bold.py \
    --data-dir data/brain/ds003604 \
    --task Sem \
    --workers 4

# Step 2: Preprocess
echo ""
echo "=========================================="
echo "STEP 2: Preprocessing fMRI data"
echo "=========================================="
python src/preprocessing/batch_preprocessing.py \
    --data-dir data/brain/ds003604 \
    --output-dir data/processed/fmri \
    --smoothing-fwhm 6.0 \
    --high-pass 0.01

# Step 3: Session-based RSA
echo ""
echo "=========================================="
echo "STEP 3: Session-based RSA analysis"
echo "=========================================="
python src/rsa/session_based_rsa.py \
    --pattern-dir data/processed/fmri \
    --output-dir data/processed/fmri \
    --metric correlation \
    --aggregation hyperalignment

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Session RDMs: data/processed/fmri/session_rdm_ses-*.npz"
echo "  - Visualizations: data/processed/fmri/session_rdm_ses-*.png"
echo "  - Comparison: data/processed/fmri/session_rdm_comparison.csv"
