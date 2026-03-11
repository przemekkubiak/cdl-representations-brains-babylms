#!/bin/bash
#SBATCH --job-name=fmri_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=batch

# Download BOLD fMRI data for all subjects
# This job does not require GPU

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules (adjust for your cluster)
# module load python/3.9
# module load git-annex

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Download BOLD files for all subjects
echo ""
echo "Downloading BOLD files..."
python scripts/batch_download_bold.py \
    --data-dir data/brain/ds003604 \
    --task Sem \
    --workers 4

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
