#!/bin/bash
#SBATCH --job-name=fmri_preproc
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=batch

# fMRI preprocessing for all subjects
# This job does not require GPU but is CPU and memory intensive

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32GB"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules (adjust for your cluster)
# module load python/3.9

# Activate virtual environment
source venv/bin/activate

# Create output directories
mkdir -p logs
mkdir -p data/processed/fmri

# Set number of parallel jobs for nilearn
export NILEARN_CACHE=nilearn_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "Preprocessing all subjects..."
python src/preprocessing/batch_preprocessing.py \
    --data-dir data/brain/ds003604 \
    --output-dir data/processed/fmri \
    --smoothing-fwhm 6.0 \
    --high-pass 0.01

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
