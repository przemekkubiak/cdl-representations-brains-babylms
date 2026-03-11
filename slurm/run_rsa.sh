#!/bin/bash
#SBATCH --job-name=session_rsa
#SBATCH --output=logs/rsa_%j.out
#SBATCH --error=logs/rsa_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

# Session-based RSA analysis
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

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p logs
mkdir -p data/processed/fmri

echo ""
echo "Running session-based RSA analysis..."
python src/rsa/session_based_rsa.py \
    --pattern-dir data/processed/fmri \
    --output-dir data/processed/fmri \
    --metric correlation \
    --aggregation hyperalignment

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
