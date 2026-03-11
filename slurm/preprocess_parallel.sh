#!/bin/bash
#SBATCH --job-name=preproc_single
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-49
#SBATCH --partition=batch

# Parallel preprocessing using SLURM job arrays
# Each array task processes one subject
# Adjust --array range based on number of subjects

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
# module load python/3.9

# Activate virtual environment
source venv/bin/activate

# Create output directory
mkdir -p logs
mkdir -p data/processed/fmri

# Set environment variables
export NILEARN_CACHE=nilearn_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Get list of subjects
SUBJECTS=($(ls -d data/brain/ds003604/sub-* | xargs -n1 basename | sort))
N_SUBJECTS=${#SUBJECTS[@]}

# Get subject for this array task
if [ $SLURM_ARRAY_TASK_ID -lt $N_SUBJECTS ]; then
    SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
    
    echo ""
    echo "Processing: $SUBJECT"
    echo ""
    
    # Process single subject
    python src/preprocessing/batch_preprocessing.py \
        --data-dir data/brain/ds003604 \
        --output-dir data/processed/fmri \
        --subjects $SUBJECT \
        --smoothing-fwhm 6.0 \
        --high-pass 0.01
else
    echo "Array task ID $SLURM_ARRAY_TASK_ID exceeds number of subjects ($N_SUBJECTS)"
    exit 0
fi

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
