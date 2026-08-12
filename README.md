# Representational Similarity Analysis: Brain Data and BabyLMs

Compare neural representations from fMRI data with language model representations using RSA.

## Overview

This repository analyzes neural representations across multiple subjects and timepoints (ses-5, ses-7, ses-9) using:
- fMRI preprocessing with GLM and HRF modeling
- Hyperalignment to align subjects to common representational space
- Session-level RDM computation aggregated across subjects

## Directory Structure

```
├── data/
│   ├── brain/ds003604/         # BIDS neuroimaging data
│   └── processed/fmri/         # Extracted patterns and RDMs
├── src/
│   ├── preprocessing/          # fMRI preprocessing
│   └── rsa/                    # RSA analysis
├── scripts/                    # Download utilities
├── slurm/                      # Cluster job scripts
└── run_pipeline.py             # Main pipeline orchestrator
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Download all tasks used in the study
python run_download.py

# Run the analysis for one task
python run_analysis.py --task Sem

# Run ROI sweeps with human-readable labels
python scripts/run_roi_pipeline.py --task Sem
```

Tasks: `Sem`, `Phon`, `Gram`, `Plaus`.

### Local Execution

```bash
# Download data for all tasks
python run_download.py

# Run one task end to end
python run_analysis.py --task Sem

# Run ROI sweeps for one task
python scripts/run_roi_pipeline.py --task Sem
```

### SLURM Cluster

```bash
# Download data
sbatch slurm/download_data.sh

# Parallel ROI runs
python scripts/run_roi_pipeline.py --task Sem

# Parallel preprocessing (one job per subject)
sbatch slurm/preprocess_parallel.sh
```

## Pipeline Steps

### 1. Download Data
```bash
python run_download.py
```

Downloads BOLD fMRI files for Sem, Phon, Gram, and Plaus.

### 2. Preprocess
```bash
python src/preprocessing/batch_preprocessing.py \
    --data-dir data/brain/ds003604 \
    --output-dir data/processed/fmri \
    --task Sem
```

Applies spatial smoothing, high-pass filtering, GLM with canonical HRF, and extracts task-specific patterns.

### 3. Session-Based RSA
```bash
python src/rsa/session_based_rsa.py \
    --pattern-dir data/processed/fmri \
    --task Sem \
    --aggregation hyperalignment
```

Creates session-level RDMs for the selected task across ses-5, ses-7, and ses-9.

## Output Files

- `sub-*_ses-*_run-*_patterns.npz` - Per-subject neural patterns
- `session_rdm_ses-*.npz` - Session-level RDMs (aggregated across subjects)
- `session_rdm_ses-*.png` - RDM visualizations
- `session_rdm_comparison.csv` - Between-session correlations

## Configuration

Pipeline parameters:
- `--smoothing-fwhm`: Spatial smoothing (default: 6.0mm)
- `--high-pass`: Filter cutoff (default: 0.01 Hz)
- `--task`: Task to process or download (`Sem`, `Phon`, `Gram`, `Plaus`)
- `--aggregation`: hyperalignment, mean, or median (default: hyperalignment)
- `--n-iter`: SRM iterations (default: 10)
- `--metric`: correlation, euclidean, or cosine (default: correlation)

## Data Info

Dataset: OpenNeuro ds003604 (https://openneuro.org/datasets/ds003604)

## License

This project is licensed under the MIT License.
