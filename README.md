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

### Local Execution

```bash
# Run full pipeline
python run_pipeline.py

# Run specific steps
python run_pipeline.py --steps preprocess rsa

# Specify subjects/sessions
python run_pipeline.py --subjects sub-5007 --sessions ses-7 ses-9
```

### SLURM Cluster

```bash
# Full pipeline
sbatch slurm/full_pipeline.sh

# Individual steps
sbatch slurm/download_data.sh
sbatch slurm/preprocess.sh
sbatch slurm/run_rsa.sh

# Parallel preprocessing (one job per subject)
sbatch slurm/preprocess_parallel.sh
```

## Pipeline Steps

### 1. Download Data
```bash
python scripts/batch_download_bold.py --data-dir data/brain/ds003604
```

Downloads BOLD fMRI files from OpenNeuro for all subjects.

### 2. Preprocess
```bash
python src/preprocessing/batch_preprocessing.py \
    --data-dir data/brain/ds003604 \
    --output-dir data/processed/fmri
```

Applies spatial smoothing (6mm FWHM), high-pass filtering (0.01 Hz), GLM with canonical HRF, and extracts stimulus-specific patterns.

### 3. Session-Based RSA
```bash
python src/rsa/session_based_rsa.py \
    --pattern-dir data/processed/fmri \
    --aggregation hyperalignment
```

Creates 3 session-level RDMs (ses-5, ses-7, ses-9) using hyperalignment to align subjects to shared representational space before aggregation.

## Output Files

- `sub-*_ses-*_run-*_patterns.npz` - Per-subject neural patterns
- `session_rdm_ses-*.npz` - Session-level RDMs (aggregated across subjects)
- `session_rdm_ses-*.png` - RDM visualizations
- `session_rdm_comparison.csv` - Between-session correlations

## Key Features

- Handles variable sessions and runs across subjects
- Uses hyperalignment (SRM) for robust cross-subject aggregation
- Supports SLURM parallel processing
- No GPU required (CPU and memory intensive only)

## Configuration

Pipeline parameters:
- `--smoothing-fwhm`: Spatial smoothing (default: 6.0mm)
- `--high-pass`: Filter cutoff (default: 0.01 Hz)
- `--aggregation`: hyperalignment, mean, or median (default: hyperalignment)
- `--n-iter`: SRM iterations (default: 10)
- `--metric`: correlation, euclidean, or cosine (default: correlation)

## Citation

Dataset: OpenNeuro ds003604 (https://openneuro.org/datasets/ds003604)

- [ ] Extract language model representations for stimuli
- [ ] Compute model RDMs
- [ ] Compare neural vs model RDMs


## License

This project is licensed under the MIT License.
