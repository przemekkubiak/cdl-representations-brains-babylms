# Representational Similarity Analysis: Brain Data and BabyLMs

This project explores the relationship between brain representations from neuroimaging data and representations learned by small-scale language models (BabyLMs) using Representational Similarity Analysis (RSA).

## Project Overview

This repository contains code and analyses for comparing neural representations from:
- Neuroimaging data (fMRI/MEG/EEG)
- Baby Language Models (BabyLMs) - small-scale language models

## Directory Structure

```
├── data/                 # Data directory (add to .gitignore)
│   ├── brain/           # Neuroimaging data
│   └── babylm/          # Language model data
├── src/                 # Source code
│   ├── preprocessing/   # Data preprocessing scripts
│   ├── models/          # Model loading and feature extraction
│   └── rsa/            # RSA analysis code
├── results/             # Analysis results and figures
├── configs/             # Configuration files
└── tests/              # Unit tests

```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Downloading the Neuroimaging Data

The neuroimaging data (OpenNeuro dataset ds003604) is available at:
https://github.com/OpenNeuroDatasets/ds003604.git

**Important:** This dataset is very large. Download it on a remote machine with sufficient storage.

### Option 1: Python script
```bash
# On the remote machine
source venv/bin/activate
python scripts/download_brain_data.py
```

### Option 2: Bash script
```bash
# On the remote machine
chmod +x scripts/download_brain_data.sh
./scripts/download_brain_data.sh
```

### Option 3: Manual download
```bash
# On the remote machine
mkdir -p data/brain
cd data/brain
git clone https://github.com/OpenNeuroDatasets/ds003604.git
```

The data will be downloaded to `data/brain/ds003604/`

## Usage

Documentation coming soon.

## Citation

If you use this code, please cite:
```
[Add citation information]
```

## License

This project is licensed under the MIT License.
