# Representational Similarity and Circuit Localization: Brain Data and BabyLMs

Compare the learning dynamics of language models to child language development
(OpenNeuro ds003604, Wang et al. 2022; children scanned at ages 5, 7, 9). The
repository measures three things along a shared developmental axis (LM training
checkpoint ~ child age) for each phenomenon (Sem, Phon, Gram, Plaus):

- Alignment: brain-LM representational similarity (RSA).
- LM localization: whether each phenomenon is carried by a specialized circuit
  or spread across the network (functional-localizer method).
- Brain localization: per-voxel condition>control selectivity, computed with the
  same metrics so brain and model are directly comparable.

Tasks / phenomena: `Sem` (semantic), `Phon` (phonological), `Gram` (grammatical),
`Plaus` (plausibility). Sessions: `ses-5`, `ses-7`, `ses-9` (ages 5, 7, 9).

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Directory Structure

```
configs/model_zoo.yaml              Model families with ordered checkpoints
contrasts/                          Localizer contrasts from ds003604 stimuli
src/
  contrast_spec.py                  Shared condition>control mapping
  language_models/
    circuit_localization.py         LM localization + specialization + ablation
    babylm_integration.py           ModelZoo checkpoint registry
    language_model_rdm.py           LM RDM extraction
  rsa/
    brain_localization.py           Brain-side voxel localizer
    session_based_rsa.py            Session-level brain RDMs
  preprocessing/                    fMRI preprocessing
scripts/
  build_contrasts.py                Build contrasts from ds003604 stimuli
  run_circuit_localization.py       LM localization across checkpoints
  checkpoint_alignment_trajectory.py  Brain-LM RSA across checkpoints
  run_brain_localization.py         Brain specialization across sessions
  codla_compare.py                  Join the three axes, test claims C1-C4
  run_codla_pipeline.sh             End-to-end orchestration per model family
```

## Models

Model families and their training checkpoints are defined in
`configs/model_zoo.yaml`. Checkpoints auto-download from HuggingFace on first use.

- `pythia-160m`, `pythia-410m`: dense training trajectory (EleutherAI Pythia).
- `babylm-gpt2`: child-scale in-domain models (https://huggingface.co/BrainAlign,
  gpt2-babylm-3/5/7/9).

List families and resolved checkpoints:

```bash
python -c "from src.language_models.babylm_integration import ModelZoo; z=ModelZoo(); print(z.list_families()); print([c['ref'] for c in z.resolve_checkpoints('pythia-160m')])"
```

## Quick Start (whole pipeline)

```bash
# Build localizer contrasts from the real ds003604 stimuli
python scripts/build_contrasts.py --source github --out-dir contrasts

# Run the full co-development pipeline for one model family
scripts/run_codla_pipeline.sh pythia-160m 1.0 --ablate
```

`run_codla_pipeline.sh` builds contrasts, runs LM localization, runs the brain-LM
RSA trajectory per phenomenon, and joins everything with `codla_compare.py`.
Omit the family argument to run both headline families (pythia-160m, babylm-gpt2).

## Experiments

### 1. LM circuit localization (specialization vs spread)

Localize each phenomenon's circuit at every checkpoint and measure how localized
it is (Gini, selectivity index), how differentiated the phenomena are (cross-
phenomenon overlap), and, optionally, whether the circuit is causal (ablation).

```bash
python scripts/run_circuit_localization.py \
    --model pythia-160m \
    --contrast-dir contrasts \
    --percentage 1.0 \
    --ablate
```

Outputs (under `data/processed/language_models/circuit_localization/`):
`localization_trajectory_<family>.csv`, per-checkpoint overlap matrices, and
trajectory / heatmap figures.

### 2. Brain-LM RSA alignment across checkpoints

```bash
python scripts/checkpoint_alignment_trajectory.py \
    --model pythia-160m \
    --task Sem \
    --normalize
```

Repeat for `Phon`, `Gram`, `Plaus`. Outputs a CSV and figure of RSA vs training
step per brain session.

### 3. Brain-side localization (specialization by age)

Requires preprocessed patterns (`sub-*_ses-*_*patterns.npz`, see brain pipeline
below).

```bash
python scripts/run_brain_localization.py \
    --pattern-dir data/processed/fmri \
    --characteristics-dir data/brain/ds003604/stimuli/Stimulus_Characteristics
```

Outputs `brain_specialization.csv` (`phenomenon, brain_localization, onset_age`)
and a per-session table.

### 4. Co-development join (CoDLA)

Joins LM localization, brain-LM RSA, and brain specialization, and tests:
C1 LM develops localization; C2 localization tracks alignment (partial correlation
controlling for step); C3 developmental order matches the brain; C4 phenomena
differentiate over training.

```bash
python scripts/codla_compare.py \
    --family pythia-160m \
    --localization data/processed/language_models/circuit_localization/localization_trajectory_pythia-160m.csv \
    --alignment Sem=path/to/align_Sem.csv Phon=path/to/align_Phon.csv Gram=path/to/align_Gram.csv Plaus=path/to/align_Plaus.csv \
    --brain-specialization data/processed/fmri/localization/brain_specialization.csv
```

## Brain RSA Pipeline (preprocessing and RDMs)

```bash
# Download BOLD data for all tasks
python run_download.py

# Run one task end to end (preprocessing + session RSA + noise ceiling)
python run_analysis.py --task Sem

# ROI sweeps with human-readable labels
python scripts/run_roi_pipeline.py --task Sem
```

Preprocess and build RDMs directly:

```bash
python src/preprocessing/batch_preprocessing.py \
    --data-dir data/brain/ds003604 \
    --output-dir data/processed/fmri \
    --task Sem

python src/rsa/session_based_rsa.py \
    --pattern-dir data/processed/fmri \
    --task Sem \
    --aggregation hyperalignment
```

Restrict to language ROIs with `run_analysis.py --aal-rois "7,8,9,10,11,12,67,68,69,70,85,86"`.

## Output Files

- `sub-*_ses-*_run-*_patterns.npz` - per-subject neural patterns
- `session_rdm_ses-*.npz` - session-level RDMs
- `localization_trajectory_<family>.csv` - LM specialization over training
- `brain_specialization.csv` - brain specialization by age
- `codla_summary_<family>.csv` - co-development claim tests C1-C4

## Data

Dataset: OpenNeuro ds003604 (https://openneuro.org/datasets/ds003604);
BIDS mirror with stimulus characteristics at
https://github.com/suchirsalhan/neurodataset_babylm.
Models: https://huggingface.co/BrainAlign.

## License

MIT License.
