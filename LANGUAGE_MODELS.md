# Language Model RDM Pipeline

This module computes representational dissimilarity matrices (RDMs) from language models and compares them with brain RDMs.

## Overview

The pipeline consists of three main components:

### 1. Speech Recognition (`speech_recognition.py`)
Transcribes stimulus `.wav` files using OpenAI Whisper.

**Usage:**
```bash
python src/language_models/speech_recognition.py \
  --task Sem \
  --model-size base \
  --output-dir data/processed/language_models
```

**Output:**
- `transcriptions_Sem.json`: Transcriptions and metadata for each stimulus

### 2. Language Model RDM Computation (`language_model_rdm.py`)
Extracts word embeddings from language models and computes RDMs.

**Supported models:**
- Hugging Face models (e.g., `gpt2`, `distilbert-base-uncased`)
- BabyLM models (from `configs/` directory)
- Any model on Hugging Face Hub

**Usage:**
```bash
python src/language_models/language_model_rdm.py \
  --model gpt2 \
  --task Sem \
  --layer -1 \
  --pooling mean
```

**Output:**
- `lm_rdm_{model_name}_Sem_layer-1.npz`: RDM matrix

### 3. Unified Runner (`run_language_models.py`)
Orchestrates RDM computation and brain comparison.

**Usage:**
```bash
# Compute RDMs for multiple models and compare with brain
python run_language_models.py \
  --models gpt2 "distilbert-base-uncased" \
  --task Sem \
  --compare-sessions ses-5 ses-7 ses-9 \
  --plot
```

**Output:**
- Language model RDMs (`.npz` files)
- Comparison statistics (`.json`)
- Heatmap plots (`.png` files)

## RDM Computation Details

### Stimulus RDM
For each stimulus (word pair in the semantic task):
1. Extract embeddings for word_A and word_B
2. Average embeddings: `stimulus_emb = (emb_A + emb_B) / 2`
3. Compute dissimilarity matrix as 1 - cosine_similarity

### Why Average Word Pairs?
The semantic task presents word pairs (e.g., "syrup" + "pancakes"). Averaging their embeddings captures the conceptual association between words, which aligns with neural processing of semantic relationships.

## Configuration

### Stimulus Characteristics
Word pairs and their attributes are defined in:
```
data/brain/ds003604/stimuli/Stimulus_Characteristics/task-Sem_Stimulus_Characteristics.tsv
```

Columns include:
- `word_A`, `word_B`: Word pair
- `word_A_frequency`, `word_B_frequency`: Word frequency
- `word_A_semantic_neighbors`: Count of semantic neighbors
- `word_A_B_association_strength`: Semantic association strength

### Model Selection
For BabyLM models, use the model identifier from the configs:
```bash
# Load from local config
python run_language_models.py --models "babylm_60M" "babylm_100M"
```

For Hugging Face models:
```bash
python run_language_models.py --models "gpt2" "distilbert-base-uncased" "t5-small"
```

## Analysis

### Brain-Language Model Correlation
The pipeline computes Spearman correlations between:
- Language model RDM (linguistic dissimilarities)
- Brain RDM (neural dissimilarities)

High correlations suggest language model structure aligns with neural representations.

### Interpretation
- **High correlation (r > 0.5)**: Model captures neural organization
- **Low correlation (r < 0.3)**: Model misses neural-linguistic alignment
- **P-value**: Statistical significance of correlation

## Installation Requirements

```bash
pip install transformers torch whisper pandas scipy numpy scikit-learn matplotlib seaborn
```

## Example Workflow

```bash
# Step 1: Transcribe stimuli (optional validation)
python src/language_models/speech_recognition.py --task Sem

# Step 2: Compute RDMs for multiple models
python run_language_models.py \
  --models gpt2 distilbert-base-uncased roberta-base \
  --task Sem \
  --compare-sessions ses-5 ses-7 ses-9 \
  --plot

# Step 3: Inspect results
ls -lh data/processed/language_models/
cat data/processed/language_models/language_model_results.json
```

## Next Steps

After computing RDMs:
1. **Analyze correlation patterns** across models and sessions
2. **Investigate which model best captures neural organization**
3. **Compare with linguistic properties** (frequency, semantic neighbors, etc.)
4. **Test with different embedding methods** (different layers, pooling strategies)
5. **Integrate with cognitive theories** of language processing

## Troubleshooting

**Model download fails:**
```bash
# Download model manually and specify path
python run_language_models.py --models /path/to/model
```

**GPU out of memory:**
```bash
# Use smaller models or reduce batch size
python run_language_models.py --models "distilbert-base-uncased"
```

**Shape mismatch errors:**
Ensure stimulus files haven't changed. Brain RDM should be (96, 96) for semantic task.
