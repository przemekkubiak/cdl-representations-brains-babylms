# Language Model RDM Pipeline - Architecture & Implementation

## Project Overview

This new phase compares **brain representational dissimilarity matrices (RDMs)** with **language model RDMs** to understand how neural representations of language relate to computational language models.

### What We're Building

1. **Speech Recognition Pipeline**: Convert `.wav` stimulus files to text using Whisper
2. **Language Model RDM Computation**: Extract word embeddings and compute RDMs
3. **Brain-LM Comparison**: Correlate brain RDMs with language model RDMs
4. **Statistical Analysis**: Test alignment across sessions and models

---

## Key Components

### 1. Speech Recognition (`src/language_models/speech_recognition.py`)

**Purpose**: Transcribe stimulus `.wav` files to validate understanding

**Key Classes**:
- `SpeechRecognitionPipeline`: Main class for transcription
  - `__init__()`: Load Whisper model (base, small, medium, large options)
  - `transcribe_wav()`: Transcribe single file
  - `process_task()`: Process all stimuli for a task (Sem, Gram, Phon, Plaus)
  - `extract_word_list()`: Get unique words and indices

**Workflow**:
```python
pipeline = SpeechRecognitionPipeline(model_size="base")
results = pipeline.process_task("Sem")  # Process semantic task
words, word_to_idx = pipeline.extract_word_list("Sem")
```

**Output**: `transcriptions_Sem.json` with per-stimulus metadata

---

### 2. Language Model RDM Computation (`src/language_models/language_model_rdm.py`)

**Purpose**: Extract embeddings from language models and compute RDMs

**Key Classes**:

#### LanguageModelEmbeddingExtractor
Loads pretrained models and extracts word embeddings
- `__init__()`: Load model from HuggingFace Hub
  - Parameters: `layer` (which layer to extract), `pooling` (mean/max/cls)
- `extract_word_embedding()`: Get embedding for single word
- `extract_batch_embeddings()`: Batch embed multiple words

**Supported Models**:
- GPT-2, GPT-3, BERT, RoBERTa, DistilBERT
- Any HuggingFace model with hidden states
- BabyLM models (custom paths)

#### LanguageModelRDMComputer
Computes RDMs from embeddings
- `load_stimulus_characteristics()`: Parse TSV with word pairs
- `compute_stimulus_rdm()`: RDM from stimulus embeddings
- `compute_rdm_from_embeddings()`: General RDM from embedding matrix
- `save_rdm()`: Store as `.npz`
- `load_rdm()`: Reload from `.npz`

**RDM Computation Logic**:
```
For each stimulus (word pair):
  1. Get embedding for word_A: emb_A
  2. Get embedding for word_B: emb_B
  3. Average: stim_emb = (emb_A + emb_B) / 2
  4. Normalize embeddings
  5. Compute: dissimilarity = 1 - cosine_similarity
  6. Ensure diagonal = 0, matrix symmetric
```

**Why Average Word Pairs?**
- Semantic task presents word pairs (e.g., "syrup" ↔ "pancakes")
- Neural activity reflects the conceptual association
- Averaging captures relationship between words

---

### 3. Unified Runner (`run_language_models.py`)

**Purpose**: Orchestrate end-to-end pipeline

**Key Class**:
- `LanguageModelPipeline`: Main orchestrator
  - `compute_lm_rdm()`: Compute RDM for single model
  - `compare_rdms()`: Compare LM RDM with brain RDM (Spearman correlation)
  - `compute_all_models()`: Process multiple models and compare
  - `plot_comparison()`: Generate heatmaps
  - `save_results()`: Export results to JSON

**Workflow**:
```python
pipeline = LanguageModelPipeline()

# Compute for multiple models
results = pipeline.compute_all_models(
    model_names=["gpt2", "distilbert-base-uncased"],
    task="Sem",
    compare_sessions=["ses-5", "ses-7", "ses-9"]
)
```

**Output**:
- Language model RDMs (`.npz` files)
- Comparison statistics (`.json`)
- Heatmap plots (`.png`)

---

## Data Structure

### Input: Stimulus Characteristics
```
data/brain/ds003604/stimuli/Stimulus_Characteristics/
├── task-Sem_Stimulus_Characteristics.tsv
├── task-Gram_Stimulus_Characteristics.tsv
├── task-Phon_Stimulus_Characteristics.tsv
└── task-Plaus_Stimulus_Characteristics.tsv
```

**Semantic Task Example**:
| run | stim_file | word_A | word_B | word_A_frequency | word_A_B_association_strength |
|-----|-----------|--------|--------|------------------|-------------------------------|
| 1 | stereo_1SH03A0.wav | syrup | pancakes | 1916 | 0.503 |
| 1 | stereo_1SH04A0.wav | trash | garbage | 9768 | 0.526 |

### Input: Stimulus Audio
```
data/brain/ds003604/stimuli/
├── Sem/
│   ├── Sem_run-01/
│   │   ├── stereo_1SH03A0.wav
│   │   ├── stereo_1SH04A0.wav
│   │   └── ... (96 total for Sem)
│   └── Sem_run-02/
├── Gram/
├── Phon/
└── Plaus/
```

### Output: Language Model RDMs
```
data/processed/language_models/
├── lm_rdm_gpt2_Sem_layer-1.npz        # (96, 96)
├── lm_rdm_distilbert_Sem_layer-1.npz  # (96, 96)
├── language_model_results.json         # Correlations with brain
├── rdm_comparison_gpt2_ses-7.png       # Heatmap plots
└── transcriptions_Sem.json             # Optional validation
```

### Output: Brain RDMs (Already Generated)
```
data/processed/fmri/
├── session_rdm_ses-5.npz  # (96, 96)
├── session_rdm_ses-7.npz  # (96, 96)
├── session_rdm_ses-9.npz  # (96, 96)
└── rdm_comparison.csv
```

---

## Usage Examples

### Quick Start: Compute RDM for GPT-2
```bash
cd /Users/przemyslawkubiak/cdl-representations-brains-babylms

python run_language_models.py \
  --models gpt2 \
  --task Sem \
  --compare-sessions ses-7
```

### Multiple Models & Sessions
```bash
python run_language_models.py \
  --models gpt2 "distilbert-base-uncased" "roberta-base" \
  --task Sem \
  --compare-sessions ses-5 ses-7 ses-9 \
  --plot
```

### Generate Transcriptions Only
```bash
python src/language_models/speech_recognition.py \
  --task Sem \
  --model-size base
```

### BabyLM Models
```bash
# If using local BabyLM checkpoints
python run_language_models.py \
  --models "/path/to/babylm_60M" "/path/to/babylm_100M"
```

---

## Statistical Analysis

### Spearman Correlation
Measures monotonic relationship between RDM dissimilarities:
```
High correlation (r > 0.5):
  → Language model structure aligns with neural organization
  
Low correlation (r < 0.3):
  → Model misses key neural-linguistic properties
  
P-value < 0.05:
  → Statistically significant alignment
```

### Why Spearman vs Pearson?
- RDM entries are not independent (many shared stimuli pairs)
- Spearman is more robust to outliers
- Correlation structure easier to interpret

### Cross-Session Reliability
If LM RDM correlates with all three brain sessions:
- Model captures consistent neural properties
- Not fitting noise, but real linguistic structure

---

## Implementation Notes

### Embedding Extraction
**Layer Selection** (`--layer`):
- `-1`: Last layer (context-aware representations)
- `-2`: Second-to-last layer (often best for semantic tasks)
- `0`: Embedding layer (most abstract)

**Pooling Strategy** (`--pooling`):
- `mean`: Average all subword tokens (robust)
- `cls`: Use [CLS] token (BERT convention)
- `max`: Maximum across tokens (rare)

### GPU Optimization
- Models automatically use GPU if available (`torch.cuda`)
- For 96 stimuli: ~30-120 seconds per model
- Embeddings cached after extraction

### Memory Considerations
- Language models: 300MB - 1.5GB (depends on model size)
- Embeddings matrix: 96 × embedding_dim × float32
- RDM matrix: 96 × 96 × float64 = ~73KB

---

## Connection to Broader Project

### Pipeline Integration
```
Raw fMRI Data
    ↓
[Preprocessing] → per-subject patterns
    ↓
[Brain RSA] → brain RDMs (done ✓)
    ├── ses-5: 91 subjects
    ├── ses-7: 217 subjects
    └── ses-9: 88 subjects
    ↓
[NEW] Language Model RDMs
    ├── Extract embeddings from LMs
    ├── Compute dissimilarity matrices
    └── Compare with brain RDMs
    ↓
[Comparison Analysis]
    ├── Which LM best captures neural org?
    ├── Linguistic properties correlation
    └── Cross-modal semantic alignment
```

### Research Questions
1. **Do language models capture neural semantics?**
   - Correlation of LM-RDM with brain-RDM
   
2. **Which model is most brain-like?**
   - Compare correlations across GPT, BERT, BabyLM, etc.
   
3. **What linguistic properties matter?**
   - Frequency, semantic neighbors, morphology
   
4. **Is alignment language-specific?**
   - Compare across tasks (Sem, Gram, Phon, Plaus)

---

## Next Steps

### Phase 1: Model Exploration
- Compute RDMs for: gpt2, distilbert, roberta, t5-small
- Compare across all sessions
- Identify best-correlating models

### Phase 2: Fine-Grained Analysis
- Test different layers and pooling strategies
- Analyze which word pairs drive correlation
- Visualize RDM clustering patterns

### Phase 3: Linguistic Properties
- Correlate word frequency with RDM dissimilarity
- Test semantic neighbor count effect
- Analyze morphosyntactic influences

### Phase 4: BabyLM Integration
- Load trained BabyLM models from configs/
- Compare language model development trajectories
- Test domain-specific language exposure effects

---

## Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| "Model not found" | `pip install transformers` or check model ID on HF Hub |
| GPU out of memory | Use smaller model (distilbert < roberta < gpt2-large) |
| Transcription errors | Increase Whisper model size (base → small) |
| Shape mismatch | Ensure brain RDM is (96, 96); check stimulus file integrity |
| Slow embedding extraction | Use CPU with smaller batch or parallel processing |

---

## File Dependencies

```
src/language_models/
├── __init__.py
├── speech_recognition.py      # Standalone, requires: whisper, pandas
└── language_model_rdm.py       # Standalone, requires: transformers, torch

run_language_models.py           # Orchestrator, requires all above

LANGUAGE_MODELS.md              # This documentation
```

No modifications needed to existing fMRI preprocessing code!

---

## Next in Conversation

Ready to:
1. Test speech recognition on Sem task
2. Compute RDMs for baseline models (gpt2, distilbert)
3. Visualize comparisons with brain RDMs
4. Iterate on layer/pooling strategies
