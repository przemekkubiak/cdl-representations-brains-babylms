#!/bin/bash
# Quick start script for language model RDM pipeline
# Tests all components with sample models

set -e

REPO_DIR="/Users/przemyslawkubiak/cdl-representations-brains-babylms"
OUTPUT_DIR="$REPO_DIR/data/processed/language_models"

echo "=========================================="
echo "Language Model RDM Pipeline - Quick Start"
echo "=========================================="

# Check Python environment
echo ""
echo "[1/4] Checking Python environment..."
python --version
python -c "import transformers; import torch; import whisper; import scipy; print('✓ All dependencies installed')"

# Create output directory
echo ""
echo "[2/4] Setting up output directory..."
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory: $OUTPUT_DIR"

# Test speech recognition (optional - slow)
echo ""
echo "[3/4] Testing speech recognition pipeline..."
echo "Note: First time loading Whisper model (~140MB download)"
python "$REPO_DIR/src/language_models/speech_recognition.py" \
  --task Sem \
  --model-size base \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | head -50 || true

# Test language model RDM computation
echo ""
echo "[4/4] Testing language model RDM computation..."
echo "Computing RDM for gpt2..."
python "$REPO_DIR/run_language_models.py" \
  --models gpt2 \
  --task Sem \
  --compare-sessions ses-7 \
  --output-dir "$OUTPUT_DIR" \
  --brain-rdm-dir "$REPO_DIR/data/processed/fmri"

# Show results
echo ""
echo "=========================================="
echo "✓ Quick start complete!"
echo "=========================================="
echo ""
echo "Output files generated:"
ls -lh "$OUTPUT_DIR"/*.npz 2>/dev/null | tail -5 || echo "  (RDM files)"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | tail -5 || echo "  (Results)"

echo ""
echo "Next steps:"
echo "1. Compute RDMs for more models:"
echo "   python run_language_models.py --models gpt2 distilbert-base-uncased roberta-base"
echo ""
echo "2. Generate plots:"
echo "   python run_language_models.py --models gpt2 --plot"
echo ""
echo "3. Compare across all sessions:"
echo "   python run_language_models.py --compare-sessions ses-5 ses-7 ses-9"
echo ""
echo "4. Load and inspect results:"
echo "   python -c \"import json; print(json.load(open('$OUTPUT_DIR/language_model_results.json')))\""
