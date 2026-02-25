#!/bin/bash

set -e  # Exit on error

# Configuration
DATA_DIR="data/brain"
REPO_URL="https://github.com/OpenNeuroDatasets/ds003604.git"

echo "========================================="
echo "OpenNeuro Dataset Download Script"
echo "Dataset: ds003604"
echo "========================================="
echo ""

# Create data directory if it doesn't exist
echo "Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

# Check available disk space
echo "Checking available disk space..."
df -h .

echo ""
read -p "This dataset is large. Do you want to continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# Clone the dataset
echo "Cloning dataset from $REPO_URL"
echo "This may take a while depending on dataset size and network speed..."
cd "$DATA_DIR"
git clone "$REPO_URL" ds003604

echo ""
echo "========================================="
echo "Download complete!"
echo "Dataset location: $DATA_DIR/ds003604"
echo "========================================="

# Optional: Show dataset structure
if [ -d "ds003604" ]; then
    echo ""
    echo "Dataset structure:"
    tree -L 2 ds003604 2>/dev/null || ls -la ds003604
fi
