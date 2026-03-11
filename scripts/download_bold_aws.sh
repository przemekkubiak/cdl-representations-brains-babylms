#!/bin/bash

# Download BOLD .nii.gz files for semantic task, subject sub-5007

# Use the AWS CLI from the virtual environment
AWS_CLI="/Users/przemyslawkubiak/cdl-representations-brains-babylms/venv/bin/aws"
DATA_DIR="data/brain/ds003604"
S3_BASE="s3://openneuro.org/ds003604"

echo "Downloading BOLD files for subject sub-5007, Semantic task..."
echo "=============================================================="

# Session 7 files
echo -e "\n[1/4] Downloading ses-7 run-01..."
$AWS_CLI s3 cp --no-sign-request \
  "$S3_BASE/sub-5007/ses-7/func/sub-5007_ses-7_task-Sem_acq-D2S2_run-01_bold.nii.gz" \
  "$DATA_DIR/sub-5007/ses-7/func/sub-5007_ses-7_task-Sem_acq-D2S2_run-01_bold.nii.gz"

echo -e "\n[2/4] Downloading ses-7 run-02..."
$AWS_CLI s3 cp --no-sign-request \
  "$S3_BASE/sub-5007/ses-7/func/sub-5007_ses-7_task-Sem_acq-D2S4_run-02_bold.nii.gz" \
  "$DATA_DIR/sub-5007/ses-7/func/sub-5007_ses-7_task-Sem_acq-D2S4_run-02_bold.nii.gz"

# Session 9 files
echo -e "\n[3/4] Downloading ses-9 run-01..."
$AWS_CLI s3 cp --no-sign-request \
  "$S3_BASE/sub-5007/ses-9/func/sub-5007_ses-9_task-Sem_acq-D2S8_run-01_bold.nii.gz" \
  "$DATA_DIR/sub-5007/ses-9/func/sub-5007_ses-9_task-Sem_acq-D2S8_run-01_bold.nii.gz"

echo -e "\n[4/4] Downloading ses-9 run-02..."
$AWS_CLI s3 cp --no-sign-request \
  "$S3_BASE/sub-5007/ses-9/func/sub-5007_ses-9_task-Sem_acq-D2S6_run-02_bold.nii.gz" \
  "$DATA_DIR/sub-5007/ses-9/func/sub-5007_ses-9_task-Sem_acq-D2S6_run-02_bold.nii.gz"

echo -e "\n=============================================================="
echo "Download complete!"
echo "Verifying files..."
ls -lh "$DATA_DIR"/sub-5007/ses-*/func/*Sem*_bold.nii.gz
