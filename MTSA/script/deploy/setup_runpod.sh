#!/bin/bash
# RunPod Setup Script for MTSA-RLVR
# Run this after starting a RunPod instance with PyTorch template

set -e

echo "======================================"
echo "MTSA-RLVR RunPod Setup"
echo "======================================"

# Navigate to project directory
if [ -f "requirements.txt" ]; then
    echo "Using current directory: $(pwd)"
elif [ -d "/workspace/mtsa-rlvr/MTSA" ]; then
    cd /workspace/mtsa-rlvr/MTSA
elif [ -d "/workspace/MTSA" ]; then
    cd /workspace/MTSA
elif [ -d "~/MTSA" ]; then
    cd ~/MTSA
else
    echo "Error: Could not find MTSA directory. Please run this script from the MTSA root."
    exit 1
fi

VENV_DIR="venv"

# 1. Create Venv
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating virtual environment in ./$VENV_DIR ..."
    # Try creating venv.
    python3 -m venv $VENV_DIR
else
    echo ">>> Virtual environment already exists."
fi

# 2. Activate for this script execution
echo ">>> Activating venv..."
source $VENV_DIR/bin/activate

# Install dependencies
echo ">>> Updating pip..."
pip install --upgrade pip

# Force specific numpy version to avoid 2.x conflicts with PyTorch/SciPy
echo ">>> Installing safe base dependencies (numpy<2.0)..."
pip install "numpy<2.0" "scipy>=1.10.0"

echo ">>> Installing Python dependencies..."
pip install -r requirements.txt

# Ensure TRL and PEFT are installed (sometimes missed due to environment conflicts)
echo ">>> Ensuring TRL and PEFT are installed..."
pip install trl>=0.12.0 peft>=0.13.0

# Install flash-attention (optional, for speed)
# echo ">>> Installing flash-attention (may take a while)..."
# pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install failed, continuing without it"

# Verify installation
echo ">>> Verifying RLVR installation..."
python -c "
from src.rlvr.core_algos import compute_grpo_outcome_advantage, AdvantageEstimator
from src.rlvr.reward_manager import NaiveRewardManager
import numpy as np
import scipy
print(f'Numpy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print('RLVR modules imported successfully!')
"

# Check GPU
echo ">>> GPU Check..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo "IMPORTANT: BEFORE RUNNING ANYTHING:"
echo "Run this command: source venv/bin/activate"
echo "======================================"
