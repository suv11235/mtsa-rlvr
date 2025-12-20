#!/bin/bash
# RunPod Setup Script for MTSA-RLVR
# Run this after starting a RunPod instance with PyTorch template

set -e

echo "======================================"
echo "MTSA-RLVR RunPod Setup"
echo "======================================"

# Navigate to project directory
cd /workspace/MTSA || cd ~/MTSA || { echo "Please cd to MTSA directory first"; exit 1; }

# Install dependencies
echo ">>> Installing Python dependencies..."
pip install -r requirements.txt

# Install flash-attention (optional, for speed)
echo ">>> Installing flash-attention (may take a while)..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install failed, continuing without it"

# Verify installation
echo ">>> Verifying RLVR installation..."
python -c "
from src.rlvr.core_algos import compute_grpo_outcome_advantage, AdvantageEstimator
from src.rlvr.reward_manager import NaiveRewardManager
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
echo ""
echo "To run RLVR training:"
echo "  Attack mode:  bash script/run_rlvr_attack.sh Qwen/Qwen2.5-7B-Instruct datasets/attack_target"
echo "  Defence mode: bash script/run_rlvr_defence.sh Qwen/Qwen2.5-7B-Instruct datasets/attack_target"
echo ""
echo "For a dry run test:"
echo "  python -m src.algorithm.mt_rlvr_train --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct --dry_run"
echo ""
