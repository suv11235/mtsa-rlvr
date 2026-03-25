#!/bin/bash
#SBATCH --job-name=unit-test-interactive-multi-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --output=MTSA/logs/%x-%j.out
#SBATCH --error=MTSA/logs/%x-%j.err

# Activation
cd ~/mtsa-rlvr/MTSA/MTSA
source ../venv/bin/activate

# Environment
export HF_HOME=/data/suvajit_majumder/huggingface_cache
export VLLM_USE_V1="0"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_MODE="disabled" # Disable wandb for debug
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Explicit 1+1 GPU Setup (matching deployment)
export CUDA_VISIBLE_DEVICES=0,1
export AUX_GPU="cuda:1"

echo ">>> STARTING UNIT TESTS IN MULTI-GPU INTERACTIVE MODE"

# Running test_training_flow.py using accelerate launch to simulate distributed env
accelerate launch --config_file script/accelerate_configs/zero2_2gpu.yaml \
    tests/test_training_flow.py

if [ $? -eq 0 ]; then
    echo ">>> UNIT TESTS PASSED"
else
    echo ">>> UNIT TESTS FAILED"
    exit 1
fi
