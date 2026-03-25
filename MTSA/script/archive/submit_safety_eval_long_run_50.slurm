#!/bin/bash
#SBATCH --job-name=mtsa-safety-eval-50
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Use 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00            # Increased time for 50 samples
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Path setup
ROOT_DIR=~/mtsa-rlvr
MTSA_DIR=$ROOT_DIR/MTSA

cd $MTSA_DIR
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set HF Token
export $(grep -v '^#' .env | xargs)

# Performance optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

CHECKPOINT_PATH="./outputs/rlvr_mtsa_long_run/checkpoint-50"

echo ">>> Starting Safety Comparison Evaluation (50 samples)..."
echo ">>> Baseline: meta-llama/Meta-Llama-3-8B-Instruct"
echo ">>> Defense: $CHECKPOINT_PATH"
echo ">>> Dataset: datasets/attack_target/HarmBench.json"

python script/eval_safety_comparison.py \
    --baseline_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --dataset_path "datasets/attack_target/HarmBench.json" \
    --num_samples 50 \
    --gpu "cuda:0"
