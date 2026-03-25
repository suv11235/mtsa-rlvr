#!/bin/bash
#SBATCH --job-name=mtsa-safety-eval
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Use 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00            # Should be fast
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

echo ">>> Starting Safety Comparison Evaluation..."
echo ">>> Baseline: meta-llama/Meta-Llama-3-8B-Instruct"
echo ">>> Defense: $CHECKPOINT_PATH"

python script/eval_safety_comparison.py \
    --baseline_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --num_samples 20 \
    --gpu "cuda:0"
