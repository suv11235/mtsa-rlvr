#!/bin/bash
#SBATCH --job-name=tamper-eval-base
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1               # Use 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00            # Short test run
#SBATCH --exclude=compute-837
#SBATCH --account=zhijing_jin
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Path setup
ROOT_DIR=~/mtsa-rlvr
MTSA_DIR=$ROOT_DIR/MTSA
TAR_REPO_DIR=$ROOT_DIR/tamper-resistance-repo

cd $MTSA_DIR
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd):$TAR_REPO_DIR

# Set HF Token
export $(grep -v '^#' .env | xargs)

# Run Tampering Evaluation (SFT Attack) on BASE MODEL
echo ">>> Starting Tampering Evaluation on BASE MODEL (meta-llama/Meta-Llama-3-8B-Instruct)..."
cd $TAR_REPO_DIR

accelerate launch --num_processes 1 --mixed_precision bf16 \
    -m red_teaming.red_teaming_evaluation \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --model_type "meta-llama/Meta-Llama-3-8B-Instruct" \
    --save_model_name "tampered_base_model" \
    --training_strategy "attack_target_forget" \
    --optimizer_type "sgd_with_momentum" \
    --learning_rate 2e-5 \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 5 \
    --max_examples 500 \
    --evaluate \
    --wandb
