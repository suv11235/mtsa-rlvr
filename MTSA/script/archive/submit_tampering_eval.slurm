#!/bin/bash
#SBATCH --job-name=mtsa-tampering-eval
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Use 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00            # Should be enough for 1 epoch of SFT
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

# 1. Prepare Merged Model (SKIPPED - mimic training LoRA-only tampering)
CHECKPOINT_PATH="./outputs/rlvr_mtsa_long_run/checkpoint-50" # Update this to your latest
if [ ! -d "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH="./outputs/rlvr_mtsa_test/checkpoint-6"
fi

# MERGED_MODEL_DIR="./outputs/merged_for_tampering"
# echo ">>> Preparing merged model for evaluation from $CHECKPOINT_PATH..."
# python script/prepare_for_tampering_eval.py \
#     --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
#     --adapter_path "$CHECKPOINT_PATH" \
#     --output_path "$MERGED_MODEL_DIR"

# 2. Run Tampering Evaluation (SFT Attack on LoRA weights)
echo ">>> Starting Tampering Evaluation (LoRA-only SFT)..."
# We run as a module from the repo root to handle relative imports correctly
cd $TAR_REPO_DIR

accelerate launch --config_file configs/accel_config_2_gpu.yaml \
    -m red_teaming.red_teaming_evaluation \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --model_type "meta-llama/Meta-Llama-3-8B-Instruct" \
    --adapter_path "$MTSA_DIR/$CHECKPOINT_PATH" \
    --save_model_name "tampered_defense_lora" \
    --training_strategy "pure_pile_bio_forget" \
    --learning_rate 2e-5 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 5 \
    --max_examples 1000 \
    --wandb
