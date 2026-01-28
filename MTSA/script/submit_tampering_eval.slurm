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

# 1. Prepare Merged Model
CHECKPOINT_PATH="./outputs/rlvr_mtsa_long_run/checkpoint-50" # Update this to your latest
if [ ! -d "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH="./outputs/rlvr_mtsa_test/checkpoint-6"
fi

MERGED_MODEL_DIR="./outputs/merged_for_tampering"
echo ">>> Preparing merged model for evaluation from $CHECKPOINT_PATH..."
python script/prepare_for_tampering_eval.py \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter_path "$CHECKPOINT_PATH" \
    --output_path "$MERGED_MODEL_DIR"

# 2. Run Tampering Evaluation (SFT Attack)
echo ">>> Starting Tampering Evaluation (Red-teaming SFT)..."
# We move to the tamper-resistance-repo/red_teaming directory to avoid import issues
cd $TAR_REPO_DIR/red_teaming

# Using accelerate launch as recommended in the repository
accelerate launch --config_file ../configs/accel_config_2_gpu.yaml \
    red_teaming_evaluation.py \
    --model_name "$MTSA_DIR/$MERGED_MODEL_DIR" \
    --model_type "meta-llama/Llama-3.1-8B-Instruct" \
    --save_model_name "tampered_defense_model" \
    --training_strategy "pure_pile_bio_forget" \
    --learning_rate 2e-5 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --wandb
