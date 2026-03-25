#!/bin/bash
#SBATCH --job-name=mtsa-comprehensive-tampering
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

ROOT_DIR=~/mtsa-rlvr
MTSA_DIR=$ROOT_DIR/MTSA
TAR_REPO_DIR=$ROOT_DIR/tamper-resistance-repo

cd $MTSA_DIR
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd):$TAR_REPO_DIR
export $(grep -v '^#' .env | xargs)

# Function to run tampering eval
run_eval() {
    local NAME=$1
    local BASE=$2
    local ADAPTER=$3
    local EXTRA_ARGS=$4

    echo ">>> Running Tampering Eval for: $NAME"
    cd $TAR_REPO_DIR
    
    # Base command
    local CMD="accelerate launch --config_file configs/accel_config_2_gpu.yaml \
        -m red_teaming.red_teaming_evaluation \
        --model_name \"$BASE\" \
        --model_type \"$BASE\" \
        --save_model_name \"tampered_$NAME\" \
        --training_strategy \"pure_pile_bio_forget\" \
        --learning_rate 2e-5 \
        --num_epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps 2 \
        --max_steps 50 \
        --max_examples 1000 \
        --wandb"

    # Add adapter or peft flags
    if [ -n "$ADAPTER" ]; then
        CMD="$CMD --adapter_path \"$MTSA_DIR/$ADAPTER\""
    else
        CMD="$CMD --peft"
    fi
    
    # Add extra args
    CMD="$CMD $EXTRA_ARGS"

    echo "Executing: $CMD"
    eval $CMD
    
    cd $MTSA_DIR
}

# 1. Baseline: Qwen with random init LoRA
run_eval "baseline_qwen" "Qwen/Qwen2.5-7B-Instruct" "" ""

# 2. Checkpoint-6: Llama based
CHECKPOINT_6="./outputs/rlvr_mtsa_test/checkpoint-6"
if [ -d "$CHECKPOINT_6" ]; then
    run_eval "checkpoint_6" "meta-llama/Meta-Llama-3-8B-Instruct" "$CHECKPOINT_6" ""
else
    echo "WARNING: $CHECKPOINT_6 not found, skipping."
fi

# 3. Checkpoint-50: Llama based
CHECKPOINT_50="./outputs/rlvr_mtsa_long_run/checkpoint-50"
if [ -d "$CHECKPOINT_50" ]; then
    run_eval "checkpoint_50" "meta-llama/Meta-Llama-3-8B-Instruct" "$CHECKPOINT_50" ""
else
    echo "WARNING: $CHECKPOINT_50 not found, skipping."
fi

echo ">>> Comprehensive Tampering Evaluation Submitted/Complete!"
