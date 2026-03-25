#!/bin/bash
#SBATCH --job-name=mtsa-tamper-final
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

CONFIG_FILE="$TAR_REPO_DIR/configs/accel_config_1_gpu.yaml"

run_eval() {
    local NAME=$1
    local BASE=$2
    local ADAPTER=$3
    local EXTRA_ARGS=$4

    echo "=========================================="
    echo ">>> TAMPERING EVAL: $NAME"
    echo "=========================================="
    
    cd $TAR_REPO_DIR
    
    # Base command
    local CMD="accelerate launch --config_file $CONFIG_FILE \
        -m red_teaming.red_teaming_evaluation \
        --model_name \"$BASE\" \
        --model_type \"$BASE\" \
        --save_model_name \"tampered_v3_$NAME\" \
        --training_strategy \"attack_target_forget\" \
        --learning_rate 2e-5 \
        --num_epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_steps 50 \
        --load_in_4bit \
        --wandb"

    if [ -n "$ADAPTER" ]; then
        CMD="$CMD --adapter_path \"$MTSA_DIR/$ADAPTER\""
    else
        CMD="$CMD --peft"
    fi
    
    CMD="$CMD $EXTRA_ARGS"

    echo "Executing: $CMD"
    eval $CMD
    
    cd $MTSA_DIR
}

# 1. Baseline Qwen
run_eval "baseline_qwen" "Qwen/Qwen2.5-7B-Instruct" "" ""

# 2. Extreme v2 (Our new attempt)
run_eval "extreme_v2_step25" "meta-llama/Meta-Llama-3-8B-Instruct" "outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-25" ""
run_eval "extreme_v2_step50" "meta-llama/Meta-Llama-3-8B-Instruct" "outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-50" ""

# 3. Extreme v1 (Prior best)
run_eval "extreme_v1_step75" "meta-llama/Meta-Llama-3-8B-Instruct" "outputs/rlvr_mtsa_extreme_tar/checkpoint-75" ""

# 4. Vanilla TAR (Paper baseline)
run_eval "vanilla_tar_step50" "meta-llama/Meta-Llama-3-8B-Instruct" "outputs/tar_vanilla_baseline/checkpoint-50" ""

echo ">>> COMPILING RESULTS <<<"
# We'll need to manually check logs or W&B for the loss values.
