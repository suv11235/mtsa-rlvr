#!/bin/bash
#SBATCH --job-name=mtsa-rerun-tampering
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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

# Function to run tampering eval
run_eval() {
    local NAME=$1
    local BASE=$2
    local ADAPTER=$3
    local EXTRA_ARGS=$4

    echo ">>> Starting Tampering Eval (50 steps) for: $NAME"
    cd $TAR_REPO_DIR
    
    # Base command
    # We use attack_target_forget to use the specific harmful dataset
    local CMD="accelerate launch --config_file configs/accel_config_1_gpu.yaml \
        -m red_teaming.red_teaming_evaluation \
        --model_name \"$BASE\" \
        --model_type \"$BASE\" \
        --save_model_name \"tampered_$NAME\" \
        --training_strategy \"attack_target_forget\" \
        --learning_rate 2e-5 \
        --num_epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_steps 50 \
        --max_examples 1000 \
        --load_in_4bit \
        --wandb"

    # Add adapter if specified
    if [ -n "$ADAPTER" ]; then
        # The modified script will load this adapter as trainable
        # No need for --peft because args.peft is disabled internally if adapter is provided
        CMD="$CMD --adapter_path \"$MTSA_DIR/$ADAPTER\""
    else
        # Baseline: uses --peft to add new random init LoRA
        CMD="$CMD --peft"
    fi
    
    CMD="$CMD $EXTRA_ARGS"

    echo "Executing: $CMD"
    eval $CMD
    
    cd $MTSA_DIR
}

# Distribute across GPUs for speed
# GPU 0: Baseline Qwen
echo ">>> Launching Baseline Qwen on GPU 0"
CUDA_VISIBLE_DEVICES=0 run_eval "baseline_qwen" "Qwen/Qwen2.5-7B-Instruct" "" "" &

# GPU 1: Extreme v2 (Our new best run attempt)
echo ">>> Launching Extreme v2 on GPU 1"
CUDA_VISIBLE_DEVICES=1 run_eval "extreme_v2_step25" "meta-llama/Meta-Llama-3-8B-Instruct" "outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-25" "" &

# GPU 2: Extreme v1 (Prior best)
echo ">>> Launching Extreme v1 on GPU 2"
CUDA_VISIBLE_DEVICES=2 run_eval "extreme_v1_step75" "meta-llama/Meta-Llama-3-8B-Instruct" "outputs/rlvr_mtsa_extreme_tar/checkpoint-75" "" &

# GPU 3: Vanilla TAR Baseline (Paper faithful)
echo ">>> Launching Vanilla TAR on GPU 3"
CUDA_VISIBLE_DEVICES=3 run_eval "vanilla_tar_step50" "meta-llama/Meta-Llama-3-8B-Instruct" "outputs/tar_vanilla_baseline/checkpoint-50" "" &

wait
echo ">>> ALL TAMPERING JOBS COMPLETE <<<"
