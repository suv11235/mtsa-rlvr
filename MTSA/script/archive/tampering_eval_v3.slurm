#!/bin/bash
#SBATCH --job-name=tampering-rerun-50steps
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# Harmful dataset path
HARMFUL_DATA="./datasets/attack_target/train_attack_target_labels.json"

run_tamper() {
    local NAME=$1
    local PATH=$2
    local GPU_ID=$3
    local EXTRA=$4

    VENV_PYTHON="./venv/bin/python3"
    echo ">>> Starting Tampering for $NAME on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID $VENV_PYTHON src/algorithm/red_team_sft.py \
        --model_name_or_path "$PATH" \
        --dataset_name "$HARMFUL_DATA" \
        --output_dir "./outputs/tamper_eval_50step/$NAME" \
        --max_steps 50 \
        --learning_rate 2e-5 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --load_in_4bit True \
        --use_peft True \
        --bf16 True \
        --logging_steps 5 \
        --save_strategy "no" \
        $EXTRA
}

# 1. Baseline Qwen
run_tamper "baseline_qwen" "Qwen/Qwen2.5-7B-Instruct" 0 "" &

# 2. Extreme v2 (Checkpoint 25)
run_tamper "extreme_v2_step25" "outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-25" 1 "" &

# 3. Extreme v1 (Checkpoint 75)
run_tamper "extreme_v1_step75" "outputs/rlvr_mtsa_extreme_tar/checkpoint-75" 2 "" &

# 4. Vanilla TAR Baseline (Checkpoint 50)
run_tamper "vanilla_tar_step50" "outputs/tar_vanilla_baseline/checkpoint-50" 3 "" &

wait
echo ">>> ALL TAMPERING JOBS COMPLETE <<<"
