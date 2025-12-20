#!/bin/bash
# RLVR Attack Training Script
# Train red-team model using GRPO/RLOO policy gradient

# Usage: bash script/run_rlvr_attack.sh <model_path> <dataset_path> [output_dir]

MODEL_PATH=${1:-"Qwen/Qwen2.5-7B-Instruct"}
DATASET_PATH=${2:-"datasets/attack_target/train_attack_target.json"}
OUTPUT_DIR=${3:-"./outputs/rlvr_attack"}

# Environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export WANDB_PROJECT="MTSA-RLVR-Attack"

echo "======================================"
echo "RLVR Attack Training"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "======================================"

mkdir -p $OUTPUT_DIR

python -m src.algorithm.mt_rlvr_train \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --adv_estimator grpo \
    --use_kl_in_reward true \
    --kl_coef 0.001 \
    --num_rollouts 4 \
    --max_prompt_length 320 \
    --max_response_length 1024 \
    --ppo_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --save_freq 100 \
    --logging_steps 10 \
    --defence_mode false \
    --use_entropy_reward false \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training complete! Output saved to $OUTPUT_DIR"
