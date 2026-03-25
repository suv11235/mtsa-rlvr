#!/bin/bash
# RLVR Defence Training Script
# Train target model using entropy-based defense

# Usage: bash script/run_rlvr_defence.sh <model_path> <dataset_path> [output_dir]

MODEL_PATH=${1:-"Qwen/Qwen2.5-7B-Instruct"}
DATASET_PATH=${2:-"datasets/attack_target/train_attack_target.json"}
OUTPUT_DIR=${3:-"./outputs/rlvr_defence"}
ATTACKER_MODEL_PATH=${4:-""}

# Environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export WANDB_PROJECT="MTSA-RLVR-Defence"
export HF_HOME=/workspace/huggingface_cache
export OUTPUT_DIR=$OUTPUT_DIR

echo "======================================"
echo "RLVR Defence Training"
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
    --save_steps 100 \
    --logging_steps 10 \
    --load_in_4bit true \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --defence_mode true \
    --use_entropy_reward true \
    --attacker_model_name_or_path "$ATTACKER_MODEL_PATH" \
    --attn_implementation sdpa \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training complete! Output saved to $OUTPUT_DIR"
