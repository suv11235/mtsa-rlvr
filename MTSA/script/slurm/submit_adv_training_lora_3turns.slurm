#!/bin/bash
#SBATCH --job-name=adv-training-lora-3turn
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate

# Environment
export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=0 # Remove restriction for multi-gpu
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# WandB Configuration
export WANDB_PROJECT="mtsa-rlvr-training"
export WANDB_NAME="adv_training_lora_3turn_zero2"

# Create output dir
OUTPUT_DIR="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/adv_training_lora_3turn"
mkdir -p $OUTPUT_DIR

# LoRA Fine-Tuning (3-Turn Adversarial Hardening)
# Using accelerate launch with ZeRO-2 for efficient multi-GPU training
accelerate launch \
    --config_file script/accelerate_configs/zero2.yaml \
    --num_processes 2 \
    src/algorithm/mt_rlvr_train.py \
    --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
    --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "qylu4156/strongreject-15k-v1" \
    --judge_type "strongreject" \
    --dataset_name datasets/attack_target/biosecurity_goals.json \
    --output_dir $OUTPUT_DIR \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit False \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --mini_batch_size 1 \
    --num_rollouts 1 \
    --num_train_epochs 1 \
    --warmup_steps 0 \
    --learning_rate 1e-6 \
    --ppo_epochs 1 \
    --max_response_length 512 \
    --use_tamper_resistance False \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.0 \
    --max_sim_turns 3 \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --save_steps 10 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache \
    --use_vllm True
