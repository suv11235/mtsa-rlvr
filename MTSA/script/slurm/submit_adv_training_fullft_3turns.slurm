#!/bin/bash
#SBATCH --job-name=adv-training-fullft-3turn
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate

# Environment
export VLLM_USE_V1="0"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# WandB Configuration
export WANDB_PROJECT="mtsa-rlvr-training"
export WANDB_NAME="adv_training_fullft_3turns_sr_bio_v0"

# Create output dir
OUTPUT_DIR="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/adv_training_fullft_3turns_sr"
mkdir -p $OUTPUT_DIR

# Full Fine-Tuning (3-Turn Adversarial Hardening)
accelerate launch --config_file script/accelerate_configs/zero3_4gpu.yaml \
    --num_processes 4 \
    -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
    --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "qylu4156/strongreject-15k-v1" \
    --judge_type "strongreject" \
    --dataset_name datasets/attack_target/biosecurity_goals.json \
    --output_dir $OUTPUT_DIR \
    --defence_mode True \
    --use_peft False \
    --load_in_4bit False \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_rollouts 4 \
    --num_train_epochs 5 \
    --warmup_steps 5 \
    --learning_rate 1e-5 \
    --use_tamper_resistance False \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.0 \
    --max_sim_turns 3 \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --save_steps 20 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache \
    --use_vllm False
