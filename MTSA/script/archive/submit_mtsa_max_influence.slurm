#!/bin/bash
#SBATCH --job-name=mtsa-max-influence-8goals
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# Performance optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export WANDB_PROJECT="mtsa-rlvr-defense"

echo "Starting MAX INFLUENCE RLVR Run (8 Goals, 10 Epochs, 8 Inner Steps)"

python -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --dataset_name "datasets/attack_target/tiny_8_goals.json" \
    --output_dir "./outputs/rlvr_mtsa_max_influence_8goals" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --per_device_train_batch_size 4 \
    --mini_batch_size 2 \
    --num_rollouts 4 \
    --max_steps 10 \
    --learning_rate 5e-5 \
    --use_tamper_resistance True \
    --tar_inner_loop_steps 8 \
    --tar_inner_lr 5e-5 \
    --tar_type "attack" \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.2 \
    --max_sim_turns 3 \
    --gradient_checkpointing True \
    --attn_implementation "sdpa" \
    --save_steps 5 \
    --logging_steps 1 \
    --cache_dir "/data/suvajit_majumder/huggingface_cache"
