#!/bin/bash
#SBATCH --job-name=mtsa-long-run
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Use 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00            # Allow 24 hours
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set Environment Variables from your .env
export $(grep -v '^#' .env | xargs)

# Performance optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export WANDB_PROJECT="mtsa-rlvr-defense"

echo "Starting Long-Run MTSA Training on $(hostname)"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

# Run the training script with production hyperparameters
python -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --dataset_name "datasets/attack_target/train_attack_target_labels.json" \
    --output_dir "./outputs/rlvr_mtsa_long_run" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --mini_batch_size 2 \
    --num_rollouts 8 \
    --max_steps 1500 \
    --learning_rate 7e-6 \
    --use_tamper_resistance True \
    --tar_inner_loop_steps 1 \
    --tar_inner_lr 5e-5 \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.5 \
    --max_sim_turns 3 \
    --gradient_checkpointing True \
    --attn_implementation "sdpa" \
    --save_steps 50 \
    --cache_dir "/data/suvajit_majumder/huggingface_cache"
