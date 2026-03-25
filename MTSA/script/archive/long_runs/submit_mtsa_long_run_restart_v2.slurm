#!/bin/bash
#SBATCH --job-name=mtsa-long-run-v4gpu
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:4               # Scale up to 4 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00            # Stay within partition limit (24h)
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

echo "Starting Long-Run MTSA Training (4-GPU Scale) on $(hostname)"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "Restarting from base Llama-3.1-8B-Instruct"

# Run the training script with production hyperparameters
# Restarting from scratch on 4 GPUs
python -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --dataset_name "datasets/attack_target/train_attack_target_labels.json" \
    --output_dir "./outputs/rlvr_mtsa_long_run_v4gpu" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --per_device_train_batch_size 8 \
    --mini_batch_size 4 \
    --num_rollouts 8 \
    --max_steps 1500 \
    --learning_rate 7e-6 \
    --use_tamper_resistance True \
    --tar_inner_loop_steps 1 \
    --tar_inner_lr 5e-5 \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.2 \
    --max_sim_turns 3 \
    --gradient_checkpointing True \
    --attn_implementation "sdpa" \
    --save_steps 10 \
    --logging_steps 1 \
    --cache_dir "/data/suvajit_majumder/huggingface_cache"
