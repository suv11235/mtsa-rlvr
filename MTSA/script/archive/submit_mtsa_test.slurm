#!/bin/bash
#SBATCH --job-name=mtsa-test
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Request 2 GPUs
#SBATCH --cpus-per-task=4          # 2 CPUs per GPU
#SBATCH --mem=64G                 
#SBATCH --time=01:00:00            # 1 hour for testing
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate

# Set Environment Variables from your .env
export $(grep -v '^#' .env | xargs)

# Performance optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled # Disable wandb for short test

echo "Starting Multi-GPU MTSA TEST on $(hostname)"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

# Run the training script with SMALL dataset and LOW max_steps
python -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --dataset_name "datasets/attack_target/train_attack_target_small.json" \
    --output_dir "./outputs/rlvr_mtsa_test" \
    --tar_type "attack" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --mini_batch_size 1 \
    --num_rollouts 2 \
    --max_steps 5 \
    --learning_rate 5e-6 \
    --use_tamper_resistance True \
    --tar_inner_loop_steps 1 \
    --tar_inner_lr 5e-5 \
    --gradient_checkpointing True \
    --attn_implementation "sdpa" \
    --cache_dir "/data/suvajit_majumder/huggingface_cache"
