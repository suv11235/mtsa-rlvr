#!/bin/bash
#SBATCH --job-name=extreme-tar-bio
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# WandB Configuration
export WANDB_PROJECT="mtsa-rlvr-training"
export WANDB_NAME="extreme-tar-bio-64steps-subset"

# Create output dir
OUTPUT_DIR="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/extreme_tar_bio_64steps"
mkdir -p $OUTPUT_DIR

# Using reduced batch sizes to avoid OOM with 64 inner steps
# per_device_train_batch_size=2, mini_batch_size=1
# Dataset: biosecurity_goals.json (21 high-quality bio-threat goals)

python src/algorithm/mt_rlvr_train.py \
    --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
    --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --dataset_name datasets/attack_target/biosecurity_goals.json \
    --output_dir $OUTPUT_DIR \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --per_device_train_batch_size 2 \
    --mini_batch_size 1 \
    --num_rollouts 4 \
    --num_train_epochs 50 \
    --warmup_steps 10 \
    --learning_rate 1e-5 \
    --use_tamper_resistance True \
    --tar_type "scramble" \
    --tar_inner_loop_steps 64 \
    --tar_inner_lr 5e-5 \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.0 \
    --max_sim_turns 1 \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --save_steps 20 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache
