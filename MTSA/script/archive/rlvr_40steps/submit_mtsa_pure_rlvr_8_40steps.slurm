#!/bin/bash
#SBATCH --job-name=pure-rlvr-8-40steps
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
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

# WandB Configuration
export WANDB_PROJECT="mtsa-rlvr-training"
export WANDB_NAME="pure-rlvr-8-40steps-warmup10-noentropy"

# Create output dir
OUTPUT_DIR="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/rlvr_mtsa_pure_rlvr_8_goals_40steps"
mkdir -p $OUTPUT_DIR

python src/algorithm/mt_rlvr_train.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --dataset_name datasets/attack_target/unique_8_goals.json \
    --output_dir $OUTPUT_DIR \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --per_device_train_batch_size 4 \
    --mini_batch_size 2 \
    --num_rollouts 4 \
    --num_train_epochs 20 \
    --warmup_steps 10 \
    --learning_rate 1e-5 \
    --use_tamper_resistance False \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.0 \
    --max_sim_turns 1 \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --save_steps 10 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache
