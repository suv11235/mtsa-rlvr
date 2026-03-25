#!/bin/bash
#SBATCH --job-name=adv-grpo-training
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr
source MTSA/venv/bin/activate

# Environment
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:$(pwd)/adv_grpo

mkdir -p adv_grpo/outputs
mkdir -p logs

# GRPO Training (Clean implementation)
accelerate launch \
    --config_file MTSA/script/accelerate_configs/zero2.yaml \
    --num_processes 2 \
    adv_grpo/main.py \
    --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "qylu4156/strongreject-15k-v1" \
    --dataset_path MTSA/datasets/attack_target/biosecurity_goals.json \
    --train_target "victim" \
    --output_dir adv_grpo/outputs \
    --vllm_gpu_memory_utilization 0.35 \
    --num_rollouts 4 \
    --num_epochs 1 \
    --learning_rate 1e-6
