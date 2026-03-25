#!/bin/bash
#SBATCH --job-name=adv-training-lora
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# Environment
export VLLM_USE_V1="0"
export VLLM_ENABLE_V1="0"
export VLLM_ENABLE_V1_MULTIPROCESSING="0"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export NCCL_SOCKET_TIMEOUT=7200000

# WandB Configuration
export WANDB_PROJECT="mtsa-rlvr-training"
export WANDB_NAME="adv_training_lora_5epochs_sr_bio_subset_5e-5_v0"

# Create output dir
OUTPUT_DIR="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/adv_training_lora_bio_subset_sr"
mkdir -p $OUTPUT_DIR

# LoRA Fine-Tuning configuration (No quantization, Full precision BF16)
accelerate launch \
    --config_file script/accelerate_configs/zero2_4gpus.yaml \
    --num_processes 4 \
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
    --per_device_train_batch_size 2 \
    --mini_batch_size 1 \
    --num_rollouts 4 \
    --num_train_epochs 5 \
    --warmup_steps 5 \
    --learning_rate 5e-5 \
    --use_tamper_resistance False \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 0.0 \
    --max_sim_turns 1 \
    --gradient_checkpointing True \
    --attn_implementation sdpa \
    --save_steps 20 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.65 \
    --vllm_distribution_strategy "all_ranks"
