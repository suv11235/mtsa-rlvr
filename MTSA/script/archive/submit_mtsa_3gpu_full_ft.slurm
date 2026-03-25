#!/bin/bash
#SBATCH --job-name=mtsa-3gpu-full-ft-zeros2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Activation
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate

# Environment
export HF_HOME=/data/suvajit_majumder/huggingface_cache
export VLLM_USE_V1="0"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_PROJECT="mtsa-full-ft-zero2"
export WANDB_NAME="full-ft-3gpu-zero2-hard-examples"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_TIMEOUT=7200 # 2 Hour timeout for collective ops

# Full 4-GPU Distributed Setup (ZeRO-3)
# In Full FT, we MUST use all GPUs for sharding to fit the model and optimizer states.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch with Accelerate (All 4 GPUs for sharding)
accelerate launch --config_file script/accelerate_configs/zero3_4gpu.yaml \
    --num_processes 4 \
    -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "meta-llama/Llama-Guard-3-8B" \
    --dataset_name "datasets/attack_target/train_attack_target.json" \
    --output_dir "checkpoints/mtsa-full-ft-zero3-allranks" \
    --defence_mode True \
    --use_peft False \
    --load_in_4bit False \
    --bf16 True \
    --kl_coef 0.001 \
    --use_kl_in_reward True \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_rollouts 4 \
    --learning_rate 5e-6 \
    --max_response_length 512 \
    --use_entropy_reward True \
    --max_sim_turns 3 \
    --use_tamper_resistance True \
    --save_steps 10 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache \
    --gradient_checkpointing True \
    --use_vllm True
