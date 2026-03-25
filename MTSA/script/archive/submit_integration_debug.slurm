#!/bin/bash
#SBATCH --job-name=integration-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --output=MTSA/logs/%x-%j.out
#SBATCH --error=MTSA/logs/%x-%j.err

# Activation
cd ~/mtsa-rlvr/MTSA/MTSA
source ../venv/bin/activate

# Environment
export HF_HOME=/data/suvajit_majumder/huggingface_cache
export VLLM_USE_V1="0"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_MODE="disabled" # Disable wandb for debug
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Explicit 1+1 GPU Setup
export CUDA_VISIBLE_DEVICES=0,1
export AUX_GPU="cuda:1"

# Launch with Accelerate (2 GPU) - Running just 2 steps to verify flow
echo ">>> STARTING INTEGRATION DEBUG RUN"
accelerate launch --config_file script/accelerate_configs/zero2_2gpu.yaml \
    -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "cais/HarmBench-Llama-2-13b-cls" \
    --judge_type "harmbench" \
    --dataset_name "datasets/attack_target/tiny_8_goals.json" \
    --output_dir "outputs/debug_integration" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --bf16 True \
    --num_train_epochs 1 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mini_batch_size 1 \
    --num_rollouts 2 \
    --learning_rate 1e-6 \
    --max_response_length 128 \
    --use_entropy_reward True \
    --entropy_reward_weight 0.0 \
    --judge_reward_weight 0.0 \
    --max_sim_turns 2 \
    --tar_inner_loop_steps 1 \
    --tar_type "attack" \
    --tar_inner_lr 2e-5 \
    --use_tamper_resistance True \
    --save_steps 0 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache \
    --gradient_checkpointing True \
    --use_vllm False

if [ $? -eq 0 ]; then
    echo ">>> INTEGRATION TEST PASSED"
else
    echo ">>> INTEGRATION TEST FAILED"
    exit 1
fi
