#!/bin/bash
#SBATCH --job-name=tamper-eval-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --output=MTSA/logs/%x-%j.out
#SBATCH --error=MTSA/logs/%x-%j.err

# Activation
cd ~/mtsa-rlvr/MTSA/MTSA
source ../venv/bin/activate

# Environment
export HF_HOME=/data/suvajit_majumder/huggingface_cache
export VLLM_USE_V1="0"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_PROJECT="mtsa-tamper-eval-experiments"
export WANDB_NAME="tamper-eval-base-40steps"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Explicit 1+1 GPU Setup
export CUDA_VISIBLE_DEVICES=0,1
export AUX_GPU="cuda:1"  # Pin Attacker/Judge to GPU 1

# Launch with Accelerate + DeepSpeed
# num_processes: 2 (1 trainer + 1 auxiliary)
accelerate launch --config_file script/accelerate_configs/zero2_2gpu.yaml \
    -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "cais/HarmBench-Llama-2-13b-cls" \
    --judge_type "harmbench" \
    --dataset_name "datasets/attack_target/tiny_8_goals.json" \
    --output_dir "outputs/tamper_eval_base" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --bf16 True \
    --num_train_epochs 1 \
    --kl_coef 0.001 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mini_batch_size 1 \
    --num_rollouts 4 \
    --learning_rate 0.0 \
    --max_response_length 512 \
    --use_entropy_reward True \
    --entropy_reward_weight 0.0 \
    --judge_reward_weight 0.0 \
    --max_sim_turns 3 \
    --tar_inner_loop_steps 40 \
    --tar_type "attack" \
    --tar_inner_lr 2e-5 \
    --use_tamper_resistance True \
    --save_steps 50 \
    --logging_steps 1 \
    --cache_dir /data/suvajit_majumder/huggingface_cache \
    --gradient_checkpointing True \
    --use_vllm False
