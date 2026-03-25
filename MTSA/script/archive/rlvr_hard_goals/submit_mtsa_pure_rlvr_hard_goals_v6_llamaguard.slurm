#!/bin/bash
#SBATCH --job-name=pure-rlvr-hard-goals-v6-lg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=MTSA/logs/%x-%j.out
#SBATCH --error=MTSA/logs/%x-%j.err

# Activation
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate

# Environment
export HF_HOME=/data/suvajit_majumder/huggingface_cache
export VLLM_USE_V1="0"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_PROJECT="mtsa-pure-rlvr-new"
export WANDB_NAME="pure-rlvr-hard-goals-3turns-v6-llamaguard"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Training
python src/algorithm/mt_rlvr_train.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "meta-llama/Llama-Guard-3-8B" \
    --judge_type "llamaguard" \
    --dataset_name "datasets/attack_target/hard_goals.json" \
    --output_dir "checkpoints/pure-rlvr-hard-goals-v6-llamaguard" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --num_train_epochs 40 \
    --per_device_train_batch_size 8 \
    --mini_batch_size 4 \
    --ppo_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --num_rollouts 8 \
    --max_response_length 256 \
    --use_entropy_reward False \
    --max_sim_turns 3 \
    --use_kl_in_reward True \
    --kl_coef 0.001 \
    --save_steps 20 \
    --logging_steps 1 \
    --use_vllm True \
    --cache_dir /data/suvajit_majumder/huggingface_cache \
    --gradient_checkpointing True
