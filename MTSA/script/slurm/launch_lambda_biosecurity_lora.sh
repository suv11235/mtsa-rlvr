#!/bin/bash
export VLLM_USE_V1=0
export VLLM_ENABLE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Launch script for LoRA fine-tuning on 2x H100 Lambda instance

set -e

# Use absolute path based on Lambda setup
cd ~/mtsa-rlvr/MTSA

# Check if venv exists, if not, user should have run setup_runpod.sh
if [ ! -d "venv" ]; then
    echo "Error: venv not found. Please run 'bash script/deploy/setup_runpod.sh' first."
    exit 1
fi

source venv/bin/activate

# Environment
export $(grep -v '^#' .env | xargs)
export VLLM_USE_V1=0
export VLLM_ENABLE_V1=0
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Prevent false-positive PG ID NCCL timeouts
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export NCCL_TIMEOUT=900
export NCCL_SOCKET_TIMEOUT=900000

# Cleanup potentially hung processes
pkill -9 -f vllm || true
pkill -9 -f torch || true
pkill -9 -f multiprocessing.spawn || true
pkill -9 -f python3 || true
sleep 5

# WandB Configuration
export WANDB_PROJECT="mtsa-rlvr-lora"
export WANDB_NAME="llama3-8b-bio-lora-5e-5"
export WANDB_MODE="offline"

OUTPUT_DIR="outputs/llama3-8b-bio-lora"
mkdir -p $OUTPUT_DIR

echo "Starting training on 2x H100..."
echo "Victim: lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2"
echo "Attacker: suv11235/red_team_model_SFT_mtsa"
echo "Judge: strongreject"

# 8x A100 Setup:
# Parallel rollout generation across 8 GPUs using split_ranks
# num_processes 8 with ZeRO-2 for efficient LoRA training

accelerate launch \
    --config_file script/accelerate_configs/zero2.yaml \
    --num_processes 8 \
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
    --bf16 True \
    --learning_rate 1e-4 \
    --num_rollouts 16 \
    --max_sim_turns 3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --mini_batch_size 1 \
    --ppo_epochs 1 \
    --max_response_length 512 \
    --use_vllm True \
    --vllm_gpu_memory_utilization 0.65 \
    --save_steps 50 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --gradient_checkpointing True \
    "$@"
