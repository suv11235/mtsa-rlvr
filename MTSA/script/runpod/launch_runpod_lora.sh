#!/bin/bash
# Multi-GPU Launch Script for RunPod (MTSA-RLVR)
# This script auto-detects GPU count and launches distributed training using Accelerate.

set -e

# Default to /workspace/mtsa-rlvr/MTSA if it exists (RunPod standard)
if [ -d "/workspace/mtsa-rlvr/MTSA" ]; then
    cd /workspace/mtsa-rlvr/MTSA
elif [ -d "$(pwd)/MTSA" ]; then
    cd MTSA
fi

echo "Working directory: $(pwd)"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment 'venv' not found. Please run 'bash script/deploy/setup_runpod.sh' first."
    exit 1
fi

# Environment Setup
if [ -f ".env" ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Optimized NCCL settings for multi-GPU
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export NCCL_TIMEOUT=900

# Redirect large caches to /workspace to avoid filling up the small root partition
export HF_HOME="/workspace/.cache/huggingface"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export PYTHONCACHEPREFIX="/workspace/.cache/python"
mkdir -p $HF_HOME $TRITON_CACHE_DIR $PYTHONCACHEPREFIX

# Detect GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs..."

# Determine config file based on GPU count or default
CONFIG_FILE="script/accelerate_configs/zero2.yaml"
if [ "$NUM_GPUS" -eq 2 ] && [ -f "script/accelerate_configs/zero2_2gpu.yaml" ]; then
    CONFIG_FILE="script/accelerate_configs/zero2_2gpu.yaml"
elif [ "$NUM_GPUS" -eq 4 ] && [ -f "script/accelerate_configs/zero2_4gpu.yaml" ]; then
    CONFIG_FILE="script/accelerate_configs/zero2_4gpu.yaml"
fi

echo "Using config: $CONFIG_FILE"

# Cleanup hung processes
pkill -9 -f vllm || true
pkill -9 -f torch || true
sleep 2

# Training Parameters
OUTPUT_DIR="/workspace/outputs/llama3-8b-bio-lora"
mkdir -p $OUTPUT_DIR

echo "Starting training on $NUM_GPUS GPUs..."

accelerate launch \
    --config_file $CONFIG_FILE \
    --num_processes $NUM_GPUS \
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
