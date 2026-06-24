#!/bin/bash
# Multi-GPU Launch Script for Vast.ai (MTSA-RLVR)
# This script handles multi-GPU sharding (split_ranks) and optimized memory management.

set -e

# Path Resolution
if [ -d "/workspace/mtsa-rlvr/MTSA" ]; then
    cd /workspace/mtsa-rlvr/MTSA
fi

echo "Working directory: $(pwd)"

# 1. Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment 'venv' not found. Please run 'bash script/deploy/setup_vastai.sh' first."
    exit 1
fi

# 2. Environment Configuration
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_HOST_IP=127.0.0.1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Force 2-hour timeout via environment for distributed procs
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ENABLE_MONITORING=0

# Redirect large caches to /workspace
export HF_HOME="/workspace/.cache/huggingface"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
mkdir -p $HF_HOME $TRITON_CACHE_DIR

# 3. Detect Resources
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs..."

# 4. Process Cleanup
echo "Cleaning up lingering processes..."
# Avoid killing unrelated python processes (e.g., host/control plane).
# Only clean known MTSA/vLLM workers unless caller explicitly opts out.
if [ "${SKIP_PROCESS_CLEANUP:-0}" != "1" ]; then
    pkill -9 -f "src/algorithm/mt_rlvr_train.py" || true
    pkill -9 -f "accelerate.*mt_rlvr_train.py" || true
    pkill -9 -f "vllm" || true
fi
sleep 5

# 5. Training Parameters
CONFIG_FILE="script/accelerate_configs/zero2.yaml"
OUTPUT_DIR="/workspace/outputs/llama3-8b-bio-lora"
mkdir -p $OUTPUT_DIR

echo "Launching RLVR Training (ZeRO-2 + split_ranks)..."

# Note: We use --main_process_port 29509 to avoid any standard collisions
accelerate launch \
    --config_file $CONFIG_FILE \
    --num_processes $NUM_GPUS \
    --main_process_port 29509 \
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
    --vllm_distribution_strategy split_ranks \
    --save_steps 50 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --gradient_checkpointing True \
    "$@"
