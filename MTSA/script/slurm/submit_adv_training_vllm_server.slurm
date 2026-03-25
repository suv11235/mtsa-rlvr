#!/bin/bash
#SBATCH --job-name=adv-training-server
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
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
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export NCCL_SOCKET_TIMEOUT=7200000

# Server configurations
VICTIM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ATTACKER_MODEL="suv11235/red_team_model_SFT_mtsa"
JUDGE_MODEL="qylu4156/strongreject-15k-v1"

# Launch vLLM servers on separate GPUs (e.g. GPUs 5, 6, 7)
export CUDA_VISIBLE_DEVICES=5
python -m vllm.entrypoints.openai.api_server --model $VICTIM_MODEL --port 8001 --gpu-memory-utilization 0.9 --max-model-len 4096 &
VICTIM_PID=$!

export CUDA_VISIBLE_DEVICES=6
python -m vllm.entrypoints.openai.api_server --model $ATTACKER_MODEL --port 8002 --gpu-memory-utilization 0.9 --max-model-len 4096 &
ATTACKER_PID=$!

export CUDA_VISIBLE_DEVICES=7
python -m vllm.entrypoints.openai.api_server --model $JUDGE_MODEL --port 8003 --gpu-memory-utilization 0.9 --max-model-len 4096 &
JUDGE_PID=$!

# Wait for servers to be ready...
sleep 60

# Now launch training on remaining GPUs (GPUs 0,1,2,3,4)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
accelerate launch \
    --config_file script/accelerate_configs/zero2_5gpus.yaml \
    --num_processes 5 \
    src/algorithm/mt_rlvr_train.py \
    --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
    --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "http://localhost:8002/v1" \
    --judge_model_name_or_path "http://localhost:8003/v1" \
    --victim_model_name_or_path "http://localhost:8001/v1" \
    --judge_type "strongreject" \
    --dataset_name datasets/attack_target/biosecurity_goals.json \
    --output_dir /data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/adv_server \
    --defence_mode True \
    --use_vllm False \
    --vllm_distribution_strategy "server"

# Cleanup
kill $VICTIM_PID $ATTACKER_PID $JUDGE_PID
