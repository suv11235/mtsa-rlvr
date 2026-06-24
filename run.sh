#!/usr/bin/env bash
# Standalone triplet SFT (modular implementation under MTSA/src)
set -euo pipefail
cd "$(dirname "$0")/MTSA"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -m src.algorithm.triplet_sft_train \
  --model Qwen/Qwen3-8B \
  --harmful_path ../circuit_breakers_train.json \
  --output_dir ../runs/triplet_qwen \
  --ultrachat_samples 5000 \
  --batch_size 4 \
  --max_steps 1500 \
  --lr 2e-4
