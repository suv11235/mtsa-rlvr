#!/usr/bin/env bash
# Interactive smoke run for TAR + triplet SFT (1 GPU, few steps).
# On CAIS login node:
#   cd ~/mtsa-rlvr/MTSA && bash script/slurm/run_tar_triplet_sft_interactive.sh
#
# Or allocate GPU shell first, then run this inside:
#   srun --partition=tamper_resistance --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --pty bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HARMFUL="${HARMFUL_PATH:-../circuit_breakers_train.json}"
if [[ ! -f "$HARMFUL" && -f "datasets/attack_target/train_attack_target_labels.json" ]]; then
  HARMFUL="datasets/attack_target/train_attack_target_labels.json"
fi

mkdir -p outputs/tar_triplet_sft_interactive

echo ">>> Interactive smoke: TAR + triplet SFT"
echo ">>> harmful_path=$HARMFUL"
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-default}"

python3 src/algorithm/tar_triplet_sft_train.py \
  --model_name_or_path "${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --harmful_path "$HARMFUL" \
  --output_dir "./outputs/tar_triplet_sft_interactive" \
  --ultrachat_samples "${ULTRACHAT_SAMPLES:-16}" \
  --limit_harmful "${LIMIT_HARMFUL:-32}" \
  --inner_lr 4e-5 \
  --outer_lr 2e-5 \
  --inner_steps "${INNER_STEPS:-2}" \
  --max_steps "${MAX_STEPS:-3}" \
  --gradient_accumulation_steps 1 \
  --batch_size 1 \
  --tar_type entropy \
  --tar_loss_scale 4.0 \
  --use_triplet_loss \
  --triplet_loss_scale 0.1 \
  --max_length 128 \
  --log_every 1 \
  --save_every 9999

echo ">>> Smoke run complete -> outputs/tar_triplet_sft_interactive"
