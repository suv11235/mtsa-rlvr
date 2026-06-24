#!/usr/bin/env bash
# Interactive smoke run for standalone triplet SFT (no TAR inner loop).
# cd ~/mtsa-rlvr/MTSA && bash script/slurm/run_triplet_sft_interactive.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

HARMFUL="${HARMFUL_PATH:-../circuit_breakers_train.json}"
if [[ ! -f "$HARMFUL" && -f "datasets/attack_target/train_attack_target_labels.json" ]]; then
  HARMFUL="datasets/attack_target/train_attack_target_labels.json"
fi

mkdir -p outputs/triplet_sft_interactive

echo ">>> Interactive smoke: standalone triplet SFT"

python3 -m src.algorithm.triplet_sft_train \
  --model "${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --harmful_path "$HARMFUL" \
  --output_dir "./outputs/triplet_sft_interactive" \
  --ultrachat_samples "${ULTRACHAT_SAMPLES:-32}" \
  --limit_harmful "${LIMIT_HARMFUL:-32}" \
  --batch_size 1 \
  --max_steps "${MAX_STEPS:-5}" \
  --max_length 128 \
  --logging_steps 1 \
  --save_steps 9999 \
  --report_to none

echo ">>> Smoke run complete -> outputs/triplet_sft_interactive"
