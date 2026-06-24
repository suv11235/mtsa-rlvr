#!/usr/bin/env bash
# TAR + triplet with hyperparameters aligned to submit_tar_vanilla_baseline.slurm,
# plus triplet/KL on benign (UltraChat) + harmful (train_attack_target_labels.json).
#
# Run from MTSA/:
#   bash script/slurm/run_tar_triplet_sft_tar_match.sh
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

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/tar_triplet_sft_tar_match}"
mkdir -p logs "$OUTPUT_DIR"

echo "=========================================================="
echo ">>> TAR + Triplet (TAR-matched: 64 inner, 750 steps, 512 len)"
echo ">>> Output: $OUTPUT_DIR"
echo "=========================================================="

python3 src/algorithm/tar_triplet_sft_train.py \
  --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --harmful_path "datasets/attack_target/train_attack_target_labels.json" \
  --output_dir "$OUTPUT_DIR" \
  --ultrachat_samples 5000 \
  --inner_lr 4e-5 \
  --outer_lr 2e-5 \
  --inner_steps 64 \
  --max_steps 750 \
  --gradient_accumulation_steps 8 \
  --batch_size 1 \
  --max_length 512 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --tar_type "entropy" \
  --tar_loss_scale 4.0 \
  --use_triplet_loss \
  --triplet_loss_scale 0.1 \
  --log_every 5 \
  --save_every 25

echo ">>> Done. Adapter saved to $OUTPUT_DIR"
