#!/usr/bin/env bash
# Stabilization fork of the TAR+triplet joint run.
#
# Changes vs run_tar_triplet_sft_crl_aligned.sh, motivated by the instability
# diagnosis (KL retain term dominating + TAR signal drowned):
#   1. KL retain is now per-response-token MEAN (triplet_rep.py fix) -> O(1).
#   2. TAR-dominant outer ratio: tar_loss_scale=4 : triplet_loss_scale=1
#      (matches TAR paper lambda_TR=4 : lambda_retain=1; was 1:1).
#   3. Fewer inner tamper steps (8, was 64) to shrink the MAML eval-vs-apply
#      mismatch and speed iteration. Probes the loop-dynamics hypothesis.
#
# Parametrized via env so the smoke run and the full run share one script:
#   INNER_STEPS (8) MAX_STEPS (750) TAR_SCALE (4.0) TRIPLET_SCALE (1.0)
#   GAMMA_KL (0.7) LOG_EVERY (5) SAVE_EVERY (25)
#
# Run from MTSA/:
#   bash script/slurm/run_tar_triplet_sft_stabilize.sh
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

INNER_STEPS="${INNER_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-750}"
TAR_SCALE="${TAR_SCALE:-4.0}"
TRIPLET_SCALE="${TRIPLET_SCALE:-1.0}"
GAMMA_KL="${GAMMA_KL:-0.7}"
LOG_EVERY="${LOG_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-25}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/tar_triplet_sft_stabilize_bf16}"
RUN_NAME="${WANDB_NAME:-tar-triplet-stabilize-kfix-inner${INNER_STEPS}-tar${TAR_SCALE%.*}}"
WANDB_PROJECT="${WANDB_PROJECT:-mtsa-tar-triplet}"

mkdir -p logs "$OUTPUT_DIR"

echo "=========================================================="
echo ">>> TAR + Triplet STABILIZE fork"
echo ">>> KL fix (per-token mean) | post_tamper alpha_mode=all"
echo ">>> tar_loss_scale=$TAR_SCALE : triplet_loss_scale=$TRIPLET_SCALE (TAR-dominant)"
echo ">>> inner_steps=$INNER_STEPS max_steps=$MAX_STEPS gamma_kl=$GAMMA_KL"
echo ">>> precision=bf16 LoRA, kl_temperature=1.0"
echo ">>> wandb project=$WANDB_PROJECT run=$RUN_NAME"
echo ">>> Output: $OUTPUT_DIR"
echo "=========================================================="

python3 src/algorithm/tar_triplet_sft_train.py \
  --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --harmful_path "datasets/attack_target/train_attack_target_labels.json" \
  --output_dir "$OUTPUT_DIR" \
  --ultrachat_samples 5000 \
  --inner_lr 4e-5 \
  --outer_lr 2e-5 \
  --inner_steps "$INNER_STEPS" \
  --max_steps "$MAX_STEPS" \
  --gradient_accumulation_steps 8 \
  --batch_size 1 \
  --max_length 512 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --tar_type "entropy" \
  --tar_loss_scale "$TAR_SCALE" \
  --use_triplet_loss \
  --triplet_loss_scale "$TRIPLET_SCALE" \
  --triplet_timing "post_tamper" \
  --alpha_mode "all" \
  --alpha_safe 0.5 \
  --beta_unsafe 0.6 \
  --gamma_kl "$GAMMA_KL" \
  --margin_safe 2.0 \
  --margin_unsafe 3.0 \
  --kl_temperature 1.0 \
  --run_name "$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --log_every "$LOG_EVERY" \
  --save_every "$SAVE_EVERY"

echo ">>> Done. Adapter saved to $OUTPUT_DIR"
