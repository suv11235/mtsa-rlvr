#!/usr/bin/env bash
# TAR + triplet aligned with CRL reference (arXiv:2506.11938 / crl-llm-defense train.sh)
# plus TAR inner tamper loop (64 steps, entropy).
#
# CRL alignment:
#   - post_tamper triplet (reps + KL on θ′ after inner tamper, with refusal CE)
#   - alpha_mode=all, margins 2/3, α=0.5 β=0.6 γ(KL)=0.7, LoRA r=16
#   - masked_hinge_mean layer-normalized (lossfix)
#   - tar_loss_scale=1.0, triplet_loss_scale=1.0 → 1:1 weighted outer terms
#
# TAR match (submit_tar_vanilla_baseline.slurm):
#   - 64 inner steps, 750 outer steps, max_length 512, grad_accum 8
#
# Run from MTSA/:
#   bash script/slurm/run_tar_triplet_sft_crl_aligned.sh
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

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/tar_triplet_sft_crl_aligned_post_tamper_bf16}"
RUN_NAME="${WANDB_NAME:-tar-triplet-crl-post-tamper-bf16}"
WANDB_PROJECT="${WANDB_PROJECT:-mtsa-tar-triplet}"

mkdir -p logs "$OUTPUT_DIR"

echo "=========================================================="
echo ">>> TAR + Triplet (CRL-aligned + TAR 64-inner / 750-step)"
echo ">>> triplet_timing=post_tamper alpha_mode=all"
echo ">>> tar_loss_scale=1.0 triplet_loss_scale=1.0 (1:1 weighted)"
echo ">>> precision=bf16 LoRA (no 4-bit), kl_temperature=1.0"
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
  --inner_steps 64 \
  --max_steps 750 \
  --gradient_accumulation_steps 8 \
  --batch_size 1 \
  --max_length 512 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --tar_type "entropy" \
  --tar_loss_scale 1.0 \
  --use_triplet_loss \
  --triplet_loss_scale 1.0 \
  --triplet_timing "post_tamper" \
  --alpha_mode "all" \
  --alpha_safe 0.5 \
  --beta_unsafe 0.6 \
  --gamma_kl 0.7 \
  --margin_safe 2.0 \
  --margin_unsafe 3.0 \
  --kl_temperature 1.0 \
  --run_name "$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --log_every 5 \
  --save_every 25

echo ">>> Done. Adapter saved to $OUTPUT_DIR"
