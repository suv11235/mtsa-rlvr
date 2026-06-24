#!/usr/bin/env bash
# Two-phase pipeline — PHASE 1: triplet-SFT (representation segregation + refusal),
# NO TAR inner tamper loop.
#
# Implemented by running tar_triplet_sft_train.py with inner_steps=0: the inner
# tamper loop is skipped and the weight save/restore becomes a no-op, so the outer
# step minimizes  tar_loss_scale * refusal_CE(harmful) + triplet_loss_scale * (triplet+KL)
# on the *untampered* adapter. This is exactly the CRL-style triplet defense baked
# into an SFT phase that rejects harmful queries while preserving benign behavior
# (KL retain). Crucially it uses the SAME LoRA config as Phase 2, so the resulting
# adapter can be carried forward and made tamper-resistant by TAR.
#
# Env: MAX_STEPS (500) TAR_SCALE (1.0) TRIPLET_SCALE (1.0) GAMMA_KL (0.7)
#      OUTPUT_DIR (./outputs/phase1_triplet_sft)
#
# Run from MTSA/:
#   bash script/slurm/run_phase1_triplet_sft.sh
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

MAX_STEPS="${MAX_STEPS:-500}"
TAR_SCALE="${TAR_SCALE:-1.0}"          # weight on refusal-CE supervision (harmful -> refuse)
TRIPLET_SCALE="${TRIPLET_SCALE:-1.0}"  # weight on triplet+KL representation segregation
GAMMA_KL="${GAMMA_KL:-0.7}"
LOG_EVERY="${LOG_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-100}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/phase1_triplet_sft}"
RUN_NAME="${WANDB_NAME:-phase1-triplet-sft}"
WANDB_PROJECT="${WANDB_PROJECT:-mtsa-tar-triplet}"

mkdir -p logs "$OUTPUT_DIR"

echo "=========================================================="
echo ">>> PHASE 1: triplet-SFT (inner_steps=0, no tamper loop)"
echo ">>> refusal_CE:triplet = $TAR_SCALE:$TRIPLET_SCALE | gamma_kl=$GAMMA_KL"
echo ">>> max_steps=$MAX_STEPS | KL=per-token mean (fixed)"
echo ">>> Output: $OUTPUT_DIR  (adapter saved here for Phase 2)"
echo "=========================================================="

python3 src/algorithm/tar_triplet_sft_train.py \
  --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --harmful_path "datasets/attack_target/train_attack_target_labels.json" \
  --output_dir "$OUTPUT_DIR" \
  --ultrachat_samples 5000 \
  --outer_lr 2e-5 \
  --inner_steps 0 \
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
  --triplet_timing "pre_tamper" \
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

echo ">>> Phase 1 done. Adapter at $OUTPUT_DIR (pass as --init_adapter to Phase 2)."
