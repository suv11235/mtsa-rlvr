#!/usr/bin/env bash
# Two-phase pipeline — PHASE 2: TAR tamper-resistance on the Phase-1 adapter.
#
# Resumes the Phase-1 triplet-SFT LoRA (INIT_ADAPTER) and runs the TAR meta-loop
# (entropy inner tamper, refusal-CE outer) to harden it against adaptive attacks.
# Triplet+KL is OFF by default (USE_TRIPLET=0): representation segregation is already
# baked in by Phase 1, and re-adding the unbounded triplet/KL terms here destabilizes
# (they are measured on heavily-tampered post_tamper weights and run away). Set
# USE_TRIPLET=1 only to experiment with light segregation maintenance during TAR.
#
# Env: INIT_ADAPTER (required) INNER_STEPS (64) MAX_STEPS (750)
#      TAR_SCALE (4.0) TRIPLET_SCALE (1.0) USE_TRIPLET (1) GAMMA_KL (0.7)
#
# Run from MTSA/:
#   INIT_ADAPTER=./outputs/phase1_triplet_sft bash script/slurm/run_phase2_tar.sh
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

INIT_ADAPTER="${INIT_ADAPTER:?set INIT_ADAPTER to the Phase-1 output dir}"
if [[ ! -f "$INIT_ADAPTER/adapter_config.json" ]]; then
  echo "ERROR: no adapter_config.json under INIT_ADAPTER=$INIT_ADAPTER" >&2
  exit 1
fi

INNER_STEPS="${INNER_STEPS:-8}"
MAX_STEPS="${MAX_STEPS:-750}"
TAR_SCALE="${TAR_SCALE:-4.0}"
TRIPLET_SCALE="${TRIPLET_SCALE:-1.0}"
USE_TRIPLET="${USE_TRIPLET:-0}"
TAR_TR_OBJECTIVE="${TAR_TR_OBJECTIVE:-max_entropy}"
RETAIN_KL_SCALE="${RETAIN_KL_SCALE:-1.0}"  # KL-vs-base capability anchor (paper lambda_retain)
INNER_LR_SAMPLES="${INNER_LR_SAMPLES:-2e-5,4e-5,1e-4}"
GAMMA_KL="${GAMMA_KL:-0.7}"
LOG_EVERY="${LOG_EVERY:-5}"
SAVE_EVERY="${SAVE_EVERY:-50}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/phase2_tar_on_triplet}"
RUN_NAME="${WANDB_NAME:-phase2-tar-on-triplet}"
WANDB_PROJECT="${WANDB_PROJECT:-mtsa-tar-triplet}"

mkdir -p logs "$OUTPUT_DIR"

TRIPLET_FLAG=(--use_triplet_loss)
if [[ "$USE_TRIPLET" == "0" ]]; then
  TRIPLET_FLAG=(--no_use_triplet_loss)
fi

echo "=========================================================="
echo ">>> PHASE 2: TAR on Phase-1 adapter"
echo ">>> init_adapter=$INIT_ADAPTER"
echo ">>> inner_steps=$INNER_STEPS max_steps=$MAX_STEPS"
echo ">>> tar_tr_objective=$TAR_TR_OBJECTIVE inner_steps=$INNER_STEPS inner_lr_samples=$INNER_LR_SAMPLES"
echo ">>> tar_scale=$TAR_SCALE retain_kl_scale=$RETAIN_KL_SCALE use_triplet=$USE_TRIPLET"
echo ">>> Output: $OUTPUT_DIR"
echo "=========================================================="

python3 src/algorithm/tar_triplet_sft_train.py \
  --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --init_adapter "$INIT_ADAPTER" \
  --harmful_path "datasets/attack_target/train_attack_target_labels.json" \
  --output_dir "$OUTPUT_DIR" \
  --ultrachat_samples 5000 \
  --inner_lr 1e-4 \
  --inner_lr_samples "$INNER_LR_SAMPLES" \
  --outer_lr 2e-5 \
  --inner_steps "$INNER_STEPS" \
  --max_steps "$MAX_STEPS" \
  --gradient_accumulation_steps 8 \
  --batch_size 1 \
  --max_length 512 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --tar_tr_objective "$TAR_TR_OBJECTIVE" \
  --tar_loss_scale "$TAR_SCALE" \
  --retain_kl_scale "$RETAIN_KL_SCALE" \
  "${TRIPLET_FLAG[@]}" \
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

echo ">>> Phase 2 done. Tamper-resistant adapter at $OUTPUT_DIR"
