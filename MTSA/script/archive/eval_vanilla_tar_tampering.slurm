#!/bin/bash
#SBATCH --job-name=vanilla-tar-tamper-eval
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# 1. Tampering (SFT) on the Vanilla TAR Checkpoint
# Use 50 steps, log every 5
echo ">>> STARTING TAMPERING (50 steps) on Vanilla TAR Checkpoint-50..."

python3 src/algorithm/red_team_sft.py \
    --model_name_or_path "outputs/tar_vanilla_baseline/checkpoint-50" \
    --dataset_name "datasets/attack_target/train_attack_target_labels.json" \
    --output_dir "outputs/vanilla_tar_post_tamper_50steps" \
    --max_steps 50 \
    --logging_steps 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_peft True \
    --load_in_4bit True \
    --attn_implementation sdpa \
    --gradient_checkpointing True \
    --save_steps 50

echo ">>> TAMPERING COMPLETE. STARTING ADVERSARIAL EVAL (ADAPTIVE ASR)..."

# 2. Adversarial Eval on the post-tampered model
# We use script/eval_safety_adaptive.py
python3 script/eval_safety_adaptive.py \
    --checkpoint_path "outputs/vanilla_tar_post_tamper_50steps" \
    --num_samples 20 \
    --max_turns 3 \
    --skip_baseline \
    --output_dir "evaluation_reports/vanilla_tar_tamper_eval" \
    --gpu_victim "cuda:0" \
    --gpu_attacker "cuda:1"

echo ">>> EVALUATION COMPLETE. Check reports in evaluation_reports/vanilla_tar_tamper_eval"
