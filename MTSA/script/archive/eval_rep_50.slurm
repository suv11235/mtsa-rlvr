#!/bin/bash
#SBATCH --job-name=eval-rep-50
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set Environment Variables from your .env
export $(grep -v '^#' .env | xargs)

# Note: Check the exact path for RepE output dir
CHECKPOINT="./outputs/rlvr_mtsa_rep_loss/checkpoint-50"
OUTPUT_BASE="evaluation_reports/rep_checkpoint_50"

mkdir -p $OUTPUT_BASE

# 1. SAFETY EVALUATION (ADAPTIVE ATTACK)
echo "=========================================================="
echo ">>> PART 1: SAFETY EVALUATION (ADAPTIVE ATTACK - 20 SAMPLES) <<<"
echo "=========================================================="
python script/eval_safety_adaptive.py \
    --checkpoint_path "$CHECKPOINT" \
    --attacker_model "suv11235/red_team_model_SFT_mtsa" \
    --dataset_path "datasets/attack_target/train_attack_target_labels.json" \
    --num_samples 20 \
    --max_turns 3 \
    --output_dir "$OUTPUT_BASE/safety_adaptive" \
    --judge_model "meta-llama/Llama-Guard-3-8B" \
    --skip_baseline

# 2. TAMPERING EVALUATION (SFT ATTACK)
echo "=========================================================="
echo ">>> PART 2: MERGING MODEL FOR TAMPERING EVAL <<<"
echo "=========================================================="
python script/prepare_for_tampering_eval.py \
    --adapter_path "$CHECKPOINT" \
    --output_path "$OUTPUT_BASE/merged_model"

echo "=========================================================="
echo ">>> PART 3: TAMPERING EVALUATION (SFT ATTACK) <<<"
echo "=========================================================="
bash script/red_team_sft.sh \
    "$OUTPUT_BASE/merged_model" \
    "datasets/attack_target/train_attack_target_labels.json" \
    "$OUTPUT_BASE/tampering_sft"
