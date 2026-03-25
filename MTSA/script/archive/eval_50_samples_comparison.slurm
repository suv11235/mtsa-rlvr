#!/bin/bash
#SBATCH --job-name=eval-50-samples
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

CHECKPOINT="./outputs/rlvr_mtsa_long_run_v4gpu/checkpoint-100"
OUTPUT_BASE="evaluation_reports/checkpoint-100-50samples"

mkdir -p $OUTPUT_BASE

echo "=========================================================="
echo ">>> ADAPTIVE SAFETY EVALUATION (50 SAMPLES CRITICAL COMPARE) <<<"
echo "=========================================================="

# We do NOT skip baseline here to get the high-resolution baseline score
python script/eval_safety_adaptive.py \
    --checkpoint_path "$CHECKPOINT" \
    --attacker_model "suv11235/red_team_model_SFT_mtsa" \
    --dataset_path "datasets/attack_target/eval_50.json" \
    --num_samples 50 \
    --max_turns 3 \
    --output_dir "$OUTPUT_BASE/safety_adaptive" \
    --judge_model "meta-llama/Llama-Guard-3-8B"

echo "=========================================================="
echo ">>> EVALUATION COMPLETE <<<"
echo "=========================================================="
