#!/bin/bash
#SBATCH --job-name=eval-scaling-100-highres-tamper
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
OUTPUT_BASE="evaluation_reports/checkpoint-100-highres-tamper"

mkdir -p $OUTPUT_BASE

echo "=========================================================="
echo ">>> PART 2: MERGING MODEL FOR TAMPERING EVAL <<<"
echo "=========================================================="
python script/prepare_for_tampering_eval.py \
    --adapter_path "$CHECKPOINT" \
    --output_path "$OUTPUT_BASE/merged_model"

echo "=========================================================="
echo ">>> PART 3: TAMPERING EVALUATION (SFT ATTACK) <<<"
echo "=========================================================="
export MAIN_PROCESS_PORT=29508
bash script/red_team_sft.sh \
    "$OUTPUT_BASE/merged_model" \
    "datasets/attack_target/train_attack_target_labels.json" \
    "$OUTPUT_BASE/tampering_sft"
