#!/bin/bash
#SBATCH --job-name=eval-extreme-tar-50-tamper-only
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

# We use the existing merged model from the highres run
OUTPUT_BASE="evaluation_reports/extreme_tar_checkpoint_50_highres"

echo "=========================================================="
echo ">>> TAMPERING EVALUATION (SFT ATTACK) - RESTARTING <<<"
echo "=========================================================="
export MAIN_PROCESS_PORT=29507
bash script/red_team_sft.sh \
    "$OUTPUT_BASE/merged_model" \
    "datasets/attack_target/train_attack_target_labels.json" \
    "$OUTPUT_BASE/tampering_sft_restart"
