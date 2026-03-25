#!/bin/bash
#SBATCH --job-name=mtsa-merge-checkpoints
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export $(grep HF_TOKEN .env | xargs)

BASE="meta-llama/Meta-Llama-3-8B-Instruct"

echo ">>> Merging EXTREME V2 Step 25..."
python3 script/prepare_for_tampering_eval.py \
    --base_model "$BASE" \
    --adapter_path "outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-25" \
    --output_path "evaluation_reports/extreme_v2_step25_merged"

echo ">>> Merging EXTREME V1 Step 75..."
python3 script/prepare_for_tampering_eval.py \
    --base_model "$BASE" \
    --adapter_path "outputs/rlvr_mtsa_extreme_tar/checkpoint-75" \
    --output_path "evaluation_reports/extreme_v1_step75_merged"

echo ">>> Merging VANILLA BASELINE Step 50..."
python3 script/prepare_for_tampering_eval.py \
    --base_model "$BASE" \
    --adapter_path "outputs/tar_vanilla_baseline/checkpoint-50" \
    --output_path "evaluation_reports/vanilla_baseline_step50_merged"

echo ">>> ALL MERGES COMPLETE <<<"
