#!/bin/bash
#SBATCH --job-name=mtsa-3way-tamper-merged
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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

ACCELERATE=$(which accelerate)

# Shallow Settings
export TAMPER_EPOCHS=1
DATASET="datasets/attack_target/tamper_100_random.json"

echo "=========================================================="
echo ">>> 3-WAY TAMPERING EVAL (100 Samples, 1 Epoch) <<<"
echo ">>> Comparing Merged Extreme V2 vs v1 vs Vanilla <<<"
echo "=========================================================="

# Task 1: Extreme TAR v2 (Merged Step 25)
echo ">>> Processing EXTREME V2 (Step 25)..."
(
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29530
$ACCELERATE launch --main_process_port $MAIN_PROCESS_PORT --num_processes=1 src/algorithm/red_team_sft.py \
                --dataset_name "$DATASET" \
                --model_name_or_path "evaluation_reports/extreme_v2_step25_merged" \
                --use_peft True \
                --bf16 True \
                --load_in_4bit True \
                --attn_implementation sdpa \
                --do_train True \
                --num_train_epochs $TAMPER_EPOCHS \
                --max_length 512 \
                --per_device_train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-5 \
                --output_dir "evaluation_reports/extreme_v2_step25_tamper_merged"
) &

# Task 2: Extreme TAR v1 (Merged Step 75)
echo ">>> Processing EXTREME V1 (Step 75)..."
(
export CUDA_VISIBLE_DEVICES=1
export MAIN_PROCESS_PORT=29531
$ACCELERATE launch --main_process_port $MAIN_PROCESS_PORT --num_processes=1 src/algorithm/red_team_sft.py \
                --dataset_name "$DATASET" \
                --model_name_or_path "evaluation_reports/extreme_v1_step75_merged" \
                --use_peft True \
                --bf16 True \
                --load_in_4bit True \
                --attn_implementation sdpa \
                --do_train True \
                --num_train_epochs $TAMPER_EPOCHS \
                --max_length 512 \
                --per_device_train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-5 \
                --output_dir "evaluation_reports/extreme_v1_step75_tamper_merged"
) &

# Task 3: Vanilla TAR Baseline (Merged Step 50)
echo ">>> Processing VANILLA BASELINE (Step 50)..."
(
export CUDA_VISIBLE_DEVICES=2
export MAIN_PROCESS_PORT=29532
$ACCELERATE launch --main_process_port $MAIN_PROCESS_PORT --num_processes=1 src/algorithm/red_team_sft.py \
                --dataset_name "$DATASET" \
                --model_name_or_path "evaluation_reports/vanilla_baseline_step50_merged" \
                --use_peft True \
                --bf16 True \
                --load_in_4bit True \
                --attn_implementation sdpa \
                --do_train True \
                --num_train_epochs $TAMPER_EPOCHS \
                --max_length 512 \
                --per_device_train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-5 \
                --output_dir "evaluation_reports/vanilla_baseline_step50_tamper_merged"
) &

wait
echo ">>> ALL TAMPERING EVALS COMPLETE <<<"
