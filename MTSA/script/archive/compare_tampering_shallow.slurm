#!/bin/bash
#SBATCH --job-name=tamper-shallow-100
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
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
echo ">>> SHALLOW TAMPERING EVAL (100 Samples, 1 Epoch) <<<"
echo "=========================================================="

# Task 1: Scaling Run Checkpoint 100
echo ">>> Processing SCALING 100..."
(
export CUDA_VISIBLE_DEVICES=0,1
export MAIN_PROCESS_PORT=29510
$ACCELERATE launch --main_process_port $MAIN_PROCESS_PORT --num_processes=2 src/algorithm/red_team_sft.py \
                --dataset_name "$DATASET" \
                --model_name_or_path "evaluation_reports/checkpoint-100-highres-tamper/merged_model" \
                --use_peft True \
                --bf16 True \
                --load_in_4bit True \
                --attn_implementation sdpa \
                --do_train True \
                --num_train_epochs $TAMPER_EPOCHS \
                --max_length 2500 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --gradient_checkpointing True \
                --learning_rate 1e-5 \
                --output_dir "evaluation_reports/checkpoint-100-shallow-tamper"
) &

# Task 2: Extreme TAR Checkpoint 50
echo ">>> Processing EXTREME TAR 50..."
(
export CUDA_VISIBLE_DEVICES=2,3
export MAIN_PROCESS_PORT=29511
$ACCELERATE launch --main_process_port $MAIN_PROCESS_PORT --num_processes=2 src/algorithm/red_team_sft.py \
                --dataset_name "$DATASET" \
                --model_name_or_path "evaluation_reports/extreme_tar_checkpoint_50_highres/merged_model" \
                --use_peft True \
                --bf16 True \
                --load_in_4bit True \
                --attn_implementation sdpa \
                --do_train True \
                --num_train_epochs $TAMPER_EPOCHS \
                --max_length 2500 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --gradient_checkpointing True \
                --learning_rate 1e-5 \
                --output_dir "evaluation_reports/extreme_tar_50-shallow-tamper"
) &

wait
echo ">>> ALL SHALLOW TAMPERING JOBS COMPLETE <<<"
