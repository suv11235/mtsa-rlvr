#!/bin/bash
#SBATCH --job-name=tar-vanilla-baseline
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# Create output dir
mkdir -p outputs/tar_vanilla_baseline

echo "=========================================================="
echo ">>> RUNNING VANILLA TAR BASELINE (Standard Framework) <<<"
echo ">>> Config: Paper-Faithful (64 Inner, Entropy Obj) <<<" 
echo "=========================================================="

python3 src/algorithm/tar_vanilla_train.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_name "datasets/attack_target/train_attack_target_labels.json" \
    --output_dir "./outputs/tar_vanilla_baseline" \
    --inner_lr 4e-5 \
    --outer_lr 2e-5 \
    --inner_steps 64 \
    --max_steps 750 \
    --gradient_accumulation_steps 8 \
    --tar_type "entropy" \
    --tar_loss_scale 4.0 \
    --max_length 512

echo ">>> VANILLA TAR BASELINE COMPLETE <<<"
