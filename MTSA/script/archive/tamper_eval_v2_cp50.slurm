#!/bin/bash
#SBATCH --job-name=tamper-v2-cp50
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

ROOT_DIR=~/mtsa-rlvr
MTSA_DIR=$ROOT_DIR/MTSA
TAR_REPO_DIR=$ROOT_DIR/tamper-resistance-repo

cd $MTSA_DIR
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd):$TAR_REPO_DIR
export $(grep -v '^#' .env | xargs)

CONFIG_FILE="$TAR_REPO_DIR/configs/accel_config_1_gpu.yaml"

cd $TAR_REPO_DIR
accelerate launch --config_file $CONFIG_FILE \
    -m red_teaming.red_teaming_evaluation \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --model_type "meta-llama/Meta-Llama-3-8B-Instruct" \
    --adapter_path "$MTSA_DIR/outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-50" \
    --save_model_name "tampered_v3_extreme_v2_step50" \
    --training_strategy "attack_target_forget" \
    --learning_rate 2e-5 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 50 \
    --load_in_4bit \
    --wandb
