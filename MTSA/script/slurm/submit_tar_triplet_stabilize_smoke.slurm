#!/bin/bash
#SBATCH --job-name=tar-trip-smoke
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
mkdir -p logs
source venv/bin/activate
pip install -q 'wandb>=0.18.0'
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
set -a
source .env
set +a
export WANDB_PROJECT="${WANDB_PROJECT:-mtsa-tar-triplet}"
export WANDB_NAME="tar-triplet-stabilize-smoke-kfix-inner8-tar4"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_DIR="./outputs/tar_triplet_sft_stabilize_smoke"

# Smoke: 60 steps, fast inner loop, TAR-dominant, KL-fixed.
export INNER_STEPS=8
export MAX_STEPS=60
export TAR_SCALE=4.0
export TRIPLET_SCALE=1.0
export GAMMA_KL=0.7
export LOG_EVERY=5
export SAVE_EVERY=1000  # don't checkpoint during smoke

bash script/slurm/run_tar_triplet_sft_stabilize.sh
