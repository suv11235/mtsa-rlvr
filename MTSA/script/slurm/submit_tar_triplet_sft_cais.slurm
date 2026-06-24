#!/bin/bash
#SBATCH --job-name=tar-triplet-1to1
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
export WANDB_NAME="tar-triplet-crl-post-tamper-bf16"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_DIR="./outputs/tar_triplet_sft_crl_aligned_post_tamper_bf16"

bash script/slurm/run_tar_triplet_sft_crl_aligned.sh
