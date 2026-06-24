#!/bin/bash
#SBATCH --job-name=phase2-tar-pure
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# PHASE 2 (pure TAR, NO triplet) resumed from an existing Phase-1 triplet-SFT adapter.
# Representation segregation is already baked in by Phase 1; Phase 2 only hardens
# tamper-resistance. Dropping the (unbounded) triplet+KL terms removes the runaway
# observed when they were measured on heavily-tampered post_tamper weights.
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
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_NAME="phase2-tar-official-inner8"

export INIT_ADAPTER="${INIT_ADAPTER:-./outputs/phase1_triplet_sft}"
export OUTPUT_DIR="./outputs/phase2_tar_official"
export USE_TRIPLET=0
export TAR_TR_OBJECTIVE=max_entropy
export TAR_SCALE=4.0
export RETAIN_KL_SCALE=1.0
export INNER_STEPS="${INNER_STEPS:-8}"
export INNER_LR_SAMPLES="2e-5,4e-5,1e-4"
export MAX_STEPS="${MAX_STEPS:-750}"
export SAVE_EVERY=50
export LOG_EVERY=5

bash script/slurm/run_phase2_tar.sh
