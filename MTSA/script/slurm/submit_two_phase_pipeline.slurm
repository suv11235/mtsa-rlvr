#!/bin/bash
#SBATCH --job-name=tar-trip-2phase
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Two-phase tamper-resistance pipeline:
#   Phase 1: triplet-SFT (representation segregation + refusal), no tamper loop.
#   Phase 2: TAR meta-loop on the Phase-1 adapter (adaptive-attack hardening).
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

PHASE1_DIR="./outputs/phase1_triplet_sft"
PHASE2_DIR="./outputs/phase2_tar_on_triplet"

echo "################  PHASE 1  ################"
OUTPUT_DIR="$PHASE1_DIR" \
WANDB_NAME="phase1-triplet-sft" \
MAX_STEPS="${PHASE1_STEPS:-500}" \
TAR_SCALE=1.0 TRIPLET_SCALE=1.0 GAMMA_KL=0.7 \
  bash script/slurm/run_phase1_triplet_sft.sh

if [[ ! -f "$PHASE1_DIR/adapter_config.json" ]]; then
  echo "ERROR: Phase 1 did not produce an adapter at $PHASE1_DIR" >&2
  exit 1
fi

echo "################  PHASE 2  ################"
INIT_ADAPTER="$PHASE1_DIR" \
OUTPUT_DIR="$PHASE2_DIR" \
WANDB_NAME="phase2-tar-on-triplet" \
INNER_STEPS="${PHASE2_INNER:-64}" \
MAX_STEPS="${PHASE2_STEPS:-750}" \
TAR_SCALE=4.0 TRIPLET_SCALE=1.0 USE_TRIPLET="${PHASE2_USE_TRIPLET:-0}" GAMMA_KL=0.7 \
  bash script/slurm/run_phase2_tar.sh

echo "################  PIPELINE COMPLETE  ################"
echo "Phase 1 adapter: $PHASE1_DIR"
echo "Phase 2 (tamper-resistant) adapter: $PHASE2_DIR"
