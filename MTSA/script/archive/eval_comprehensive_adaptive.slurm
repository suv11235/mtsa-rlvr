#!/bin/bash
#SBATCH --job-name=eval-comp-adaptive
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# Paths
BASE_DIR="/data/suvajit_majumder/mtsa-rlvr/MTSA"

# Function to run eval
run_eval() {
    local NAME=$1
    local CP=$2
    local EXTRA_ARGS=$3
    
    echo "=================================================="
    echo ">>> EVALUATING: $NAME"
    echo "=================================================="
    
    python script/eval_safety_adaptive.py \
        --checkpoint_path "$CP" \
        --num_samples 50 \
        --output_dir "evaluation_reports/comp_adaptive_$NAME" \
        $EXTRA_ARGS
}

# 1. Baseline
run_eval "baseline" "" ""

# 2. 1-Step TAR (Long Run CP-50) - Corrected Path
run_eval "tar_1step_cp50" "$BASE_DIR/outputs/rlvr_mtsa_long_run_v4gpu/checkpoint-50" "--skip_baseline"

# 3. 64-Step TAR (Extreme v2 CP-100)
run_eval "64step_tar_cp100" "$BASE_DIR/outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-100" "--skip_baseline"
