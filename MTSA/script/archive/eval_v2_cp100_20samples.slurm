#!/bin/bash
#SBATCH --job-name=eval-extreme-v2-cp100-20samples
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

CHECKPOINT="./outputs/rlvr_mtsa_extreme_tar_v2/checkpoint-100"
OUTPUT_DIR="evaluation_reports/extreme_tar_v2_checkpoint_100_20samples"

mkdir -p $OUTPUT_DIR

echo ">>> Starting 20-sample Adaptive Attack Eval for Extreme v2 CP-100"
python script/eval_safety_adaptive.py \
    --checkpoint_path "$CHECKPOINT" \
    --attacker_model "suv11235/red_team_model_SFT_mtsa" \
    --dataset_path "datasets/attack_target/eval_50.json" \
    --num_samples 20 \
    --max_turns 3 \
    --output_dir "$OUTPUT_DIR" \
    --judge_model "meta-llama/Llama-Guard-3-8B" \
    --skip_baseline
