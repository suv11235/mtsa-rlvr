#!/bin/bash
#SBATCH --job-name=mtsa-eval-adaptive
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Use 2 GPUs for Attacker/Victim split
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00            # 4 hours should be plenty for 100 samples
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set Environment Variables from your .env
export $(grep -v '^#' .env | xargs)

echo "Starting Longer Adaptive Evaluation on $(hostname)"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

# Run the adaptive evaluation script
# We use a larger number of samples (100) from the full dataset
python script/eval_safety_adaptive.py \
    --baseline_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --checkpoint_path "./outputs/rlvr_mtsa_test/checkpoint-6" \
    --attacker_model "suv11235/red_team_model_SFT_mtsa" \
    --dataset_path "datasets/attack_target/train_attack_target_labels.json" \
    --num_samples 100 \
    --max_turns 3 \
    --skip_baseline \
    --gpu_victim "cuda:0" \
    --gpu_attacker "cuda:1"
