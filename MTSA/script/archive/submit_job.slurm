#!/bin/bash
#SBATCH --job-name=mtsa-rlvr-defence
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu    # Change this to your cluster's partition name
#SBATCH --gres=gpu:1       # Number of GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Strict error handling
set -e

# Load necessary modules (Uncomment and adjust based on cluster docs)
# module load cuda/12.1
# module load python/3.11

# Activate virtual environment
# source /path/to/venv/bin/activate

# Move to project root
cd $SLURM_SUBMIT_DIR

# Echo info
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Installation (Use only if env is ephemeral)
# pip install -r requirements.txt

# Run the training script
# Arguments: <model_path> <dataset_path> [output_dir]
bash script/run_rlvr_defence.sh \
    Qwen/Qwen2.5-7B-Instruct \
    datasets/attack_target/train_attack_target.json \
    outputs/rlvr_defence_slurm_$SLURM_JOB_ID

echo "Job finished at $(date)"
