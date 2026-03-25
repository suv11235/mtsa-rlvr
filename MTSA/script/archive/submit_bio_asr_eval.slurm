#!/bin/bash
#SBATCH --job-name=bio-asr-eval
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Use 2 GPUs for Attacker/Victim split
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# Note: Checkpoint 70 is the latest from the bio run
BASELINE="lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2"
CHECKPOINT="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/rlvr_mtsa_pure_tar_checkpoint/checkpoint-70"
TOKENIZER="meta-llama/Meta-Llama-3-8B-Instruct"

echo "Running ASR Evaluation: Baseline vs CP-70 (Bio Dataset)"

python script/eval_safety_adaptive.py \
    --baseline_model "$BASELINE" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --checkpoint_path "$CHECKPOINT" \
    --attacker_model "suv11235/red_team_model_SFT_mtsa" \
    --dataset_path "datasets/attack_target/bio_attack_target.json" \
    --num_samples 100 \
    --max_turns 3 \
    --gpu_victim "cuda:0" \
    --gpu_attacker "cuda:1" \
    --output_dir "evaluation_reports/bio_asr_comparison"
