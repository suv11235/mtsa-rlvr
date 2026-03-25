#!/bin/bash
#SBATCH --job-name=bio-lora-eval-cp20
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

# Evaluation Configuration
BASELINE="lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2"
CHECKPOINT="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/adv_training_lora_bio_subset_sr/checkpoint-20"
TOKENIZER="meta-llama/Llama-3.1-8B-Instruct"

echo "Running ASR Evaluation for CP20 (Early Hardening) on BIOSECURITY SUBSET"

python script/eval_safety_adaptive.py \
    --baseline_model "$BASELINE" \
    --tokenizer_name_or_path "$TOKENIZER" \
    --checkpoint_path "$CHECKPOINT" \
    --attacker_model "suv11235/red_team_model_SFT_mtsa" \
    --judge_model "qylu4156/strongreject-15k-v1" \
    --judge_type "strongreject" \
    --dataset_path "datasets/attack_target/biosecurity_goals.json" \
    --num_samples 21 \
    --max_turns 3 \
    --gpu_victim "cuda:0" \
    --gpu_attacker "cuda:1" \
    --output_dir "evaluation_reports/bio_lora_eval_cp20_sr"
