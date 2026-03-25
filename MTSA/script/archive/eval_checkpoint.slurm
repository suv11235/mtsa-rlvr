#!/bin/bash
#SBATCH --job-name=eval-checkpoint
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2               # Use 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00            
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Path setup
ROOT_DIR=~/mtsa-rlvr
MTSA_DIR=$ROOT_DIR/MTSA
TAR_REPO_DIR=$ROOT_DIR/tamper-resistance-repo

cd $MTSA_DIR
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd):$TAR_REPO_DIR

# Set HF Token and other env vars
export $(grep -v '^#' .env | xargs)

# Check if checkpoint number is provided
if [ -z "$1" ]; then
    echo "Usage: sbatch eval_checkpoint.slurm <checkpoint_number>"
    exit 1
fi

CP_NUM=$1
CHECKPOINT_PATH="/data/suvajit_majumder/mtsa-rlvr/MTSA/outputs/rlvr_mtsa_long_run_v4gpu/checkpoint-${CP_NUM}"
OUTPUT_DIR="evaluation_reports/checkpoint-${CP_NUM}"
mkdir -p $OUTPUT_DIR

echo "=========================================================="
echo ">>> PART 1: SAFETY EVALUATION (ADAPTIVE ATTACK) <<<"
echo "=========================================================="
python script/eval_safety_adaptive.py \
    --baseline_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --attacker_model "suv1123/red_team_model_SFT_mtsa" \
    --dataset_path "datasets/attack_target/train_attack_target_labels.json" \
    --num_samples 20 \
    --max_turns 3 \
    --gpu_victim "cuda:0" \
    --gpu_attacker "cuda:1" \
    --output_dir "$OUTPUT_DIR/safety_adaptive"

echo "=========================================================="
echo ">>> PART 2: MERGING MODEL FOR TAMPERING EVAL <<<"
echo "=========================================================="
MERGED_MODEL_DIR="$OUTPUT_DIR/merged_model"
python script/prepare_for_tampering_eval.py \
    --base_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --adapter_path "$CHECKPOINT_PATH" \
    --output_path "$MERGED_MODEL_DIR"

echo "=========================================================="
echo ">>> PART 3: TAMPERING EVALUATION (SHALLOW SFT ATTACK) <<<"
echo "=========================================================="
cd $TAR_REPO_DIR
accelerate launch --config_file configs/accel_config_2_gpu.yaml \
    -m red_teaming.red_teaming_evaluation \
    --model_name "$MTSA_DIR/$MERGED_MODEL_DIR" \
    --model_type "meta-llama/Meta-Llama-3-8B-Instruct" \
    --save_model_name "tampered_checkpoint_${CP_NUM}" \
    --training_strategy "pure_pile_bio_forget" \
    --learning_rate 2e-5 \
    --num_epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 100 \
    --max_examples 100 \
    --wandb

echo ">>> Evaluation Complete!"
echo ">>> Safety results in: $MTSA_DIR/$OUTPUT_DIR/safety_adaptive"
