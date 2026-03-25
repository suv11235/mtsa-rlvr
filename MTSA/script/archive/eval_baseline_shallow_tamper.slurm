#!/bin/bash
#SBATCH --job-name=tamper-baseline-shallow
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Strict error handling
set -e

# Load Python Environment
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export $(grep -v '^#' .env | xargs)

ACCELERATE=$(which accelerate)

# Shallow Settings
export TAMPER_EPOCHS=1
DATASET="datasets/attack_target/tamper_100_random.json"

echo "=========================================================="
echo ">>> BASELINE SHALLOW TAMPERING EVAL (100 Samples, 1 Epoch) <<<"
echo "=========================================================="

export MAIN_PROCESS_PORT=29512
$ACCELERATE launch --main_process_port $MAIN_PROCESS_PORT --num_processes=2 src/algorithm/red_team_sft.py \
                --dataset_name "$DATASET" \
                --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
                --use_peft True \
                --bf16 True \
                --load_in_4bit True \
                --attn_implementation sdpa \
                --do_train True \
                --num_train_epochs $TAMPER_EPOCHS \
                --max_length 2500 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --gradient_checkpointing True \
                --learning_rate 1e-5 \
                --output_dir "evaluation_reports/baseline-shallow-tamper"

echo ">>> BASELINE SHALLOW TAMPERING COMPLETE <<<"
