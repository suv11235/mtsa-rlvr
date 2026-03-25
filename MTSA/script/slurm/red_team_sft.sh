# Automatically detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    num_processes=$(nvidia-smi --list-gpus | wc -l)
else
    # Fallback: check CUDA_VISIBLE_DEVICES or default to 1
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    else
        num_processes=1
    fi
fi
echo ">>> Found ${num_processes} GPUs. Enabling Multi-GPU training..."

export WANDB_MODE=disabled
export HF_HOME=/data/suvajit_majumder/huggingface_cache
export HF_TOKEN=${HF_TOKEN:-""}

MODEL_PATH=$1
Red_team_data=$2
OUTPUT_DIR=$3

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=./model_output/red_team_model
    current_time=$(date "+%Y%m%d%H%M%S")  
    OUTPUT_DIR="${OUTPUT_DIR}_data${current_time}"
fi

mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
True_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${True_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_DIR/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
True_DIR="$(dirname "${SCRIPT_DIR}")"
ACCELERATE="${True_DIR}/venv/bin/accelerate"

$ACCELERATE launch --main_process_port ${MAIN_PROCESS_PORT:-29502} --num_processes=$num_processes src/algorithm/red_team_sft.py \
                --dataset_name $Red_team_data \
                --model_name_or_path $MODEL_PATH \
                --use_peft True \
                --bf16 True \
                --load_in_8bit False \
                --load_in_4bit True \
                --attn_implementation sdpa\
                --do_train True \
                --eval_strategy "no" \
                --save_strategy "steps" \
                --num_train_epochs ${TAMPER_EPOCHS:-3} \
                --max_length 2500 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --gradient_checkpointing True \
                --learning_rate 1e-5 \
                --optim "adamw_torch" \
                --lr_scheduler_type "cosine" \
                --warmup_ratio 0.1 \
                --remove_unused_columns False \
                --ddp_find_unused_parameters False \
                --output_dir $OUTPUT_DIR \
    




