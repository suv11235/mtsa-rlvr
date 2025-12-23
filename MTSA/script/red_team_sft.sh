export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export HF_HOME=/workspace/huggingface_cache


Qwen_model_path=$1
Red_team_data=$2

OUTPUT_DIR=./model_output/red_team_model
current_time=$(date "+%Y%m%d%H%M%S")  
OUTPUT_DIR="${OUTPUT_DIR}_data${current_time}"
mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
True_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${True_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
SCRIPT_NAME=$(basename "$0")
DESTINATION_PATH="$OUTPUT_DIR/$SCRIPT_NAME"
cp "$0" "$DESTINATION_PATH"
num_processes=1


accelerate launch --main_process_port 29502 --num_processes=$num_processes src/algorithm/red_team_sft.py \
                --dataset_name $Red_team_data \
                --model_name_or_path $Qwen_model_path \
                --torch_dtype "bfloat16" \
                --use_peft True \
                --bf16 True \
                --load_in_8bit False \
                --load_in_4bit True \
                --attn_implementation sdpa\
                --do_train True \
                --eval_strategy "no" \
                --save_strategy "steps" \
                --num_train_epochs 3 \
                --max_length 2500 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --gradient_checkpointing True \
                --learning_rate 1e-5 \
                --optim "adamw_torch" \
                --lr_scheduler_type "cosine" \
                --warmup_ratio 0.1 \
                --remove_unused_columns False \
                --output_dir $OUTPUT_DIR \
    




