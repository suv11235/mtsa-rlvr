#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AUX_GPU="cuda:3"
export HF_TOKEN=${HF_TOKEN:-""}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

# Model Definitions
DEFENDER_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
ATTACKER_MODEL="suv11235/red_team_model_SFT_mtsa" 

# Hyperparameters
LR=5e-6
STEPS=50  # Hard 3 examples, 50 steps is plenty to overfit/test
ROLLOUTS=4
BATCH_SIZE=1 # Full FT takes memory

echo ">>> Starting MTSA-RLVR Defense Training (Full FT, 3+1 GPU)..."
echo "    Defender: $DEFENDER_MODEL"
echo "    Attacker: $ATTACKER_MODEL"
echo "    Setup: Process Group uses 3 GPUs. Aux Judge uses GPU 3."

# Determine python path
if [ -f "/home/ubuntu/mtsa-rlvr/MTSA/venv/bin/python" ]; then
    PYTHON_EXEC="/home/ubuntu/mtsa-rlvr/MTSA/venv/bin/python"
    ACCEL_EXEC="/home/ubuntu/mtsa-rlvr/MTSA/venv/bin/accelerate"
else
    PYTHON_EXEC="python"
    ACCEL_EXEC="accelerate"
fi

# Ensure DeepSpeed config is used by passing it to accelerate or relying on the yaml
# The yaml specifies deepspeed.

$ACCEL_EXEC launch --config_file script/accelerate_configs/zero2_3gpu.yaml \
    -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "$DEFENDER_MODEL" \
    --attacker_model_name_or_path "$ATTACKER_MODEL" \
    --dataset_name "datasets/attack_target/train_attack_target.json" \
    --output_dir "./outputs/rlvr_mtsa_defence_full_ft" \
    --defence_mode True \
    --use_peft False \
    --load_in_4bit False \
    --bf16 True \
    --mini_batch_size $BATCH_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --num_rollouts $ROLLOUTS \
    --max_steps $STEPS \
    --learning_rate $LR \
    --use_tamper_resistance True \
    --tar_inner_loop_steps 1 \
    --tar_inner_lr 5e-5 \
    --judge_reward_weight 1.0 \
    --entropy_reward_weight 1.0 \
    --max_sim_turns 3 \
    --cache_dir "/home/ubuntu/model_cache"
