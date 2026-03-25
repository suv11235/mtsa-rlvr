#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_TOKEN=${HF_TOKEN:-""}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

# Model Definitions
DEFENDER_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
# This automatically loads 8B base + your SFT adapter
ATTACKER_MODEL="suv11235/red_team_model_SFT_mtsa" 

# Hyperparameters
LR=5e-6
STEPS=300
ROLLOUTS=4
BATCH_SIZE=2

echo ">>> Starting MTSA-RLVR Defense Training..."
echo "    Defender: $DEFENDER_MODEL"
echo "    Attacker: $ATTACKER_MODEL"

# Determine python path based on environment (local vs lambda)
if [ -f "/home/ubuntu/mtsa-rlvr/MTSA/venv/bin/python" ]; then
    PYTHON_EXEC="/home/ubuntu/mtsa-rlvr/MTSA/venv/bin/python"
else
    PYTHON_EXEC="python"
fi

$PYTHON_EXEC -m src.algorithm.mt_rlvr_train \
    --model_name_or_path "$DEFENDER_MODEL" \
    --attacker_model_name_or_path "$ATTACKER_MODEL" \
    --dataset_name "datasets/attack_target/train_attack_target.json" \
    --output_dir "./outputs/rlvr_mtsa_defence" \
    --defence_mode True \
    --use_peft True \
    --load_in_4bit True \
    --mini_batch_size $BATCH_SIZE \
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
