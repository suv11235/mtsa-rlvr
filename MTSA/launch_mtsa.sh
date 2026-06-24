#!/bin/bash
cd ~/mtsa-rlvr/MTSA
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_HOST_IP=127.0.0.1
export VLLM_USE_V1=0

# Clean up any old log
rm -f training_debug.log

accelerate launch \
    --config_file script/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    --main_process_port 29509 \
    src/algorithm/mt_rlvr_train.py \
    --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
    --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
    --judge_model_name_or_path "qylu4156/strongreject-15k-v1" \
    --judge_type "strongreject" \
    --dataset_name datasets/attack_target/biosecurity_goals.json \
    --output_dir outputs/llama3-8b-bio-lora \
    --defence_mode True \
    --use_peft True \
    --bf16 True \
    --learning_rate 1e-4 \
    --num_rollouts 16 \
    --max_sim_turns 3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --mini_batch_size 1 \
    --ppo_epochs 1 \
    --max_response_length 512 \
    --use_vllm True \
    --vllm_distribution_strategy split_ranks \
    --save_steps 50 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --gradient_checkpointing True > training_debug.log 2>&1
