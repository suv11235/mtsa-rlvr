#!/bin/bash
#SBATCH --job-name=adv-training-cscs
#SBATCH --account=aa004
#SBATCH --constraint=gpu                     # Use GPU nodes
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --gpus-per-node=4                    # CSCS runs 4 x GH200 per node
#SBATCH --cpus-per-task=16                   # Align with your workload
#SBATCH --mem=128G                           # Memory request
#SBATCH --time=10:00:00                      # Job time limit (normal limits up to 24h)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

# Change into the experiment base
cd ~/mtsa-rlvr/MTSA

# ----------------------------------------------------
# ENVIRONMENT VARIABLES
# ----------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# IMPORTANT: Create a .env file on CSCS with your HF_TOKEN and WANDB keys
export $(grep -v '^#' .env | xargs)

# WandB Configuration
export WANDB_PROJECT="mtsa-rlvr-training"
export WANDB_NAME="adv_training_cscs_zero2_podman"

# Use CSCS $SCRATCH for I/O and caching
export OUTPUT_DIR="$SCRATCH/mtsa_outputs/adv_training_lora_3turn"
export HF_HOME="$SCRATCH/huggingface_cache"
mkdir -p $OUTPUT_DIR
mkdir -p $HF_HOME

# ----------------------------------------------------
# NATIVE PYXIS CONTAINER EXECUTION
# ----------------------------------------------------
# We pull the highly optimized GH200 base image directly from DockerHub via Enroot
IMAGE_PATH="docker.io/drikster80/vllm-gh200-openai" 

# Run the training through srun executing natively inside the fetched container
srun \
    --container-image="$IMAGE_PATH" \
    --container-mounts="$SCRATCH:$SCRATCH,$(pwd):/workspace/MTSA" \
    --container-workdir="/workspace/MTSA" \
    --container-remap-root \
    bash -c 'export DS_BUILD_OPS=0 && pip install wandb "accelerate>=0.34.0" "deepspeed>=0.14.0" --no-build-isolation && \
    accelerate launch \
        --config_file script/accelerate_configs/zero2.yaml \
        --num_processes 3 \
        src/algorithm/mt_rlvr_train.py \
        --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
        --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
        --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
        --judge_model_name_or_path "qylu4156/strongreject-15k-v1" \
        --judge_type "strongreject" \
        --dataset_name datasets/attack_target/biosecurity_goals.json \
        --output_dir $OUTPUT_DIR \
        --defence_mode True \
        --use_peft True \
        --load_in_4bit False \
        --bf16 True \
        --per_device_train_batch_size 1 \
        --mini_batch_size 1 \
        --num_rollouts 1 \
        --num_train_epochs 1 \
        --warmup_steps 0 \
        --learning_rate 1e-6 \
        --ppo_epochs 1 \
        --max_response_length 512 \
        --use_tamper_resistance False \
        --judge_reward_weight 1.0 \
        --entropy_reward_weight 0.0 \
        --max_sim_turns 3 \
        --gradient_checkpointing True \
        --attn_implementation sdpa \
        --save_steps 10 \
        --logging_steps 1 \
        --cache_dir $HF_HOME \
        --use_vllm True \
        --vllm_distribution_strategy split_ranks'
