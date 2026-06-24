#!/bin/bash
# Vast.ai Setup Script for MTSA-RLVR
# This script installs dependencies, sets up the venv, and configures the environment.

set -e

# Vast.ai typically mounts at /workspace
WORKSPACE_DIR="/workspace"
PROJECT_DIR="$WORKSPACE_DIR/mtsa-rlvr/MTSA"

echo "Initializing Vast.ai environment..."

# 1. Install System Dependencies
# Note: Vast.ai containers are usually root.
apt-get update
apt-get install -y build-essential python3-dev python3-venv git git-lfs screen htop

# 2. Setup Project Directory
mkdir -p "$WORKSPACE_DIR/outputs"
mkdir -p "$WORKSPACE_DIR/.cache"

# 3. Create Virtual Environment
cd "$PROJECT_DIR"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 4. Install Requirements
echo "Installing dependencies (this may take a while)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install vllm==0.5.4 # Specific version verified for stability

# 5. Fast-track DeepSpeed/vLLM Compatibility
# Fabric Manager is usually handled by the Vast host, but we ensure it's healthy
if command -v systemctl &> /dev/null; then
    systemctl start nvidia-fabricmanager || true
fi

# 6. Configure Accelerate
# Creating a default config for 8x GPUs
mkdir -p ~/.cache/huggingface/accelerate
cat <<EOT > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT

echo "Setup complete! You can now run: bash script/vastai/launch_vastai_lora.sh"
