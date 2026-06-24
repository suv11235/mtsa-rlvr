#!/bin/bash
# Usage: ./script/deploy/deploy_to_runpod.sh <ip_address> [ssh_port]

REMOTE_IP=$1
SSH_PORT=${2:-22}
PEM_KEY="$HOME/.ssh/id_ed25519_runpod"

if [ -z "$REMOTE_IP" ]; then
  echo "Usage: ./script/deploy/deploy_to_runpod.sh <ip_address> [ssh_port]"
  echo "Example: ./script/deploy/deploy_to_runpod.sh 1.2.3.4 2222"
  exit 1
fi

SSH_USER="root"
SSH_OPTS_BASE="-p $SSH_PORT -i $PEM_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes"
SSH_CMD="ssh $SSH_OPTS_BASE"

echo "========================================================"
echo "Deploying MTSA to RunPod ($SSH_USER@$REMOTE_IP:$SSH_PORT)..."
echo "Using key: $PEM_KEY"
echo "========================================================"

# Create directory first
echo "Creating remote directory /workspace/mtsa-rlvr ..."
$SSH_CMD -o ConnectTimeout=10 $SSH_USER@$REMOTE_IP "mkdir -p /workspace/mtsa-rlvr"
if [ $? -ne 0 ]; then
    echo "Error: Failed to connect to RunPod. Please check IP/Port and ensure your key is added to the pod."
    exit 1
fi

# Sync MTSA folder
# Excludes large artifacts/caches to speed up transfer
echo "Syncing files to remote..."
rsync -avz --progress \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude 'venv' \
  --exclude 'wandb' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'output' \
  --exclude 'outputs' \
  --exclude '*.pth' \
  --exclude '*.pt' \
  --exclude 'presentation' \
  -e "ssh $SSH_OPTS_BASE" \
  MTSA/ $SSH_USER@$REMOTE_IP:/workspace/mtsa-rlvr/MTSA/

# Optional: Sync adv_grpo if it exists and is outside MTSA
if [ -d "adv_grpo" ]; then
    echo "Syncing adv_grpo to remote..."
    rsync -avz --progress \
      --exclude '__pycache__' \
      -e "ssh $SSH_OPTS_BASE" \
      adv_grpo/ $SSH_USER@$REMOTE_IP:/workspace/mtsa-rlvr/adv_grpo/
fi

echo ""
echo "========================================================"
echo "Deployment Complete!"
echo "========================================================"
echo "To connect to your instance run:"
echo "ssh -p $SSH_PORT -i $PEM_KEY root@$REMOTE_IP"
echo ""
echo "To setup the environment on the pod:"
echo "cd /workspace/mtsa-rlvr/MTSA && bash script/deploy/setup_runpod.sh"
echo "========================================================"
