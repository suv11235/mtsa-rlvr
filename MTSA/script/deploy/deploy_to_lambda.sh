#!/bin/bash
# Usage: ./script/deploy/deploy_to_lambda.sh <ip_address> [pem_key_path]

REMOTE_IP=$1
PEM_KEY=$2

if [ -z "$REMOTE_IP" ]; then
  echo "Usage: ./script/deploy/deploy_to_lambda.sh <ip_address> [pem_key_path]"
  exit 1
fi

SSH_USER="ubuntu"
SSH_CMD="ssh"

if [ ! -z "$PEM_KEY" ]; then
    chmod 400 "$PEM_KEY"
    SSH_CMD="ssh -i $PEM_KEY"
fi

echo "========================================================"
echo "Deploying MTSA to Lambda ($SSH_USER@$REMOTE_IP)..."
echo "========================================================"

# Create directory first
echo "Creating remote directory..."
$SSH_CMD -o ConnectTimeout=10 $SSH_USER@$REMOTE_IP "mkdir -p ~/mtsa-rlvr"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create remote directory. Please check your connection and try again."
    exit 1
fi

# Sync MTSA folder
# Excludes large artifacts/caches to speed up transfer
echo "Syncing file to remote..."
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
  -e "$SSH_CMD" \
  MTSA/ $SSH_USER@$REMOTE_IP:~/mtsa-rlvr/MTSA/

echo ""
echo "========================================================"
echo "Deployment Complete!"
echo "========================================================"
echo "To connect to your instance run:"
if [ ! -z "$PEM_KEY" ]; then
    echo "ssh -i $PEM_KEY $SSH_USER@$REMOTE_IP"
else
    echo "ssh $SSH_USER@$REMOTE_IP"
fi
echo "========================================================"
