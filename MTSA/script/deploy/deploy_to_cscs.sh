#!/bin/bash
# Usage: ./script/deploy_to_cscs.sh <username>

USERNAME=$1

if [ -z "$USERNAME" ]; then
  echo "Usage: ./script/deploy_to_cscs.sh <cscs_username>"
  echo "Example: ./script/deploy_to_cscs.sh smajumder"
  exit 1
fi

SSH_USER="$USERNAME"
REMOTE_HOST="ela" # This is the entry point for CSCS, which we proxyjump through to clariden
SSH_CMD="ssh"

echo "========================================================"
echo "Deploying MTSA to CSCS ($SSH_USER@$REMOTE_HOST)..."
echo "========================================================"

# Create directory first
echo "Creating remote directory..."
$SSH_CMD -o ConnectTimeout=10 $SSH_USER@$REMOTE_HOST "mkdir -p ~/mtsa-rlvr"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create remote directory. Please check your SSH config and try again."
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
  . $SSH_USER@$REMOTE_HOST:~/mtsa-rlvr/MTSA/

echo ""
echo "========================================================"
echo "Deployment Complete!"
echo "========================================================"
echo "To connect to your instance run:"
echo "ssh $SSH_USER@$REMOTE_HOST"
echo "========================================================"
