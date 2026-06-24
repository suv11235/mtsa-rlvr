#!/bin/bash
# Local Helper to sync code to Vast.ai
# Usage: bash sync_to_vast.sh <IP> <PORT>

IP=$1
PORT=$2

if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Usage: bash sync_to_vast.sh <IP> <PORT>"
    exit 1
fi

echo "Syncing to Vast.ai ($IP:$PORT)..."

rsync -avz --progress \
  -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude 'venv' \
  --exclude 'wandb' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'output' \
  --exclude 'outputs' \
  MTSA/ root@$IP:/workspace/mtsa-rlvr/MTSA/

echo "Done. You can now log in and run setup:"
echo "ssh -p $PORT root@$IP"
echo "cd /workspace/mtsa-rlvr/MTSA && bash script/deploy/setup_vastai.sh"
