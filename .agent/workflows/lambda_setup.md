---
description: Guide for setting up and running MTSA codebase on Lambda Labs GPU instances
---

# Running on Lambda Labs

Follow these steps to deploy and run the codebase on a remote Lambda Labs instance.

## 1. Prerequisites
- A running Lambda Labs instance (Ubuntu).
- The IP address of the instance (e.g., `192.168.1.1`).
- Your SSH key file (e.g., `~/.ssh/my-key.pem`) locally.

## 2. Deploy Code
Use the helper script to sync your local code to the remote server.

```bash
# From the root workspace directory
./MTSA/script/deploy/deploy_to_lambda.sh <INSTANCE_IP> <PATH_TO_PEM_KEY>
```
Example:
```bash
./MTSA/script/deploy/deploy_to_lambda.sh 204.45.12.34 ~/.ssh/lambda.pem
```

## 3. SSH and Setup Environment
Login to the node:
```bash
ssh -i ~/.ssh/lambda.pem ubuntu@<INSTANCE_IP>
```

Once logged in, run the setup commands:
```bash
cd ~/mtsa-rlvr/MTSA

# Run standard setup (works for Lambda too)
bash script/deploy/setup_runpod.sh
```

## 4. Launch Training
Once dependencies are installed, you can start your training jobs.

**Example: Defence Training (RLVR)**
```bash
# Don't forget to log in to HuggingFace if accessing gated models (like Llama 3)
huggingface-cli login

# Start training
bash script/slurm/run_rlvr_defence.sh "Qwen/Qwen2.5-7B-Instruct" "datasets/attack_target/train_attack_target.json"
```

## 5. Monitoring
To monitor the training from your local machine:
1.  **WandB**: Ensure you set `export WANDB_API_KEY=...` on the remote machine or login with `wandb login`.
2.  **Logs**: Tail logs via SSH: `tail -f ~/mtsa-rlvr/MTSA/outputs/rlvr_defence/training.log`.
