import os
from modal import App, Image, Mount, Volume, gpu

# Define the Modal App
app = App("mtsa-rlvr-training")

# Define the persistent volumes
# One for model weights/checkpoints
model_volume = Volume.from_name("mtsa-models", create_if_missing=True)
# One for datasets if not mounted locally
data_volume = Volume.from_name("mtsa-datasets", create_if_missing=True)

# Define the container image
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("/workspace/MTSA/requirements.txt")
    .pip_install("modal")
    # Add any extra dependencies needed for Modal environment
    .env({
        "PYTHONPATH": "/workspace/MTSA",
        "WANDB_PROJECT": "MTSA-RLVR-Modal",
    })
)

# Define Mounts for the code
# We mount the MTSA directory to /workspace/MTSA in the container
mtsa_mount = Mount.from_local_dir(
    "../MTSA", 
    remote_path="/workspace/MTSA",
    condition=lambda p: not p.startswith(".git") and "outputs" not in p and "__pycache__" not in p
)

@app.function(
    image=image,
    mounts=[mtsa_mount],
    volumes={"/models": model_volume, "/data": data_volume},
    gpu=gpu.A100(count=1), # Default to A100, can be overridden
    timeout=86400, # 24 hours
)
def run_rlvr_attack(model_path="Qwen/Qwen2.5-7B-Instruct", dataset_path="datasets/attack_target/train_attack_target.json"):
    import subprocess
    
    # Ensure dataset path is relative to /workspace/MTSA or /data
    if not dataset_path.startswith("/"):
        full_dataset_path = f"/workspace/MTSA/{dataset_path}"
    else:
        full_dataset_path = dataset_path
        
    output_dir = "/models/rlvr_attack"
    os.makedirs(output_dir, exist_ok=True)
    
    # We use python -m src.algorithm.mt_rlvr_train directly
    cmd = [
        "python", "-m", "src.algorithm.mt_rlvr_train",
        "--model_name_or_path", model_path,
        "--dataset_name", full_dataset_path,
        "--output_dir", output_dir,
        "--adv_estimator", "grpo",
        "--use_kl_in_reward", "true",
        "--kl_coef", "0.001",
        "--num_rollouts", "4",
        "--max_prompt_length", "320",
        "--max_response_length", "1024",
        "--ppo_epochs", "1",
        "--per_device_train_batch_size", "4",
        "--learning_rate", "1e-6",
        "--num_train_epochs", "1",
        "--save_freq", "100",
        "--logging_steps", "10",
        "--defence_mode", "false",
        "--use_entropy_reward", "false"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace/MTSA")

@app.function(
    image=image,
    mounts=[mtsa_mount],
    volumes={"/models": model_volume, "/data": data_volume},
    gpu=gpu.A100(count=1),
    timeout=86400,
)
def run_rlvr_defence(model_path="Qwen/Qwen2.5-7B-Instruct", dataset_path="datasets/attack_target/train_attack_target.json"):
    import subprocess
    
    if not dataset_path.startswith("/"):
        full_dataset_path = f"/workspace/MTSA/{dataset_path}"
    else:
        full_dataset_path = dataset_path
        
    output_dir = "/models/rlvr_defence"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python", "-m", "src.algorithm.mt_rlvr_train",
        "--model_name_or_path", model_path,
        "--dataset_name", full_dataset_path,
        "--output_dir", output_dir,
        "--adv_estimator", "grpo",
        "--use_kl_in_reward", "true",
        "--kl_coef", "0.001",
        "--num_rollouts", "4",
        "--max_prompt_length", "320",
        "--max_response_length", "1024",
        "--ppo_epochs", "1",
        "--per_device_train_batch_size", "4",
        "--learning_rate", "1e-6",
        "--num_train_epochs", "1",
        "--save_freq", "100",
        "--logging_steps", "10",
        "--defence_mode", "true",
        "--use_entropy_reward", "true"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace/MTSA")

@app.local_entrypoint()
def main(mode="attack", model="Qwen/Qwen2.5-7B-Instruct"):
    if mode == "attack":
        run_rlvr_attack.remote(model)
    elif mode == "defence":
        run_rlvr_defence.remote(model)
    else:
        print(f"Unknown mode: {mode}")
