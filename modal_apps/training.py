import os
from pathlib import Path
from modal import App, Image, Volume, gpu, Secret
from modal.mount import Mount

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
    .add_local_dir(Path(__file__).parent.parent / "MTSA", remote_path="/workspace/MTSA", copy=True)
    .pip_install_from_requirements(Path(__file__).parent.parent / "MTSA" / "requirements.txt")
    .pip_install("modal", "deepspeed")
    # Add any extra dependencies needed for Modal environment
    .env({
        "PYTHONPATH": "/workspace/MTSA",
        "WANDB_PROJECT": "MTSA-RLVR-Modal",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
)

@app.function(
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("wandb")],
    gpu="A100-80GB:2",
    timeout=86400, # 24 hours
)
def run_rlvr_attack(model_path="Qwen/Qwen2.5-7B-Instruct", dataset_path="datasets/attack_target/train_attack_target.json", victim_path="Qwen/Qwen2.5-7B-Instruct", use_peft=True, zero_stage=0, dry_run=False, use_tamper_resistance=False):
    import subprocess
    
    # Ensure dataset path is relative to /workspace/MTSA or /data
    if not dataset_path.startswith("/"):
        full_dataset_path = f"/workspace/MTSA/{dataset_path}"
    else:
        full_dataset_path = dataset_path
        
    output_dir = "/models/rlvr_attack"
    os.makedirs(output_dir, exist_ok=True)
    
    # We use accelerate launch for multi-GPU support
    args = [
        "--model_name_or_path", model_path,
        "--dataset_name", full_dataset_path,
        "--output_dir", output_dir,
        "--adv_estimator", "grpo",
        "--use_kl_in_reward", "true",
        "--kl_coef", "0.001",
        "--num_rollouts", "4",
        "--max_prompt_length", "320",
        "--max_response_length", "2048",
        "--ppo_epochs", "1",
        "--temperature", "1.0",
        "--per_device_train_batch_size", "2",
        "--learning_rate", "1e-6",
        "--num_train_epochs", "1",
        "--save_steps", "100",
        "--logging_steps", "10",
        "--defence_mode", "false",
        "--attack_mode", "true",
        "--victim_model_name_or_path", victim_path,
        "--use_entropy_reward", "false",
        "--use_peft", str(use_peft).lower(),
        "--attn_implementation", "sdpa",
        "--use_tamper_resistance", str(use_tamper_resistance).lower(),
    ]
    
    if dry_run:
        args.append("--dry_run")
    
    # Enable DeepSpeed if not using PEFT or if stage > 0
    if not use_peft or zero_stage > 0:
        launch_cmd = [
            "accelerate", "launch",
            "--config_file", f"/workspace/MTSA/script/accelerate_configs/zero{zero_stage}.yaml",
            "--num_processes", "2",
        ]
    else:
        launch_cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", "2",
            "--mixed_precision", "bf16",
        ]
    
    cmd = launch_cmd + ["-m", "src.algorithm.mt_rlvr_train"] + args
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace/MTSA")

@app.function(
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("wandb")],
    gpu="A100-80GB:2",
    timeout=86400,
)
def run_rlvr_defence(model_path="Qwen/Qwen2.5-7B-Instruct", dataset_path="datasets/attack_target/train_attack_target.json", use_peft=True, zero_stage=0, dry_run=False, use_tamper_resistance=False):
    import subprocess
    
    if not dataset_path.startswith("/"):
        full_dataset_path = f"/workspace/MTSA/{dataset_path}"
    else:
        full_dataset_path = dataset_path
        
    output_dir = "/models/rlvr_defence"
    os.makedirs(output_dir, exist_ok=True)
    
    # We use accelerate launch for multi-GPU support
    args = [
        "--model_name_or_path", model_path,
        "--dataset_name", full_dataset_path,
        "--output_dir", output_dir,
        "--adv_estimator", "grpo",
        "--use_kl_in_reward", "true",
        "--kl_coef", "0.001",
        "--num_rollouts", "4",
        "--max_prompt_length", "320",
        "--max_response_length", "2048",
        "--ppo_epochs", "1",
        "--temperature", "1.0",
        "--per_device_train_batch_size", "2",
        "--learning_rate", "1e-6",
        "--num_train_epochs", "1",
        "--save_steps", "100",
        "--logging_steps", "10",
        "--defence_mode", "true",
        "--use_entropy_reward", "true",
        "--use_peft", str(use_peft).lower(),
        "--attn_implementation", "sdpa",
        "--use_tamper_resistance", str(use_tamper_resistance).lower(),
    ]
    
    if dry_run:
        args.append("--dry_run")
    
    # Enable DeepSpeed if not using PEFT or if stage > 0
    if not use_peft or zero_stage > 0:
        launch_cmd = [
            "accelerate", "launch",
            "--config_file", f"/workspace/MTSA/script/accelerate_configs/zero{zero_stage}.yaml",
            "--num_processes", "2",
        ]
    else:
        launch_cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", "2",
            "--mixed_precision", "bf16",
        ]
    
    cmd = launch_cmd + ["-m", "src.algorithm.mt_rlvr_train"] + args
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace/MTSA")

@app.local_entrypoint()
def main(mode="attack", model="Qwen/Qwen2.5-7B-Instruct", dataset="datasets/attack_target/train_attack_target.json", victim="Qwen/Qwen2.5-7B-Instruct", peft: bool = True, zero: int = 0, dry_run: bool = False, tamper: bool = False):
    if mode == "attack":
        run_rlvr_attack.remote(model, dataset_path=dataset, victim_path=victim, use_peft=peft, zero_stage=zero, dry_run=dry_run, use_tamper_resistance=tamper)
    elif mode in ["defence", "defense"]:
        run_rlvr_defence.remote(model, dataset_path=dataset, use_peft=peft, zero_stage=zero, dry_run=dry_run, use_tamper_resistance=tamper)
    else:
        print(f"Unknown mode: {mode}")
