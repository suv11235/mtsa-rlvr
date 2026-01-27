import os
from pathlib import Path
from modal import App, Image, Volume, gpu, Secret
from modal.mount import Mount

# Define the Modal App
app = App("mtsa-rlvr-tar-training")

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
    .pip_install("modal")
    # Add any extra dependencies needed for Modal environment
    .env({
        "PYTHONPATH": "/workspace/MTSA",
        "WANDB_PROJECT": "MTSA-RLVR-TAR-Modal",
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
def run_mtsa_tar_defense(
    model_path="Qwen/Qwen2.5-7B-Instruct", 
    dataset_path="datasets/attack_target/train_attack_target.json",
    attacker_model_path=None,
    num_rollouts=4,
    tar_steps=4,
    tar_lr=5e-5,
    use_peft=True,
    zero_stage=0,
    dry_run=False
):
    import subprocess
    
    # Ensure dataset path is relative to /workspace/MTSA or /data
    if not dataset_path.startswith("/"):
        full_dataset_path = f"/workspace/MTSA/{dataset_path}"
    else:
        full_dataset_path = dataset_path
        
    output_dir = "/models/rlvr_tar_defence"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command for MTSA + TAR + Tokenbuncher (Entropy Defense)
    # We use accelerate launch for multi-GPU support
    args = [
        "--model_name_or_path", model_path,
        "--dataset_name", full_dataset_path,
        "--output_dir", output_dir,
        "--adv_estimator", "grpo",
        "--use_kl_in_reward", "true",
        "--kl_coef", "0.001",
        "--num_rollouts", str(num_rollouts),
        "--max_prompt_length", "320",
        "--max_response_length", "1024",
        "--ppo_epochs", "1",
        "--per_device_train_batch_size", "2",
        "--learning_rate", "1e-6",
        "--num_train_epochs", "1",
        "--save_steps", "100",
        "--logging_steps", "10",
        "--defence_mode", "true",
        "--use_entropy_reward", "true",
        "--use_tamper_resistance", "true",
        "--tar_inner_loop_steps", str(tar_steps),
        "--tar_inner_lr", str(tar_lr),
        "--use_peft", str(use_peft).lower(),
        "--attn_implementation", "sdpa",
    ]
    
    if attacker_model_path:
        args.extend(["--attacker_model_name_or_path", attacker_model_path])
        
    if dry_run:
        args.append("--dry_run")

    # Enable DeepSpeed if not using PEFT or if stage > 0
    if not use_peft or zero_stage > 0:
        # Use DeepSpeed configuration
        launch_cmd = [
            "accelerate", "launch",
            "--config_file", f"/workspace/MTSA/script/accelerate_configs/zero{zero_stage}.yaml",
            "--num_processes", "2",
        ]
    else:
        # Standard Multi-GPU launch for PEFT
        launch_cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_processes", "2",
            "--mixed_precision", "bf16",
        ]
    
    cmd = launch_cmd + ["-m", "src.algorithm.mt_rlvr_train"] + args
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/workspace/MTSA")
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        raise RuntimeError("Training script failed")

@app.local_entrypoint()
def main(
    model="Qwen/Qwen2.5-7B-Instruct", 
    dataset="datasets/attack_target/train_attack_target.json",
    attacker=None,
    peft: bool = True,
    zero: int = 0,
    dry_run: bool = False
):
    """
    Main entry point for running MTSA + TAR + Tokenbuncher training on Modal.
    Usage: modal run modal/train_mt_rlvr_tar.py --model ... --dataset ... [--attacker ...] [--peft False] [--zero 3] [--dry-run]
    """
    print(f"Starting MTSA+TAR+Tokenbuncher Training on Modal")
    print(f"Target Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"PEFT: {peft}, ZeRO Stage: {zero}")
    if attacker:
        print(f"Attacker Model: {attacker}")
    if dry_run:
        print(">>> DRY RUN MODE ENABLED")
        
    run_mtsa_tar_defense.remote(
        model_path=model,
        dataset_path=dataset,
        attacker_model_path=attacker,
        use_peft=peft,
        zero_stage=zero,
        dry_run=dry_run
    )
