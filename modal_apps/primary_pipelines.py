import os
from pathlib import Path
from modal import App, Image, Volume, gpu, Secret

# Define the Modal App
app = App("mtsa-primary-pipelines")

# Persistent Storage
# 1. Models and LoRA Checkpoints
model_volume = Volume.from_name("mtsa-models", create_if_missing=True)
# 2. Datasets and global cache
data_volume = Volume.from_name("mtsa-datasets", create_if_missing=True)

# Optimized GPU-ready Image
# Using a base image that already has CUDA and PyTorch saves massive build time
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch", 
        "transformers", 
        "accelerate", 
        "vllm>=0.6.0", 
        "deepspeed", 
        "peft", 
        "trl", 
        "wandb", 
        "datasets",
        "bitsandbytes",
        "scipy",
        "python-dotenv"
    )
    # Mount the local MTSA directory for source code access
    .add_local_dir(Path(__file__).parent.parent / "MTSA", remote_path="/workspace/MTSA")
    .env({
        "PYTHONPATH": "/workspace/MTSA",
        "WANDB_PROJECT": "MTSA-RLVR-Modal",
        "HF_HOME": "/data/huggingface_cache",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
)

# Shared configuration for GPU jobs
GPU_CONFIG = "A100-80GB:4" # Default to 4x A100 for RLVR
TIMEOUT = 86400 # 24 hours

@app.function(
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("wandb")],
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
)
def run_mt_rlvr_pipeline(
    mode="attack", 
    model="lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2", 
    dataset="datasets/attack_target/biosecurity_goals.json",
    num_processes=4,
    **kwargs
):
    """Primary RLVR Training Pipeline (Attack or Defense)"""
    import subprocess
    
    os.makedirs("/models/rlvr", exist_ok=True)
    os.makedirs("/data/huggingface_cache", exist_ok=True)
    
    # Base command
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(num_processes),
        "--mixed_precision", "bf16",
        "-m", "src.algorithm.mt_rlvr_train",
        "--model_name_or_path", model,
        "--dataset_name", f"/workspace/MTSA/{dataset}",
        "--output_dir", f"/models/rlvr/{mode}_run",
        "--per_device_train_batch_size", "1",
        "--mini_batch_size", "1",
        "--bf16", "true",
        "--use_vllm", "true",
        "--vllm_gpu_memory_utilization", "0.4" # Leave room for training
    ]
    
    if mode == "defence":
        cmd.extend(["--defence_mode", "true", "--use_entropy_reward", "true"])
    else:
        cmd.extend(["--attack_mode", "true", "--attacker_model_name_or_path", "suv11235/red_team_model_SFT_mtsa"])

    # Append any extra kwargs as flags
    for k, v in kwargs.items():
        cmd.extend([f"--{k}", str(v)])

    print(f"🚀 Launching MT-RLVR ({mode}) on Modal...")
    subprocess.run(cmd, check=True, cwd="/workspace/MTSA")

@app.function(
    image=image,
    volumes={"/models": model_volume, "/data": data_volume},
    secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("wandb")],
    gpu="H100:1", # SFT/DPO often faster on single H100
    timeout=TIMEOUT,
)
def run_red_team_sft(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset="datasets/red_team/sft_data.jsonl"
):
    """Red-Team SFT Pipeline (Supervised Fine-Tuning)"""
    import subprocess
    
    output_dir = "/models/red_team_sft"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python3", "-m", "src.algorithm.red_team_sft",
        "--model_name_or_path", model,
        "--dataset_name", f"/workspace/MTSA/{dataset}",
        "--output_dir", output_dir,
        "--per_device_train_batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-5",
        "--bf16", "true",
        "--use_peft", "true"
    ]
    
    print(f"🚀 Launching Red-Team SFT on Modal...")
    subprocess.run(cmd, check=True, cwd="/workspace/MTSA")

@app.local_entrypoint()
def main(pipeline="rlvr", mode="attack", model=None):
    if pipeline == "rlvr":
        params = {"mode": mode}
        if model: params["model"] = model
        run_mt_rlvr_pipeline.remote(**params)
    elif pipeline == "sft":
        run_red_team_sft.remote(model=model) if model else run_red_team_sft.remote()
    else:
        print(f"Unknown pipeline: {pipeline}. Options: rlvr, sft")
