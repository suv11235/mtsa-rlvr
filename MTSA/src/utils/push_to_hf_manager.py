from huggingface_hub import HfApi
import os

def create_model_card(repo_id, experiment_name, params, dataset):
    card = f"""---
license: apache-2.0
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- alignment
- safety
- tamper-resistance
- rlvr
- mtsa
---

# {repo_id.split('/')[-1]}

This model is a LoRA adapter for `meta-llama/Llama-3.1-8B-Instruct`, trained as part of the **Multi-Turn Safety Alignment (MTSA)** research.

## Experiment Description
**Experiment**: {experiment_name}
This checkpoint was trained using the MTSA-RLVR framework, which combines Multi-Turn Reinforcement Learning from Human Feedback (RLHF) with Tamper Resistance (TAR) to produce safeguards that are robust to both input-space jailbreaks and weight-space fine-tuning attacks.

## Training Details
- **Base Model**: Llama-3.1-8B-Instruct
- **Dataset**: `{dataset}`
- **Methodology**: Multi-Turn RLVR + Tamper Resistance (Inner Loop)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
{chr(10).join([f"| {k} | {v} |" for k, v in params.items()])}

## Usage
To use this adapter, load it using `peft`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype='auto', device_map='auto')
model = PeftModel.from_pretrained(model, "{repo_id}")
```

## Citation
If you use this model in your research, please cite the MTSA workshop paper.
"""
    return card

def push_checkpoints():
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    
    checkpoints = [
        {
            "repo_id": "suv11235/mtsa-extreme-tar-v1-llama-3.1-8b",
            "folder": "outputs/rlvr_mtsa_extreme_tar/checkpoint-75",
            "experiment": "Extreme Tamper Resistance v1",
            "params": {
                "Outer Learning Rate": "5e-6",
                "Inner Learning Rate": "2e-5",
                "Inner Loop Steps": "64",
                "TAR Type": "Adversarial SFT (Attack)",
                "Simulation Turns": "3"
            },
            "dataset": "datasets/attack_target/train_attack_target_labels.json (9.6k samples)"
        },
        {
            "repo_id": "suv11235/mtsa-rlvr-scaling-baseline-llama-3.1-8b",
            "folder": "outputs/rlvr_mtsa_long_run_v4gpu/checkpoint-110",
            "experiment": "Standard MT-RLVR Scaling Run",
            "params": {
                "Outer Learning Rate": "7e-6",
                "Inner Learning Rate": "5e-5",
                "Inner Loop Steps": "1",
                "TAR Type": "Adversarial SFT (Attack)",
                "Simulation Turns": "3"
            },
            "dataset": "datasets/attack_target/train_attack_target_labels.json"
        },
        {
            "repo_id": "suv11235/vanilla-tar-baseline-llama-3.1-8b",
            "folder": "outputs/tar_vanilla_baseline/checkpoint-50",
            "experiment": "Vanilla TAR Baseline (Paper Reproduction)",
            "params": {
                "Outer Learning Rate": "1e-5",
                "Inner Learning Rate": "1e-4",
                "Inner Loop Steps": "1",
                "TAR Type": "Entropy Maximization",
                "Method": "SFT-based Meta-Learning"
            },
            "dataset": "datasets/attack_target/train_attack_target_labels.json"
        },
        {
            "repo_id": "suv11235/mtsa-rlvr-representation-loss-llama-3.1-8b",
            "folder": "outputs/rlvr_mtsa_rep_loss/checkpoint-75",
            "experiment": "MT-RLVR with Representation Loss",
            "params": {
                "Outer Learning Rate": "7e-6",
                "Inner Learning Rate": "5e-5",
                "Inner Loop Steps": "1",
                "Simulation Turns": "3",
                "Loss Function": "Latent Representation Shift"
            },
            "dataset": "datasets/attack_target/train_attack_target_labels.json"
        }
    ]

    for ckpt in checkpoints:
        print(f"\n>>> Processing {ckpt['repo_id']}...")
        folder_path = os.path.expanduser(f"~/mtsa-rlvr/MTSA/{ckpt['folder']}")
        
        # 1. Create Model Card
        readme_content = create_model_card(ckpt['repo_id'], ckpt['experiment'], ckpt['params'], ckpt['dataset'])
        readme_path = os.path.join(folder_path, "README.md")
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Generated model card at {readme_path}")

        # 2. Push to Hub
        try:
            api.create_repo(repo_id=ckpt['repo_id'], private=False, exist_ok=True)
            print(f"Uploading folder to {ckpt['repo_id']}...")
            api.upload_folder(
                folder_path=folder_path,
                repo_id=ckpt['repo_id'],
                ignore_patterns=["checkpoint-*", "global_step*", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "training_state.pt"]
            )
            print(f"✅ Successfully pushed {ckpt['repo_id']}")
        except Exception as e:
            print(f"❌ Error pushing {ckpt['repo_id']}: {e}")

if __name__ == "__main__":
    push_checkpoints()
