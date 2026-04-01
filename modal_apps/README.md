## Usage (Primary Pipelines)

Use `modal_apps/primary_pipelines.py` to run the official MTSA training loops.

### 1. MT-RLVR (Multi-Turn RL)
Runs the interactive Multi-Turn Safety Alignment loop with **vLLM** and **DeepSpeed**.

```bash
# Run Attack RLVR (default)
modal run modal_apps/primary_pipelines.py --pipeline rlvr --mode attack

# Run Defence RLVR
modal run modal_apps/primary_pipelines.py --pipeline rlvr --mode defence
```

### 2. Red-Team SFT
Runs the Supervised Fine-Tuning pipeline for the red-teaming (attacker) model.

```bash
modal run modal_apps/primary_pipelines.py --pipeline sft --model meta-llama/Meta-Llama-3-8B-Instruct
```

---

## 🏗️ Configuration & Architecture

- **GPU Acceleration**:
  - `rlvr`: Default `A100-80GB:2` (Required for colocated vLLM + Training).
  - `sft`: Default `H100:1`.
- **Volumes**:
  - `/models`: Shared volume (`mtsa-models`) for saving checkpoints.
  - `/data`: Shared volume (`mtsa-datasets`) for dataset storage and HuggingFace cache.
- **Code Mount**: The local `MTSA/` folder is automatically mirrored into the container at `/workspace/MTSA`, enabling live code updates without rebuilding the entire image.
- **Secrets**: Requires `huggingface-secret` (with `HF_TOKEN`) and `wandb` API keys to be configured in your Modal dashboard.
