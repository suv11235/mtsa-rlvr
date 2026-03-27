# adv_grpo (Prototype)

`adv_grpo/` is a prototype implementation of an **adversarial GRPO-style loop** that:
- Generates rollouts via **vLLM** in a **separate subprocess** (to reduce NCCL/ZeRO interaction issues)
- Computes rewards via a simplified multi-turn simulation + judge scoring (see `adv_grpo/src/reward.py`)
- Trains either the **victim/defender** or the **attacker** (see `--train_target`)

This directory is **not** the main MTSA paper codepath; for the official framework use `MTSA/`.

## Install

`adv_grpo/` shares most dependencies with MTSA. A practical setup is:

```bash
cd MTSA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

From the workspace root:

```bash
python adv_grpo/main.py \
  --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
  --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
  --judge_model_name_or_path "qylu4156/strongreject-15k-v1" \
  --dataset_path "MTSA/datasets/attack_target/train_attack_target.json" \
  --train_target "victim" \
  --output_dir "adv_grpo/outputs"
```

Notes:
- `HF_TOKEN` may be required to download gated models.
- vLLM engines are started per-rank with a `port_offset`; see `adv_grpo/src/vllm_engine.py`.

