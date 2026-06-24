# MTSA-RLVR Workspace Docs

This repository is a **workspace** that collects several related projects around:
- **MTSA** (Multi-Turn Safety Alignment) + **RLVR** training/evaluation
- **Tamper resistance** / simulated weight-space attacks (TAR)
- A separate **Token Buncher** codebase (`rlvr-safety/`)
- Prototype experiments (e.g. `adv_grpo/`) and infra helpers (SLURM, Modal)

## Where to Start

- **I want the main MTSA + multi-turn RLVR code** → `MTSA/README.md`
  - Training entrypoint: `MTSA/src/algorithm/mt_rlvr_train.py` (run via `python -m src.algorithm.mt_rlvr_train`)
  - Common scripts: `MTSA/script/slurm/`
  - Eval entrypoints: `MTSA/src/eval/`
- **I want the “Token Buncher” project** → `rlvr-safety/README.md`
- **I want the ICLR’25 TAR reference implementation** → `tamper-resistance-repo/README.md`
- **I want the slide deck** → `presentation/README.md`
- **I want the adv-GRPO prototype** → `adv_grpo/README.md`
- **I want to run on Lambda** → `.agent/workflows/lambda_setup.md`
- **I want to run on Modal** → `modal_apps/README.md`

## MTSA Quickstart (Local)

From the workspace root:

```bash
cd MTSA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set required credentials as **environment variables** (recommended) or via an untracked `MTSA/.env`:
- `HF_TOKEN` (needed for gated models like Llama)
- `WANDB_API_KEY` (optional; disable via `WANDB_MODE=disabled`)

Security note: keep tokens **out of git** (both the workspace and `MTSA/` ignore `.env` by default).

Sanity check the entrypoint/options:

```bash
cd MTSA
python -m src.algorithm.mt_rlvr_train --help
```

## Common MTSA Workflows

### RLVR defence training (script wrapper)

```bash
cd MTSA
bash script/slurm/run_rlvr_defence.sh "Qwen/Qwen2.5-7B-Instruct" "datasets/attack_target/train_attack_target.json" "./outputs/rlvr_defence"
```

### RLVR attack training (script wrapper)

```bash
cd MTSA
bash script/slurm/run_rlvr_attack.sh "Qwen/Qwen2.5-7B-Instruct" "datasets/attack_target/train_attack_target.json" "./outputs/rlvr_attack"
```

### Adaptive multi-turn evaluation (baseline + checkpoint)

```bash
cd MTSA
python -m src.eval.eval_safety_adaptive \
  --baseline_model "meta-llama/Llama-3.1-8B-Instruct" \
  --checkpoint_path "./outputs/rlvr_defence" \
  --dataset_path "datasets/attack_target/train_attack_target_labels.json"
```

### Capability-regularized adversarial training (GSM8K control)

Adds a supervised NLL term on a control dataset (default: `openai/gsm8k`) to preserve math capability during adversarial/RLVR training.

Example (starting from a TAR checkpoint):

```bash
cd MTSA
python -m src.algorithm.mt_rlvr_train \
  --model_name_or_path "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2" \
  --tokenizer_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --attacker_model_name_or_path "suv11235/red_team_model_SFT_mtsa" \
  --judge_model_name_or_path "qylu4156/strongreject-15k-v1" \
  --judge_type "strongreject" \
  --dataset_name "datasets/attack_target/biosecurity_goals.json" \
  --defence_mode True \
  --use_capability_regularizer True \
  --capability_weight 0.05 \
  --capability_answer_mode final
```

### CSCS Alps (SLURM + Pyxis container)

`MTSA/script/slurm/submit_adv_training_cscs.slurm` shows a working pattern for running inside a Pyxis container.

```bash
cd MTSA
sbatch script/slurm/submit_adv_training_cscs.slurm
```

### Modal (managed GPUs)

See `modal_apps/README.md` for prerequisites and invocation. The entrypoint is `modal_apps/training.py`.

## Tests (MTSA)

Many tests in `MTSA/tests/` are **logic-flow/unit** tests that mock heavy dependencies.

```bash
cd MTSA
pytest -q
```

## Repo Map (Top Level)

- `MTSA/`: primary MTSA + RLVR implementation (paper code + SLURM tooling)
- `adv_grpo/`: prototype GRPO variant with vLLM-in-subprocess rollouts
- `rlvr-safety/`: separate Token Buncher codebase (own install/run scripts)
- `tamper-resistance-repo/`: reference TAR implementation (upstream-style repo)
- `modal_apps/`: run MTSA training on Modal GPUs
- `presentation/`: React/Vite slide deck for the project
- `EXPERIMENT_LOG.md`: chronological experiment/eval log
- `PROGRESS.md`: roadmap and progress notes
