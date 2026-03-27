## 🏗️ Repository Layout & Functionalities

### 1. [MTSA/](file:///Users/suvajitmajumder/mtsa-rlvr/MTSA/) (Core Framework)
The primary production-grade environment for **Multi-Turn Safety Alignment**.
*   **`src/algorithm/`**: Contains `mt_rlvr_train.py`, the main RLVR training loop.
*   **`src/rlvr/`**: The "brain" of the operation. Handles multi-turn environment simulation where an **Attacker** and **Victim** interact.
*   **`src/rlvr/reward_manager/`**: Implementation of verifiable rewards. Includes `multiturn_reward.py` for StrongREJECT/Llama-Guard scoring and entropy-based defense signals.
*   **`script/slurm/`**: Production deployment scripts for CSCS Alps (GH200) and Lambda Labs (H100).
*   **`src/utils/`**: General utilities for model loading, dataset processing, and result evaluation.

### 2. [adv_grpo/](file:///Users/suvajitmajumder/mtsa-rlvr/adv_grpo/) (Adversarial Research)
Experimental implementation of **Adversarial Group Relative Policy Optimization (GRPO)**.
*   Used for testing more efficient RL algorithms (GRPO vs PPO) specifically for red-teaming tasks.
*   Contains a lightweight standalone trainer and an adversarial simulation logic (`tar.py`).

### 3. [rlvr-safety/](file:///Users/suvajitmajumder/mtsa-rlvr/rlvr-safety/) (Token Research)
Specialized codebase for **"Token Bunching"** and advanced safety filtering research.
*   Focuses on how tokenization-level manipulations can affect safety guardrails.
*   Includes a separate `verl` installation and dedicated transfer-to-lambda workflows.

### 4. [tamper-resistance-repo/](file:///Users/suvajitmajumder/mtsa-rlvr/tamper-resistance-repo/) (TAR Reference)
The reference implementation for **Tamper-Resistant** safeguards.
*   Crucial for benchmarking how resilient a model is to direct payload tampering.
*   Contains core datasets and objective functions used to validate TAR v1/v2.

### 5. [presentation/](file:///Users/suvajitmajumder/mtsa-rlvr/presentation/) & [docs/](file:///Users/suvajitmajumder/mtsa-rlvr/docs/)
*   **`presentation/`**: A React/Vite-based slide deck for reporting project findings.
*   **`docs/`**: Centralized onboarding pointers and technical deep-dives to keep the monorepo navigable.

---

## 🛠️ Monorepo Best Practices (Research Focus)

Following patterns from **DeepSeek-Style** and **Google Research** monorepos:

### ♻️ Common Functionalities & Repurposing
Several "Primitives" are currently shared or duplicated across folders. To improve velocity:
1.  **Payload Extraction**: Logic for stripping `<think>` tags and identifying `### Attack Payload` should be centralized in a shared `libs/`.
2.  **Reward Primitives**: Entropy-based rewards and Cosine Similarity are used in both `MTSA` and `adv_grpo`. These are candidate for a unified `reward_utils.py`.
3.  **Judge Interface**: Standardize the **StrongREJECT** and **HarmBench** judge prompts. Instead of hardcoding them in `multiturn_reward.py`, they should be loaded from a shared `configs/judges/` registry.

### 📈 Scalable Research Workflow
*   **Environment Parity**: Always use the `.env` pattern at the root for `HF_TOKEN` and `WANDB_API_KEY` to ensure scripts work across all sub-folders.
*   **Checkpoint Standard**: Models trained in `adv_grpo` or `rlvr-safety` should follow the same LoRA/Safetensors format to be evaluated by the `MTSA/src/eval/` suite.
*   **Data Factory**: Centralize dataset preparation (e.g., `biosecurity_goals.json`) to prevent version mismatch between different training algorithms.

---

## 🚀 Getting Started

**Main Entry Point**:
```bash
cd MTSA
# Follow instructions in MTSA/README.md
```

**Viewing Results**:
```bash
# Check the canonical log of all experiments
cat EXPERIMENT_LOG.md
```

For cluster-specific setup:
1. Refer to the [CSCS Setup Guide](MTSA/README.md#2-cscs-cluster-deployment--training-pyxis).
2. Ensure you have the [sshservice-cli](MTSA/sshservice-cli/) configured for MFA/MTSA access.
