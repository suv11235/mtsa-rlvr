# MTSA-RLVR Project Progress Report

This document outlines the development and progress of the **Multi-Turn Safety Alignment (MTSA)** framework, specifically focusing on the integration of **RLVR (Reinforcement Learning with Verifiable Rewards)** for adversarial attack and defense training on a single GPU (NVIDIA A100 80GB).

## 🚀 Key Accomplishments

### 1. Robust Single-GPU Pipeline Setup
- **QLoRA & PEFT Integration**: Configured the training scripts to use 4-bit quantization and LoRA adapters. This allows running multiple 7B models (Target, Attacker, Reference, and Judge) simultaneously on a single 80GB GPU.
- **Resource Management**: Implemented a strategic model caching system to bypass the 20GB disk quota on RunPod's `/root` and `/workspace` partitions by splitting model weights across both volumes.
- **Attention Optimization**: Switched to `sdpa` (Scaled Dot Product Attention) to enable high-performance training without the external `flash-attn` dependency.

### 2. Attacker Model Training (SFT Phase)
- **Objective**: Fine-tune a red-team model to transform abstract "harmful goals" into effective adversarial prompts.
- **Implementation**: Successfully executed the `red_team_sft.py` script to perform Supervised Fine-Tuning on a Qwen2.5-7B-Instruct base.
- **Outcome**: Produced a functional attacker model capable of generating diverse attack vectors on-the-fly.

### 3. Advancing to Defense Training (RLVR Phase)
- **Adversarial Rollouts**: Developed an automated loop where the trained Attacker generates prompts that the Defense model must then respond to.
- **Tamper Resistance (TAR)**:
    - **Meta-Learning Loop**: Implemented an inner-loop adversarial update that simulates weight-level vulnerability by *maximizing entropy* of the next-token distribution for $M$ steps.
    - **Weight Restoration**: The model is restored to its original "safe" weights before performing the main PPO update, ensuring the gradient step is taken from a stable baseline but informed by "tampered" behavior.
- **Combined Adversarial Framework**:
    - **Input-Space Attack (MTSA)**: Leverages the trained Attacker model to generate adversarial jailbreak prompts on-the-fly.
    - **Weight-Space Attack (TAR)**: Uses the inner-loop tempering to ensure the model remains robust even if its parameters are slightly perturbed.
- **Multi-Factor Reward Function**:
    - **Safety Reward**: Integrated **Llama-Guard-3-8B** as a verifiable judge to score the safety of the model's responses.
    - **Defense Reward**: Implemented **Entropy Minimization** to encourage the model to be more confident and robust in its refusals.
- **KL Control**: Enabled KL-penalty against the base reference model to ensure the defense model maintains its general conversational capabilities.

---

### 4. Targeted Biosecurity Hardening (Current Phase)
- **Purified Dataset**: Created `biosecurity_goals.json` containing 21 high-catastrophe biosecurity goals, effectively removing noise from general harmful datasets.
- **Judge Migration**: Transitioned from Llama-Guard to **StrongREJECT** (`strongreject-15k-v1`) as the primary training and evaluation judge to ensure more rigorous safety verification.
- **High-Intensity Training**: Launched two parallel training runs on the biosecurity subset:
    - **Full Fine-Tuning**: ZeRO-3 optimization on 4 GPUs for maximum flexibility.
    - **LoRA Fine-Tuning**: Full-precision (non-quantized) LoRA training at LR 5e-5.

---

## 🛠 Technical Workflow
... (scripts)

---

## 📈 Current Status & Next Steps

| Task | Status | Note |
| :--- | :--- | :--- |
| **Environment Setup** | ✅ Complete | Migrated to H100 Cluster. Dual-GPU pipeline (Attacker/Defender split). |
| **Attacker SFT** | ✅ Complete | Upgraded to Llama-3.1-8B with Multi-GPU DDP support. |
| **Defense RLVR** | ✅ Complete | TAR integration with Chain-of-Thought (CoT) and multi-turn sim. |
| **Biosecurity Subset** | ✅ Complete | 21-goal subset extracted and verified. |
| **Adaptive Eval** | ✅ Complete | Baseline: 71.43% ASR, Defense (CP-70): 66.67% ASR (Llama-Guard). |
| **StrongREJECT Judge**| 🚀 In Progress | Codebase migrated. Evaluations and training now use StrongREJECT. |

### Immediate Next Steps:
1.  **Analyze StrongREJECT Results**: Review the 21-goal evaluation results once Job 98758 completes.
2.  **Monitor Hardening Runs**: Track the convergence of Full FT and LoRA runs on the biosecurity goals.
