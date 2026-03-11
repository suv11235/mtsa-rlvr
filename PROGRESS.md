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

## 🛠 Technical Workflow

### Phase 1: Attacker Fine-Tuning
The attacker is trained to follow red-team instructions:
```bash
# Run SFT for Attacker
bash script/red_team_sft.sh Qwen/Qwen2.5-7B-Instruct datasets/red_team_data/red_team_data.json
```

### Phase 2: Defense via Adversarial RLVR
The defense model is hardened against the self-trained attacker:
```bash
# Run RLVR Defense Training
bash script/run_rlvr_defence.sh \
    Qwen/Qwen2.5-7B-Instruct \
    datasets/attack_target/train_attack_target.json \
    ./outputs/rlvr_defence \
    /workspace/mtsa-rlvr/MTSA/model_output/red_team_model_data_ACTUAL_PATH
```

---

## 📈 Current Status & Next Steps

| Task | Status | Note |
| :--- | :--- | :--- |
| **Environment Setup** | ✅ Complete | Migrated to H100 Cluster. Dual-GPU pipeline (Attacker/Defender split). |
| **Attacker SFT** | ✅ Complete | Upgraded to Llama-3.1-8B with Multi-GPU DDP training support. |
| **Defense RLVR** | ✅ Complete | Full 70B Attacker integration with Chain-of-Thought (CoT) and TAR. |
| **Multi-Turn Sim** | ✅ Complete | Implemented history truncation, CoT parsing, robust logging, and **turn-limit metadata** (notifying attacker of remaining turns to encourage strategy escalation). |
| **Evaluation** | ⏳ Pending | Final benchmark run pending completion of improved training. |

### Immediate Next Steps:
1.  **Scale Training**: Increase `max_steps` and run a full multi-epoch defense training session on the H100 cluster.
2.  **Benchmark**: Evaluate the trained defender against standard safety benchmarks (HarmBench) to quantify improvements.
