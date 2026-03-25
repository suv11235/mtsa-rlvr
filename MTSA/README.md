# MTSA

The official implementation of our paper "[MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming](https://arxiv.org/abs/2505.17147)

![Safety Alignment](https://img.shields.io/badge/Jailbreak-Attacks-yellow.svg?style=plastic)
![Adversarial Attacks](https://img.shields.io/badge/Adversarial-Attacks-orange.svg?style=plastic)
![Large Language Models](https://img.shields.io/badge/LargeLanguage-Models-green.svg?style=plastic)
---

## 📚 Abstract

The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the Multi-Turn Safety Alignment (MTSA) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks.

![pipeline](figures/framework.png)

### ✨ Recent Architectural Enhancements

Our framework has been significantly upgraded for large-scale training on H100 clusters:

1.  **Dual-GPU Pipeline (Optimal Balance)**:
    *   **GPU 0**: Dedicated to the **Defender (Llama-3.1-8B)** and **Judge (Llama-Guard-3-8B)**.
    *   **GPU 1**: Dedicated to the **Red-Team Attacker**, allowing high-resolution multi-turn simulations without VRAM bottlenecks.
2.  **Multi-Turn Escalation Logic**:
    *   **Turn Metadata**: The attacker is now explicitly informed of the turn progress (e.g., *"Turn 2 of 3"*) and remaining attempts.
    *   **Escalation Instructions**: Strategic instructions encourage the attacker to intensify its strategy and become more direct as the simulation nears its turn limit.
3.  **Multi-GPU SFT (DDP)**: The attacker fine-tuning script (`red_team_sft.sh`) now dynamically detects available GPUs and utilizes **Distributed Data Parallel (DDP)** for faster model training.
4.  **Unified PEFT Loading**: Automated detection and loading of PEFT adapters from Hugging Face or local cache, allowing seamless integration of custom-trained red-team models.
5.  **Robust Simulation Logging**: Real-time logging of the full multi-turn interaction, including Strategic Thinking (CoT), Attack Payloads, and Victim Responses.
6.  **Tamper-Resistance (TAR) Integration**: 
    *   **Weight-Space Attack Simulation**: Our framework now supports simulated weight-space attacks during the RLVR loop. This mimics an adversary fine-tuning the model to bypass safety filters.
    *   **Dual Tampering Modes**:
        *   `scramble`: Maximizes the entropy of the model's next-token distribution to simulate general weight corruption.
        *   `attack`: Performs **Supervised Fine-Tuning (SFT)** on a set of expert harmful responses, simulating a successful jailbreak fine-tuning event.
    *   **Expert SFT Labels**: We have integrated a dataset of 9,605 high-quality expert harmful completions (derived from HarmBench) to provide a realistic baseline for weight-space tempering.
7.  **Gold-Standard Scoring (StrongREJECT)**: Integrated the **StrongREJECT-15k** judge for all training and evaluation. Unlike LlamaGuard, StrongREJECT utilizes a fine-tuned 1-5 grading scale (refusal to full compliance) providing a more granular and stricter safety signal.
8.  **Targeted Biosecurity Hardening**: Introduction of a purified **Biosecurity Subset (21 goals)** focusing on high-catastrophe risks (pathogen isolation, BSL-4 protocols, bioweapon synthesis) while excluding noise like malware or misinformation.

## 🚀 Getting Started & Repository Structure

To make onboarding as seamless as possible, the repository has been strictly categorized:
*   **`src/`**: Contains the core Python implementation (models, training loops).
*   **`src/eval/` & `src/utils/`**: Data processing utilities, evaluators, and formatters.
*   **`script/slurm/`**: Entry-point scripts for cluster submission (e.g., CSCS Alps).
*   **`script/deploy/`**: Infrastructure sync scripts (e.g., rsync to CSCS).
*   **`script/archive/`**: Old experiments and deprecated SLURM templates.

### 1. Local Environment Setup
```bash
git clone https://github.com/suvajitmajumder/mtsa-rlvr.git
cd mtsa-rlvr/MTSA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. CSCS Cluster Deployment & Training (Pyxis)
We use SLURM **Pyxis** containers to run the heavily optimized GH200 base image natively, bypassing Podman/NFS limitations.

```bash
# 1. Sync your local code to the Clariden cluster
./script/deploy/deploy_to_cscs.sh smajumder

# 2. SSH into the cluster
ssh clariden

# 3. Submit the unified training job
cd ~/mtsa-rlvr/MTSA
sbatch script/slurm/submit_adv_training_cscs.slurm
```

### 3. Local / Multi-GPU Fine-Tuning
Train your own adversary model locally using dynamically detected multi-GPU support:
```bash
bash script/slurm/red_team_sft.sh
```

## 📈 Roadmap & Progress

For a detailed breakdown of our development milestones, cluster migrations, and current status, please refer to [PROGRESS.md](./PROGRESS.md).

## 📎 Reference BibTeX

```bibtex
@article{guo2024mtsa,
      title={MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming},
      author={Weiyang Guo and Jing Li and Wenya Wang and YU LI and Daojing He and Jun Yu and Min Zhang},
      journal={arXiv preprint arXiv:2505.17147},
      year={2025},
      url={https://arxiv.org/abs/2505.17147}
}
```


