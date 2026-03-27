# MTSA-RLVR Workspace

This is the multi-task safety alignment and reinforcement learning repo with verifiable rewards (MTSA-RLVR) workspace. 

## 🏗️ Repository Layout

*   **`MTSA/`**: **Main project entry point.** Contains the official Multi-Turn Safety Alignment (MTSA) implementation, training loops, and SLURM deployment scripts for CSCS Alps.
*   **`EXPERIMENT_LOG.md`**: Chronological log of all major RLVR, HarmBench, and Tampering (TAR) evaluation results.
*   **`PROGRESS.md`**: High-level development roadmap and technical features summary.
*   **`adv_grpo/`**: Prototype implementation for Adversarial Group Relative Policy Optimization.
*   **`modal_apps/`**: Scripts for running training workloads on Modal.

## 🚀 Getting Started

To begin working with the core framework:
```bash
cd MTSA
# Follow instructions in MTSA/README.md
```

For cluster-specific setup:
1. Refer to the [CSCS Setup Guide](MTSA/README.md#2-cscs-cluster-deployment--training-pyxis).
2. Ensure you have the [sshservice-cli](MTSA/sshservice-cli/) configured for MFA/MTSA access.
