#!/bin/bash
#SBATCH --job-name=diag_logits
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=tests/diag_harmbench_logits.log
#SBATCH --error=tests/diag_harmbench_logits.err

source ~/mtsa-rlvr/MTSA/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python tests/diag_harmbench_logits.py
