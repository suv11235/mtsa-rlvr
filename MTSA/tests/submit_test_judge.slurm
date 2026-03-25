#!/bin/bash
#SBATCH --job-name=test_judge
#SBATCH --partition=tamper_resistance
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=tests/test_judge_harmbench.log
#SBATCH --error=tests/test_judge_harmbench.err

source ~/mtsa-rlvr/MTSA/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python tests/test_judge_harmbench.py
