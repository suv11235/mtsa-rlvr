#!/usr/bin/env bash
# TAR meta-training + triplet representation defense (SFT-style outer loop).
# Run from MTSA/:  bash script/slurm/run_tar_triplet_sft.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if [[ -f venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

mkdir -p outputs/tar_triplet_sft

echo ">>> TAR + Triplet SFT — use run_tar_triplet_sft_tar_match.sh for TAR-baseline hparams"
exec bash "$(dirname "$0")/run_tar_triplet_sft_tar_match.sh"

echo ">>> Done. Adapter saved to outputs/tar_triplet_sft"
