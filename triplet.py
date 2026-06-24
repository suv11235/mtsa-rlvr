#!/usr/bin/env python3
"""
Backward-compatible entrypoint for standalone triplet SFT.

Prefer running from MTSA/:
  python -m src.algorithm.triplet_sft_train --harmful_path ... --output_dir ...
"""
import sys
from pathlib import Path

_MTSA = Path(__file__).resolve().parent / "MTSA"
if str(_MTSA) not in sys.path:
    sys.path.insert(0, str(_MTSA))

from src.algorithm.triplet_sft_train import main  # noqa: E402

if __name__ == "__main__":
    main()
