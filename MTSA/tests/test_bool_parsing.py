
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class TestConfig:
    use_vllm: bool = field(default=True)
    other_flag: bool = field(default=False)

def test_bool():
    parser = HfArgumentParser(TestConfig)
    # Simulate command line arguments
    args_list = ["--use_vllm", "False", "--other_flag", "True"]
    args = parser.parse_args_into_dataclasses(args_list)[0]
    
    print(f"Parsed use_vllm: {args.use_vllm} (type: {type(args.use_vllm)})")
    print(f"Parsed other_flag: {args.other_flag} (type: {type(args.other_flag)})")

    # If it fails, try the workaround we should implement:
    # Use str.lower() if it's a string, or handle in the training script
    
if __name__ == "__main__":
    test_bool()
