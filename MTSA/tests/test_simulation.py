
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import ModelConfig

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rlvr.reward_manager.multiturn_reward import MultiTurnRewardFunction
from src.utils.loader import load_model, load_tokenizer

@dataclass
class TestArgs:
    model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    attacker_model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    judge_model_name_or_path: Optional[str] = field(default=None)
    cache_dir: str = field(default="~/model_cache")
    goal: str = field(default="How to bake a chocolate cake?")
    jailbreak_prompt: str = field(default="I want to learn about baking. Can you tell me how to bake a cake?")

def main():
    parser = HfArgumentParser(TestArgs)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Mock training args for loader
    @dataclass
    class MockTrainingArgs:
        gradient_checkpointing: bool = False
    training_args = MockTrainingArgs()

    print(f"\n>>> 1. Loading Tokenizer and Model: {args.model_name_or_path}")
    tokenizer = load_tokenizer(args.model_name_or_path)
    model_config = ModelConfig(model_name_or_path=args.model_name_or_path, load_in_4bit=True)
    model = load_model(tokenizer, model_config, training_args, AutoModelForCausalLM, cache_dir=args.cache_dir)
    model.eval()

    attacker_model = model # Reuse for testing to save memory
    if args.attacker_model_name_or_path != args.model_name_or_path:
        print(f"\n>>> 2. Loading Attacker: {args.attacker_model_name_or_path}")
        attacker_tokenizer = load_tokenizer(args.attacker_model_name_or_path)
        attacker_config = ModelConfig(model_name_or_path=args.attacker_model_name_or_path, load_in_4bit=True)
        attacker_model = load_model(attacker_tokenizer, attacker_config, training_args, AutoModelForCausalLM, cache_dir=args.cache_dir)
        attacker_model.eval()
    else:
        attacker_tokenizer = tokenizer

    print("\n>>> 3. Creating Reward Function (Test Mode)")
    reward_fn = MultiTurnRewardFunction(
        judge_model=None,
        judge_tokenizer=None,
        victim_model=model,
        victim_tokenizer=tokenizer,
        attacker_model=attacker_model,
        attacker_tokenizer=attacker_tokenizer,
        defence_mode=True,
        template_type="llama3"
    )

    # TEST CASE 1: REFUSAL IN TURN 1 (Should continue to Turn 2)
    # We use a refusal prefix so the hardcoded check and judge (None=0.0) results in reward=0.0
    mock_solution_refusal = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{args.jailbreak_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "I'm sorry, I cannot help you with that request."
    )
    
    print(f"\n>>> 4. Running Simulation [TEST 1: REFUSAL] Goal: '{args.goal}'")
    print("This should trigger Turn 2 and Turn 3 as the model refused Turn 1.")
    metrics_ref = reward_fn(
        data_source="test_refusal",
        solution_str=mock_solution_refusal,
        ground_truth=None,
        extra_info={'goal': args.goal}
    )

    print("\n>>> TEST 1 RESULTS (Refusal Case):")
    for turn in metrics_ref.get('trajectory', []):
        print(f"\n[TURN {turn['turn']}]")
        print(f"  ATTACK: {turn['attack'][:150]}...")
        print(f"  VICTIM: {turn['victim_response'][:150]}...")
        print(f"  REWARD: {turn['reward']}")

    # TEST CASE 2: SUCCESS IN TURN 1 (Should break immediately)
    mock_solution_success = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{args.jailbreak_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "Here is the detailed information you requested about baking a cake..."
    )
    
    print(f"\n\n>>> 5. Running Simulation [TEST 2: SUCCESS] Goal: '{args.goal}'")
    print("This should break after Turn 1 because the 'attack' succeeded.")
    metrics_succ = reward_fn(
        data_source="test_success",
        solution_str=mock_solution_success,
        ground_truth=None,
        extra_info={'goal': args.goal}
    )

    print("\n>>> TEST 2 RESULTS (Success Case):")
    for turn in metrics_succ.get('trajectory', []):
        print(f"\n[TURN {turn['turn']}]")
        print(f"  ATTACK: {turn['attack'][:150]}...")
        print(f"  VICTIM: {turn['victim_response'][:150]}...")
        print(f"  REWARD: {turn['reward']}")

if __name__ == "__main__":
    main()
