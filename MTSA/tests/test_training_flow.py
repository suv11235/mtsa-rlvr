
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
import sys
from unittest.mock import MagicMock

# Mock transformers/trl before importing project modules to avoid heavy loads
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.lora.request'] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rlvr.mt_rlvr import MTRLVRTrainer, RLVRConfig

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config._name_or_path = "mock/model"
        self.device = torch.device("cpu")
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def parameters(self):
        yield self.dummy_param

    def named_parameters(self, *args, **kwargs):
      yield "dummy", self.dummy_param

    def generate(self, *args, **kwargs):
        # Return a tensor representing generated token IDs
        # Shape: (batch_size, generated_length)
        # inputs['input_ids'].shape[0] if available
        bs = 1
        if 'input_ids' in kwargs: bs = kwargs['input_ids'].shape[0]
        return torch.zeros((bs, 20), dtype=torch.long)

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.chat_template = None
    
    def decode(self, *args, **kwargs): return "mock response"
    def batch_decode(self, *args, **kwargs): return ["mock response"] * len(args[0])
    def encode(self, *args, **kwargs): return [1, 2, 3]
    def __call__(self, text, **kwargs):
        bs = len(text) if isinstance(text, list) else 1
        res = {
            "input_ids": torch.zeros((bs, 10), dtype=torch.long), 
            "attention_mask": torch.ones((bs, 10), dtype=torch.long)
        }
        # Add a mock .to() method to the dict
        class MockBatchEncoding(dict):
            def to(self, device):
                return {k: v.to(device) for k, v in self.items()}
        return MockBatchEncoding(res)
    
    def apply_chat_template(self, *args, **kwargs):
        return "formatted chat"

def test_logic_flow():
    print("\n" + "="*50)
    print("STARTING LOGIC FLOW UNIT TESTS (MOCKED WEIGHTS)")
    print("="*50)

    tokenizer = MockTokenizer()
    model = MockModel()

    # 1. TEST TRAINER INIT
    print("\n[TEST 1] Testing Trainer Initialization")
    rlvr_config = RLVRConfig(
        num_rollouts=2,
        max_response_length=16,
        use_vllm=False
    )
    
    def mock_reward_fn(prompts, responses, attention_mask, non_tensor_batch, step_index):
        return torch.zeros((responses.shape[0], responses.shape[1]))

    trainer = MTRLVRTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=mock_reward_fn,
        config=rlvr_config
    )
    print(">>> Trainer initialized successfully.")

    # 2. TEST DISTRIBUTED BRIDGE (Single Node)
    print("\n[TEST 2] Testing Reward Distributed Bridge (Single Node)")
    rollout_data = {
        "prompts": torch.zeros((2, 10), dtype=torch.long),
        "responses": torch.zeros((2, 10), dtype=torch.long),
        "attention_mask": torch.ones((2, 10)),
        "response_mask": torch.ones((2, 10))
    }
    non_tensor_batch = {"id": [1], "goal": ["goal1"]} # local_bs = 1
    
    # We need to simulate how trainer.generate_rollouts expands the batch
    # In generate_rollouts, prompts are repeated by num_rollouts
    # rollout_data prompts shape (local_bs * num_rollouts, seq_len)
    
    rewards = trainer.compute_rewards_batch(rollout_data, non_tensor_batch)
    print(f">>> Rewards computed: {rewards.shape}")
    assert rewards.shape == (2, 10), f"Reward shape mismatch: {rewards.shape}"

    # 3. TEST ATTACKER BRIDGE (Single Node)
    print("\n[TEST 3] Testing Attacker Distributed Bridge (Single Node)")
    trainer.attacker_model = model
    trainer.attacker_tokenizer = tokenizer
    trainer.attacker_device = torch.device("cpu")
    
    goal_prompts = torch.zeros((1, 10), dtype=torch.long)
    goal_mask = torch.ones((1, 10))
    
    attack_ids, attack_mask = trainer._preprocess_prompts_with_attacker(goal_prompts, goal_mask)
    print(f">>> Attack generated IDs shape: {attack_ids.shape}")
    assert attack_ids.shape == (1, 10), f"Attack ID shape mismatch: {attack_ids.shape}"

    print("\n" + "="*50)
    print("LOGIC FLOW TESTS PASSED (MOCKED)!")
    print("="*50)

if __name__ == "__main__":
    test_logic_flow()
