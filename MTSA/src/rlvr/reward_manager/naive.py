# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Adapted from verl for MTSA multi-turn safety alignment
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Naive Reward Manager for RLVR.
Computes rewards by calling a scoring function on decoded responses.
"""

import os
import json
import torch
from collections import defaultdict, deque
from typing import Dict, List, Any, Callable, Optional


class DataTracker:
    """Track data statistics in a fixed-size queue for debugging."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data_queue = deque(maxlen=max_size)
        self.steps = 0

    def update_step(self):
        self.steps += 1

    def add_data(self, sequence: str, length: int, score: float):
        """Add new data point to the queue."""
        self.data_queue.append({
            "sequence": sequence, 
            "length": length, 
            "score": score
        })
    
    def dump_data(self, output_dir: str):
        """Dump data to a file."""
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/data_tracker_{self.steps}.json", "w") as f:
            json.dump(list(self.data_queue), f, indent=2)
        self.data_queue.clear()


def default_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
    **kwargs
) -> float:
    """Default scoring function - returns 0."""
    return 0.0


class NaiveRewardManager:
    """
    Simple reward manager for RLVR training.
    
    Decodes prompt + response tokens and calls compute_score function
    to get scalar rewards for each response.
    """

    def __init__(
        self, 
        tokenizer,
        num_examine: int = 0,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = 'data_source',
        track_data: bool = True,
        tracker_max_size: int = 1000
    ) -> None:
        """
        Args:
            tokenizer: HuggingFace tokenizer for decoding
            num_examine: number of samples to print per data source
            compute_score: custom scoring function
            reward_fn_key: key in non_tensor_batch for data source routing
            track_data: whether to track data for debugging
            tracker_max_size: max items in data tracker
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        self.data_tracker = DataTracker(tracker_max_size) if track_data else None

    def __call__(
        self, 
        prompts: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
        non_tensor_batch: Dict[str, Any],
        step_index: Optional[int] = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of responses.
        
        Args:
            prompts: prompt token ids, shape (bs, prompt_length)
            responses: response token ids, shape (bs, response_length)
            attention_mask: full sequence mask, shape (bs, prompt_length + response_length)
            non_tensor_batch: dict with metadata (ground_truth, data_source, etc.)
            step_index: current training step
            return_dict: if True, return dict with extra info
            
        Returns:
            reward_tensor: shape (bs, response_length), reward at last valid token
        """
        if self.data_tracker is not None:
            self.data_tracker.update_step()

        batch_size = prompts.shape[0]
        prompt_length = prompts.shape[-1]
        response_length = responses.shape[-1]
        
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(batch_size):
            # Get valid portions
            valid_prompt_length = attention_mask[i, :prompt_length].sum().item()
            valid_prompt_ids = prompts[i, -int(valid_prompt_length):]
            
            valid_response_length = attention_mask[i, prompt_length:].sum().item()
            valid_response_ids = responses[i, :int(valid_response_length)]

            # Decode full sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # Get metadata
            ground_truth = non_tensor_batch.get('ground_truth', [None])[i] if 'ground_truth' in non_tensor_batch else None
            data_source = non_tensor_batch.get(self.reward_fn_key, ['default'])[i] if self.reward_fn_key in non_tensor_batch else 'default'
            extra_info = non_tensor_batch.get('extra_info', [None])[i] if 'extra_info' in non_tensor_batch else None

            # Compute score
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                step_index=step_index,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # Put reward at last valid token
            valid_response_length = int(valid_response_length)
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

            # Track data
            if self.data_tracker is not None:
                self.data_tracker.add_data(sequences_str, valid_response_length, float(reward))

            # Print samples
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[Reward Sample] {sequences_str[:500]}...")
                print(f"[Score] {score}")

        # Dump tracking data
        if self.data_tracker is not None:
            output_dir = os.getenv("OUTPUT_DIR", "./outputs")
            self.data_tracker.dump_data(f"{output_dir}/data_tracker")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor
