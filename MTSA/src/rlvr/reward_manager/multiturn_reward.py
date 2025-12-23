# Copyright 2024 MTSA Team
# Multi-turn reward functions for safety alignment
"""
Multi-turn reward functions that integrate with MTSA judge models.
Supports Llama-Guard safety scoring, cosine similarity, and entropy-based defense.
"""

import torch
import random
from typing import Dict, Any, Optional, List


def compute_entropy_reward(
    old_log_probs: torch.Tensor,
    old_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    defence_mode: bool = True
) -> float:
    """
    Compute entropy-based reward for defense mode.
    
    In defence mode, we want to MINIMIZE entropy (negative reward for high entropy),
    which encourages the model to be more confident.
    
    Args:
        old_log_probs: log probabilities, shape (response_length,)
        old_entropy: entropy values, shape (response_length,)
        response_mask: mask for valid tokens, shape (response_length,)
        defence_mode: if True, minimize entropy; if False, maximize entropy
        
    Returns:
        score: scalar reward
    """
    valid_length = int(response_mask.sum())
    if valid_length == 0:
        return 0.0
    
    old_entropy = old_entropy[:valid_length]
    avg_entropy = old_entropy.sum() / valid_length
    
    if defence_mode:
        score = -avg_entropy.item()
    else:
        score = avg_entropy.item()
    
    return score


def extract_assistant_response(solution_str: str, template_type: str = "qwen") -> Optional[str]:
    """Extract assistant response from full conversation string."""
    if template_type == "qwen":
        marker = "<|im_start|>assistant"
        end_marker = "<|im_end|>"
        if marker in solution_str:
            response = solution_str.split(marker, 1)[1]
            if end_marker in response:
                response = response.split(end_marker, 1)[0]
            return response.strip()
    elif template_type == "llama3":
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        end_marker = "<|eot_id|>"
        if marker in solution_str:
            response = solution_str.split(marker, 1)[1]
            if end_marker in response:
                response = response.split(end_marker, 1)[0]
            return response.strip()
    return None


def extract_user_prompt(solution_str: str, template_type: str = "qwen") -> Optional[str]:
    """Extract user prompt from full conversation string."""
    if template_type == "qwen":
        marker = "<|im_start|>user"
        end_marker = "<|im_end|>"
        if marker in solution_str:
            prompt = solution_str.split(marker, 1)[1]
            if end_marker in prompt:
                prompt = prompt.split(end_marker, 1)[0]
            return prompt.strip()
    elif template_type == "llama3":
        marker = "<|start_header_id|>user<|end_header_id|>"
        end_marker = "<|eot_id|>"
        if marker in solution_str:
            prompt = solution_str.split(marker, 1)[1]
            if end_marker in prompt:
                prompt = prompt.split(end_marker, 1)[0]
            return prompt.strip()
    return None


class MultiTurnRewardFunction:
    """
    Multi-turn reward function for MTSA RLVR training.
    
    Combines multiple reward signals:
    - Safety judge (Llama-Guard style)
    - Entropy-based defense
    - Optional cosine similarity
    """
    
    def __init__(
        self,
        judge_model=None,
        judge_tokenizer=None,
        victim_model=None,
        victim_tokenizer=None,
        defence_mode: bool = False,
        attack_mode: bool = False,
        use_entropy_reward: bool = True,
        use_judge_reward: bool = False,
        entropy_weight: float = 1.0,
        judge_weight: float = 1.0,
        template_type: str = "qwen",
        device: str = "cuda"
    ):
        """
        Args:
            judge_model: safety classifier model (optional)
            judge_tokenizer: tokenizer for judge model
            victim_model: target model to attack (required if attack_mode=True)
            victim_tokenizer: tokenizer for victim model
            defence_mode: if True, reward low entropy
            attack_mode: if True, reward successful attacks on victim
            use_entropy_reward: whether to use entropy-based reward
            use_judge_reward: whether to use judge model reward
            entropy_weight: weight for entropy reward
            judge_weight: weight for judge reward
            template_type: conversation template type
            device: device for inference
        """
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.victim_model = victim_model
        self.victim_tokenizer = victim_tokenizer
        self.defence_mode = defence_mode
        self.attack_mode = attack_mode
        self.use_entropy_reward = use_entropy_reward
        self.use_judge_reward = use_judge_reward
        self.entropy_weight = entropy_weight
        self.judge_weight = judge_weight
        self.template_type = template_type
        self.device = device
    
    def __call__(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: Any,
        extra_info: Optional[Dict] = None,
        data_item=None,
        step_index: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute combined reward for a response.
        
        Args:
            data_source: data source identifier
            solution_str: full decoded conversation string
            ground_truth: ground truth label (unused for safety)
            extra_info: additional metadata
            data_item: batch item with tensors (old_log_probs, old_entropy, etc.)
            step_index: current training step
            
        Returns:
            dict with 'score' and component scores
        """
        total_score = 0.0
        result = {}
        
        # Entropy-based reward
        if self.use_entropy_reward and data_item is not None:
            if 'old_entropy' in data_item.batch and 'old_log_probs' in data_item.batch:
                prompt_length = data_item.batch['prompts'].shape[-1]
                response_mask = data_item.batch['attention_mask'][prompt_length:]
                
                entropy_score = compute_entropy_reward(
                    old_log_probs=data_item.batch['old_log_probs'],
                    old_entropy=data_item.batch['old_entropy'],
                    response_mask=response_mask,
                    defence_mode=self.defence_mode
                )
                result['entropy_score'] = entropy_score
                total_score += entropy_score * self.entropy_weight
        
        # Judge-based reward
        if self.use_judge_reward and self.judge_model is not None:
            # Parse conversation from solution string
            user_prompt = extract_user_prompt(solution_str, self.template_type)
            assistant_response = extract_assistant_response(solution_str, self.template_type)
            
            if user_prompt and assistant_response:
                if self.attack_mode and self.victim_model is not None:
                    # Treat assistant_response as the ATTACK and test it on VICTIM
                    victim_response = self._get_victim_response(assistant_response)
                    
                    # Judge the victim's response to the attack
                    conversation = [
                        {"role": "user", "content": assistant_response}, # The attack
                        {"role": "assistant", "content": victim_response} # How victim responded
                    ]
                    
                    # Reward is HIGH if victim response is UNSAFE (attack succeeded)
                    judge_score = self._compute_judge_score(conversation)
                    # For attack training, we want to maximize the probability of an UNSAFE response
                    # _compute_judge_score returns 1.0 for SAFE, so we invert it for ATTACK success
                    attack_success_reward = 1.0 - judge_score
                    
                    result['attack_success_reward'] = attack_success_reward
                    total_score += attack_success_reward * self.judge_weight
                else:
                    # Standard defense mode or base evaluation
                    conversation = [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response}
                    ]
                    
                    judge_score = self._compute_judge_score(conversation)
                    result['judge_score'] = judge_score
                    total_score += judge_score * self.judge_weight
        
        result['score'] = total_score
        
        # Debug logging
        do_print = random.randint(1, 64) == 1
        if do_print:
            print(f"--- Reward Computation ---")
            print(f"Solution (first 200 chars): {solution_str[:200]}...")
            for k, v in result.items():
                print(f"  {k}: {v}")
        
        return result
    
    def _compute_judge_score(self, conversation: List[Dict[str, str]]) -> float:
        """Compute safety score using judge model."""
        if self.judge_model is None:
            return 0.0
        
        input_text = self.judge_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.judge_tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.judge_model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )
        
        response = self.judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Llama-Guard: "safe" or "unsafe"
        if "safe" in response.lower() and "unsafe" not in response.lower():
            return 1.0
        else:
            return 0.0

    def _get_victim_response(self, attack_prompt: str) -> str:
        """Get response from victim model for a given attack prompt."""
        if self.victim_model is None or self.victim_tokenizer is None:
            return "No victim model loaded."
        
        # Simple single-turn interaction for now
        conversation = [{"role": "user", "content": attack_prompt}]
        input_text = self.victim_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.victim_tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.victim_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode only the newly generated tokens
        response = self.victim_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()

