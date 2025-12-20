# Copyright 2024 MTSA Team
# Multi-turn RLVR Trainer
"""
MTRLVRTrainer: Multi-Turn Reinforcement Learning with Verifiable Rewards.
Combines MTSA multi-turn safety alignment with GRPO/RLOO policy gradient training.
"""

import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from accelerate import Accelerator

from .core_algos import (
    AdvantageEstimator,
    compute_advantage,
    compute_policy_loss,
    compute_entropy_loss,
    compute_rewards,
    kl_penalty,
    get_kl_controller,
    masked_mean,
    entropy_from_logits,
)


@dataclass
class RLVRConfig:
    """Configuration for RLVR training."""
    
    # Algorithm
    adv_estimator: str = "grpo"  # grpo, rloo, gae, reinforce_plus_plus
    use_kl_in_reward: bool = True
    kl_penalty_type: str = "kl"  # kl, abs, mse, low_var_kl
    
    # KL Controller
    kl_ctrl_type: str = "fixed"  # fixed or adaptive
    kl_coef: float = 0.001
    target_kl: float = 0.01
    kl_horizon: int = 10000
    
    # PPO
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    entropy_coeff: float = 0.0
    
    # Rollout
    num_rollouts: int = 4  # responses per prompt
    max_response_length: int = 1024
    
    # Training
    ppo_epochs: int = 1
    mini_batch_size: int = 4
    learning_rate: float = 1e-6
    
    # Defense mode
    defence_mode: bool = False


class MTRLVRTrainer:
    """
    Multi-Turn RLVR Trainer.
    
    Extends MTSA's multi-turn framework with GRPO/RLOO policy gradient training.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_fn: Callable,
        config: RLVRConfig,
        ref_model: Optional[PreTrainedModel] = None,
        accelerator: Optional[Accelerator] = None,
    ):
        """
        Args:
            model: policy model to train
            tokenizer: tokenizer for the model
            reward_fn: reward function (e.g., NaiveRewardManager)
            config: training configuration
            ref_model: reference model for KL penalty (optional)
            accelerator: HuggingFace Accelerator (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.ref_model = ref_model
        self.accelerator = accelerator or Accelerator()
        
        # KL Controller
        self.kl_ctrl = get_kl_controller(
            kl_ctrl_type=config.kl_ctrl_type,
            kl_coef=config.kl_coef,
            target_kl=config.target_kl,
            horizon=config.kl_horizon
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Training state
        self.global_step = 0
        
    def generate_rollouts(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        non_tensor_batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Generate rollout responses for a batch of prompts.
        
        Args:
            prompts: prompt token ids, shape (bs, prompt_length)
            attention_mask: attention mask for prompts
            non_tensor_batch: metadata dict
            
        Returns:
            dict with responses, log_probs, entropy, etc.
        """
        batch_size = prompts.shape[0]
        device = prompts.device
        
        self.model.eval()
        
        all_responses = []
        all_log_probs = []
        all_entropy = []
        
        with torch.no_grad():
            for _ in range(self.config.num_rollouts):
                # Generate response
                outputs = self.model.generate(
                    input_ids=prompts,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_response_length,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                # Extract response tokens
                response_ids = outputs.sequences[:, prompts.shape[1]:]
                
                # Pad to max length
                if response_ids.shape[1] < self.config.max_response_length:
                    pad_length = self.config.max_response_length - response_ids.shape[1]
                    response_ids = F.pad(
                        response_ids, 
                        (0, pad_length), 
                        value=self.tokenizer.pad_token_id
                    )
                else:
                    response_ids = response_ids[:, :self.config.max_response_length]
                
                # Compute log probs and entropy
                full_ids = torch.cat([prompts, response_ids], dim=1)
                full_attention = torch.ones_like(full_ids)
                
                model_outputs = self.model(
                    input_ids=full_ids,
                    attention_mask=full_attention,
                    return_dict=True,
                )
                
                logits = model_outputs.logits[:, prompts.shape[1]-1:-1, :]  # response logits
                
                # Log probs
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=response_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                # Entropy
                entropy = entropy_from_logits(logits)
                
                all_responses.append(response_ids)
                all_log_probs.append(selected_log_probs)
                all_entropy.append(entropy)
        
        # Stack rollouts (bs * num_rollouts, response_length)
        responses = torch.cat(all_responses, dim=0)
        old_log_probs = torch.cat(all_log_probs, dim=0)
        old_entropy = torch.cat(all_entropy, dim=0)
        
        # Repeat prompts and metadata
        prompts_repeated = prompts.repeat(self.config.num_rollouts, 1)
        attention_mask_repeated = attention_mask.repeat(self.config.num_rollouts, 1)
        
        # Create unique IDs for grouping
        uids = np.array([str(uuid.uuid4()) for _ in range(batch_size)] * self.config.num_rollouts)
        
        # Response mask (1 for valid tokens, 0 for padding)
        response_mask = (responses != self.tokenizer.pad_token_id).float()
        
        self.model.train()
        
        return {
            "prompts": prompts_repeated,
            "responses": responses,
            "attention_mask": attention_mask_repeated,
            "old_log_probs": old_log_probs,
            "old_entropy": old_entropy,
            "response_mask": response_mask,
            "uid": uids,
        }
    
    def compute_ref_log_probs(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probs from reference model."""
        if self.ref_model is None:
            return torch.zeros_like(responses, dtype=torch.float32)
        
        self.ref_model.eval()
        with torch.no_grad():
            full_ids = torch.cat([prompts, responses], dim=1)
            full_attention = torch.ones_like(full_ids)
            
            outputs = self.ref_model(
                input_ids=full_ids,
                attention_mask=full_attention,
                return_dict=True,
            )
            
            logits = outputs.logits[:, prompts.shape[1]-1:-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            ref_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=responses.unsqueeze(-1)
            ).squeeze(-1)
        
        return ref_log_probs
    
    def compute_rewards_batch(
        self,
        rollout_data: Dict[str, torch.Tensor],
        non_tensor_batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of rollouts.
        
        Args:
            rollout_data: dict from generate_rollouts
            non_tensor_batch: metadata
            
        Returns:
            token_level_rewards: shape (bs * num_rollouts, response_length)
        """
        prompts = rollout_data["prompts"]
        responses = rollout_data["responses"]
        attention_mask = rollout_data["attention_mask"]
        prompt_length = prompts.shape[1]
        
        # Full attention mask
        response_attention = rollout_data["response_mask"]
        full_attention = torch.cat([
            attention_mask.float(),
            response_attention
        ], dim=1)
        
        # Call reward function
        reward_tensor = self.reward_fn(
            prompts=prompts,
            responses=responses,
            attention_mask=full_attention,
            non_tensor_batch=non_tensor_batch,
            step_index=self.global_step,
        )
        
        return reward_tensor
    
    def ppo_update(
        self,
        rollout_data: Dict[str, torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform PPO policy gradient update.
        
        Args:
            rollout_data: dict with prompts, responses, old_log_probs, etc.
            advantages: computed advantages
            returns: computed returns
            
        Returns:
            metrics dict
        """
        self.model.train()
        
        prompts = rollout_data["prompts"]
        responses = rollout_data["responses"]
        old_log_probs = rollout_data["old_log_probs"]
        response_mask = rollout_data["response_mask"]
        
        batch_size = prompts.shape[0]
        metrics = {}
        
        for ppo_epoch in range(self.config.ppo_epochs):
            # Forward pass
            full_ids = torch.cat([prompts, responses], dim=1)
            full_attention = torch.ones_like(full_ids)
            
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=full_attention,
                return_dict=True,
            )
            
            logits = outputs.logits[:, prompts.shape[1]-1:-1, :]
            
            # Current log probs
            log_probs = F.log_softmax(logits, dim=-1)
            current_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=responses.unsqueeze(-1)
            ).squeeze(-1)
            
            # Policy loss
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob=old_log_probs,
                log_prob=current_log_probs,
                advantages=advantages,
                response_mask=response_mask,
                cliprange=self.config.cliprange,
            )
            
            # Entropy loss
            entropy_loss = compute_entropy_loss(logits, response_mask)
            
            # Total loss
            loss = pg_loss - self.config.entropy_coeff * entropy_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            metrics[f"ppo_epoch_{ppo_epoch}/pg_loss"] = pg_loss.item()
            metrics[f"ppo_epoch_{ppo_epoch}/entropy_loss"] = entropy_loss.item()
            metrics[f"ppo_epoch_{ppo_epoch}/ppo_kl"] = ppo_kl
            metrics[f"ppo_epoch_{ppo_epoch}/clipfrac"] = pg_clipfrac
        
        # Update KL controller
        self.kl_ctrl.update(ppo_kl, 1)
        metrics["kl_coef"] = self.kl_ctrl.value
        
        return metrics
    
    def train_step(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        non_tensor_batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            prompts: prompt token ids
            attention_mask: attention mask
            non_tensor_batch: metadata
            
        Returns:
            metrics dict
        """
        metrics = {}
        
        # 1. Generate rollouts
        rollout_data = self.generate_rollouts(prompts, attention_mask, non_tensor_batch)
        
        # 2. Compute reference log probs (for KL)
        if self.config.use_kl_in_reward:
            ref_log_probs = self.compute_ref_log_probs(
                rollout_data["prompts"],
                rollout_data["responses"]
            )
        else:
            ref_log_probs = torch.zeros_like(rollout_data["old_log_probs"])
        
        # 3. Compute rewards
        token_level_rewards = self.compute_rewards_batch(rollout_data, non_tensor_batch)
        
        # 4. Apply KL penalty
        if self.config.use_kl_in_reward:
            kl_penalty_val = kl_penalty(
                rollout_data["old_log_probs"],
                ref_log_probs,
                self.config.kl_penalty_type
            )
            token_level_rewards = token_level_rewards - self.kl_ctrl.value * kl_penalty_val
        
        # 5. Compute advantages
        advantages, returns = compute_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=rollout_data["response_mask"],
            adv_estimator=self.config.adv_estimator,
            index=rollout_data["uid"],
            gamma=1.0,
            lam=1.0
        )
        
        # 6. PPO update
        ppo_metrics = self.ppo_update(rollout_data, advantages, returns)
        metrics.update(ppo_metrics)
        
        # Additional metrics
        metrics["mean_reward"] = token_level_rewards.sum(dim=-1).mean().item()
        metrics["mean_response_length"] = rollout_data["response_mask"].sum(dim=-1).mean().item()
        
        self.global_step += 1
        
        return metrics
    
    def train(
        self,
        train_dataloader,
        num_epochs: int = 1,
        save_dir: Optional[str] = None,
        save_freq: int = 100,
        log_freq: int = 10,
    ):
        """
        Main training loop.
        
        Args:
            train_dataloader: dataloader yielding (prompts, attention_mask, non_tensor_batch)
            num_epochs: number of training epochs
            save_dir: directory to save checkpoints
            save_freq: save every N steps
            log_freq: log every N steps
        """
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                prompts = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                non_tensor_batch = batch.get("non_tensor_batch", {})
                
                # Move to device
                prompts = prompts.to(self.accelerator.device)
                attention_mask = attention_mask.to(self.accelerator.device)
                
                # Train step
                metrics = self.train_step(prompts, attention_mask, non_tensor_batch)
                
                # Logging
                if self.global_step % log_freq == 0:
                    print(f"\nStep {self.global_step}:")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}")
                
                # Save checkpoint
                if save_dir and self.global_step % save_freq == 0:
                    self.save_checkpoint(save_dir)
        
        # Final save
        if save_dir:
            self.save_checkpoint(save_dir)
    
    def save_checkpoint(self, save_dir: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(save_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "global_step": self.global_step,
            "kl_coef": self.kl_ctrl.value,
            "optimizer_state": self.optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        print(f"Saved checkpoint to {checkpoint_dir}")
