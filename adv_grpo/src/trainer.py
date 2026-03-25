import os
import gc
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import uuid
from tqdm import tqdm

from src.core_algos import compute_advantage, compute_policy_loss, compute_entropy_loss, get_kl_controller, compute_rewards
from typing import Dict, Any, Callable

class AdvGRPOTrainer:
    def __init__(self, model, tokenizer, reward_fn: Callable, config: Dict, accelerator, lr_scheduler=None, optimizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        
        self.optimizer = optimizer or torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.get('learning_rate', 1e-6))
        self.lr_scheduler = lr_scheduler
        
        # From reward function get the vllm engine if policy matches
        self.vllm_engine = getattr(reward_fn, 'attacker_vllm_engine' if config.get('attack_mode') else 'victim_vllm_engine', None)
        
        # KL Settings
        self.kl_ctrl = get_kl_controller(
            kl_ctrl_type=config.get('kl_ctrl_type', 'fixed'),
            kl_coef=config.get('kl_coef', 0.001),
            target_kl=config.get('target_kl', 0.01),
            horizon=config.get('kl_horizon', 10000)
        )
        
        self.global_step = 0
        self.tmp_lora_dir = "/tmp/adv_grpo_lora"
        if self.accelerator.is_main_process:
            os.makedirs(self.tmp_lora_dir, exist_ok=True)
            
    def sync_lora_for_vllm(self):
        lora_path = os.path.join(self.tmp_lora_dir, f"step_{self.global_step}")
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(lora_path)
            else:
                self.model.save_pretrained(lora_path)
        
        if dist.is_initialized():
            dist.barrier()
            
        return lora_path

    def generate_rollouts(self, prompts_ids: torch.Tensor, attention_mask: torch.Tensor):
        if not self.vllm_engine:
            raise ValueError("vLLM Engine not found for generation.")
            
        lora_path = self.sync_lora_for_vllm()
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(f"policy_step_{self.global_step}", self.global_step + 1, lora_path)
        
        bs = prompts_ids.shape[0]
        prompt_list = []
        for i in range(bs):
            valid_len = int(attention_mask[i].sum())
            prompt_list.append(prompts_ids[i, -valid_len:].tolist())
        
        # Repeat prompts for GRPO
        num_rollouts = self.config.get('num_rollouts', 4)
        repeated_prompts = []
        for p in prompt_list:
            repeated_prompts.extend([p] * num_rollouts)
            
        formatted_prompts = [{"prompt_token_ids": p} for p in repeated_prompts]
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            n=1, 
            max_tokens=self.config.get('max_response_length', 512), 
            temperature=self.config.get('temperature', 1.0),
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        print(f">>> [Rank {self.accelerator.local_process_index}] Generating {len(formatted_prompts)} rollouts via vLLM...")
        outputs = self.vllm_engine.generate(
            formatted_prompts, 
            sampling_params, 
            lora_request=lora_request
        )
        
        response_ids_list = []
        max_len = self.config.get('max_response_length', 512)
        for out in outputs:
            tokens = out.outputs[0].token_ids
            if len(tokens) < max_len:
                tokens += [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            response_ids_list.append(tokens)
            
        responses_tensor = torch.tensor(response_ids_list, device=self.device)
        prompts_repeated = prompts_ids.repeat_interleave(num_rollouts, dim=0)
        attention_repeated = attention_mask.repeat_interleave(num_rollouts, dim=0)
        
        # Compute exact old log probs
        with torch.no_grad():
            self.model.eval()
            full_ids = torch.cat([prompts_repeated, responses_tensor], dim=1)
            full_att = torch.cat([attention_repeated, torch.ones_like(responses_tensor)], dim=1)
            
            out = self.model(input_ids=full_ids, attention_mask=full_att)
            logits = out.logits[:, prompts_repeated.shape[1]-1:-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            old_log_probs = torch.gather(log_probs, dim=-1, index=responses_tensor.unsqueeze(-1)).squeeze(-1)
            
        response_mask = (responses_tensor != self.tokenizer.pad_token_id).float()
        uids = np.repeat(np.arange(bs), num_rollouts) # Grouping for GRPO
        
        self.model.train()
        return {
            "prompts": prompts_repeated,
            "responses": responses_tensor,
            "attention_mask": attention_repeated,
            "old_log_probs": old_log_probs,
            "response_mask": response_mask,
            "uid": uids
        }

    def train_step(self, prompts: torch.Tensor, attention_mask: torch.Tensor, non_tensor_batch: Dict):
        rollout_data = self.generate_rollouts(prompts, attention_mask)
        
        # 1. Compute rewards via provided fn
        # Expand goals
        num_rollouts = self.config.get('num_rollouts', 4)
        goals = non_tensor_batch.get('goal', ['']*prompts.shape[0])
        repeated_goals = np.repeat(goals, num_rollouts)
        
        # Group inputs for batch processing
        token_rewards = []
        rollouts_strs = []
        valid_lens = []
        for i in range(len(repeated_goals)):
            rollout_tokens = rollout_data['responses'][i]
            valid_len = int(rollout_data['response_mask'][i].sum().item())
            rollout_str = self.tokenizer.decode(rollout_tokens[:valid_len], skip_special_tokens=True)
            rollouts_strs.append(rollout_str)
            valid_lens.append(valid_len)
            
        if hasattr(self.reward_fn, 'compute_batch'):
            rw_dicts = self.reward_fn.compute_batch(repeated_goals.tolist(), rollouts_strs, None, self.global_step)
        else:
            rw_dicts = [self.reward_fn(g, r, None, self.global_step) for g, r in zip(repeated_goals, rollouts_strs)]
            
        for i, rw_dict in enumerate(rw_dicts):
            score = rw_dict.get('score', 0.0)
            valid_len = valid_lens[i]
            rw_tensor = torch.zeros(rollout_data['responses'].shape[1], device=self.device)
            if valid_len > 0:
                rw_tensor[valid_len - 1] = score
            token_rewards.append(rw_tensor)
            
        token_level_rewards = torch.stack(token_rewards)
        
        # 2. Add KL Penalty (approximated here by self-KL since no ref model in simplest version)
        ref_log_probs = rollout_data['old_log_probs'].detach() # In complete system, fetch from frozen ref
        
        final_rewards = compute_rewards(
            token_level_scores=token_level_rewards,
            old_log_prob=rollout_data['old_log_probs'],
            ref_log_prob=ref_log_probs,
            kl_ratio=self.kl_ctrl.value if self.config.get('use_kl', True) else 0.0
        )
        
        # 3. Advantages
        advantages, returns = compute_advantage(
            token_level_rewards=final_rewards,
            response_mask=rollout_data["response_mask"],
            adv_estimator=self.config.get('adv_estimator', 'grpo'),
            index=rollout_data["uid"],
            values=None
        )
        
        # 4. PPO/GRPO updates
        mini_batch_size = self.config.get('mini_batch_size', 4)
        ppo_epochs = self.config.get('ppo_epochs', 1)
        
        bs_total = rollout_data["prompts"].shape[0]
        indices = torch.randperm(bs_total).tolist()
        
        epoch_pg_loss = 0
        
        for ppo_ep in range(ppo_epochs):
            for i in range(0, bs_total, mini_batch_size):
                mb_idx = indices[i:i+mini_batch_size]
                
                mb_prompts = rollout_data['prompts'][mb_idx]
                mb_responses = rollout_data['responses'][mb_idx]
                mb_old_log_probs = rollout_data['old_log_probs'][mb_idx]
                mb_mask = rollout_data['response_mask'][mb_idx]
                mb_adv = advantages[mb_idx]
                mb_att = rollout_data['attention_mask'][mb_idx]
                
                full_ids = torch.cat([mb_prompts, mb_responses], dim=1)
                full_att = torch.cat([mb_att, torch.ones_like(mb_responses)], dim=1)
                
                out = self.model(input_ids=full_ids, attention_mask=full_att)
                logits = out.logits[:, mb_prompts.shape[1]-1:-1, :]
                
                log_probs = F.log_softmax(logits, dim=-1)
                curr_log_probs = torch.gather(log_probs, dim=-1, index=mb_responses.unsqueeze(-1)).squeeze(-1)
                
                pg_loss, _, ppo_kl, _ = compute_policy_loss(
                    old_log_prob=mb_old_log_probs,
                    log_prob=curr_log_probs,
                    advantages=mb_adv,
                    response_mask=mb_mask,
                    cliprange=self.config.get('cliprange', 0.2)
                )
                
                loss = pg_loss
                
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_pg_loss += pg_loss.item()
                
        metrics = {"loss": epoch_pg_loss / max(1, (bs_total // mini_batch_size) * ppo_epochs)}
        metrics["mean_reward"] = token_level_rewards.sum(dim=-1).mean().item()
        
        self.global_step += 1
        return metrics

    def train(self, dataloader, epochs, save_dir=None):
        for ep in range(epochs):
            print(f"Epoch {ep+1}/{epochs}")
            for batch in tqdm(dataloader):
                prompts = batch['input_ids'].to(self.device)
                att_mask = batch['attention_mask'].to(self.device)
                non_tensor = batch.get('non_tensor_batch', {})
                
                metrics = self.train_step(prompts, att_mask, non_tensor)
                
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                    
                if self.global_step % self.config.get('log_freq', 10) == 0:
                    print(f"Step {self.global_step} Output: {metrics}")
                    
            if save_dir:
                self.model.save_pretrained(os.path.join(save_dir, f"checkpoint_{ep}"))
