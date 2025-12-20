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
Core functions to implement RLVR algorithms (GRPO, RLOO, REINFORCE++).
Adapted from verl/trainer/ppo/core_algos.py for multi-turn safety alignment.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Tuple, Literal, Optional


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


def get_kl_controller(kl_ctrl_type: str, kl_coef: float = 0.001, 
                      target_kl: float = 0.01, horizon: int = 10000):
    """Get KL controller based on type."""
    if kl_ctrl_type == 'fixed':
        return FixedKLController(kl_coef=kl_coef)
    elif kl_ctrl_type == 'adaptive':
        assert horizon > 0, f'horizon must be larger than 0. Got {horizon}'
        return AdaptiveKLController(init_kl_coef=kl_coef, target_kl=target_kl, horizon=horizon)
    else:
        raise NotImplementedError(f"Unknown KL controller type: {kl_ctrl_type}")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Compute masked mean along dimension or globally if dim is None."""
    if dim is None:
        # Global mean across all dimensions
        return (tensor * mask).sum() / mask.sum().clamp(min=1e-8)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, 
                  shift_mean: bool = True) -> torch.Tensor:
    """Whiten values using masked statistics."""
    masked_values = values * mask
    count = mask.sum()
    mean = masked_values.sum() / count.clamp(min=1e-8)
    var = ((masked_values - mean * mask) ** 2 * mask).sum() / count.clamp(min=1e-8)
    std = torch.sqrt(var + 1e-8)
    if shift_mean:
        return (values - mean) / std
    return values / std


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


# ============================================================================
# Advantage Estimation Functions
# ============================================================================

def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor, 
    values: torch.Tensor, 
    response_mask: torch.Tensor,
    gamma: float = 1.0, 
    lam: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        token_level_rewards: shape (bs, response_length)
        values: shape (bs, response_length)
        response_mask: shape (bs, response_length). Token after [EOS] have mask zero.
        gamma: discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward.
    Groups responses by prompt index and normalizes within groups.
    
    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        index: array of prompt indices for grouping (bs,)
        epsilon: numerical stability constant
    
    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO (Leave-One-Out baseline).
    Based on https://arxiv.org/abs/2402.14740
    
    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        index: array of prompt indices for grouping (bs,)
        epsilon: numerical stability constant

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - \
                           id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor,
    gamma: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    Based on https://arxiv.org/abs/2501.03262
    
    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        gamma: discount factor
    
    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++ with baseline.
    Based on https://arxiv.org/abs/2501.03262
    
    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        index: array of prompt indices for grouping (bs,)
        epsilon: numerical stability constant

    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = masked_whiten(scores, response_mask)

    return scores, scores


# ============================================================================
# Loss Functions
# ============================================================================

def compute_rewards(
    token_level_scores: torch.Tensor, 
    old_log_prob: torch.Tensor, 
    ref_log_prob: torch.Tensor, 
    kl_ratio: float
) -> torch.Tensor:
    """Compute rewards with KL penalty."""
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def kl_penalty(
    logprob: torch.Tensor, 
    ref_logprob: torch.Tensor, 
    kl_penalty_type: str = 'kl'
) -> torch.Tensor:
    """
    Compute KL divergence penalty.
    
    Args:
        logprob: log probabilities from policy
        ref_logprob: log probabilities from reference
        kl_penalty_type: 'kl', 'abs', 'mse', or 'low_var_kl'
    """
    if kl_penalty_type == "kl":
        return logprob - ref_logprob

    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty_type == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty_type == 'low_var_kl':
        # J. Schulman approximation
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    raise NotImplementedError(f"Unknown KL penalty type: {kl_penalty_type}")


def compute_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange: float = 0.2,
    cliprange_low: Optional[float] = None,
    cliprange_high: Optional[float] = None,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean"
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Compute PPO policy loss with clipping.
    
    Args:
        old_log_prob: log probs from behavior policy, shape (bs, response_length)
        log_prob: log probs from current policy, shape (bs, response_length)
        advantages: advantage values, shape (bs, response_length)
        response_mask: mask for valid tokens, shape (bs, response_length)
        cliprange: PPO clipping range
        cliprange_low: lower clip bound (defaults to cliprange)
        cliprange_high: upper clip bound (defaults to cliprange)
        clip_ratio_c: dual-clip lower bound for negative advantages
        loss_agg_mode: "token-mean", "seq-mean-token-sum", or "seq-mean-token-mean"

    Returns:
        pg_loss: policy gradient loss
        pg_clipfrac: fraction of samples clipped
        ppo_kl: estimated KL divergence
        pg_clipfrac_lower: fraction clipped for negative advantages
    """
    assert clip_ratio_c > 1.0, f"clip_ratio_c should be > 1.0, got {clip_ratio_c}"

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask).item()

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask).item()

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    ).item()

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # Aggregate loss
    if loss_agg_mode == "token-mean":
        pg_loss = masked_mean(pg_losses, response_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(pg_losses * response_mask, dim=-1)
        pg_loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(pg_losses * response_mask, dim=-1) / torch.sum(response_mask, dim=-1)
        pg_loss = torch.mean(seq_losses)
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(
    logits: torch.Tensor, 
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute entropy loss for regularization.

    Args:
        logits: shape (bs, response_length, vocab_size)
        response_mask: shape (bs, response_length)

    Returns:
        entropy_loss: scalar tensor
    """
    entropy = entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor, 
    returns: torch.Tensor, 
    values: torch.Tensor, 
    response_mask: torch.Tensor, 
    cliprange_value: float = 0.2
) -> Tuple[torch.Tensor, float]:
    """
    Compute value function loss (for critic training).

    Args:
        vpreds: predicted values from current critic, shape (bs, response_length)
        returns: target returns, shape (bs, response_length)
        values: old values from behavior critic, shape (bs, response_length)
        response_mask: mask for valid tokens, shape (bs, response_length)
        cliprange_value: clipping range for value function

    Returns:
        vf_loss: value function loss
        vf_clipfrac: fraction clipped
    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask).item()
    return vf_loss, vf_clipfrac


# ============================================================================
# Advantage Estimator Enum
# ============================================================================

class AdvantageEstimator:
    """Enumeration for advantage estimation methods."""
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    RLOO = 'rloo'


def compute_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    adv_estimator: str,
    index: Optional[np.ndarray] = None,
    values: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    lam: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch to appropriate advantage computation function.
    
    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        adv_estimator: one of AdvantageEstimator values
        index: prompt indices for grouping (required for GRPO, RLOO)
        values: value estimates (required for GAE)
        gamma: discount factor
        lam: GAE lambda
    
    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    if adv_estimator == AdvantageEstimator.GAE:
        assert values is not None, "GAE requires values"
        return compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=gamma,
            lam=lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        assert index is not None, "GRPO requires index"
        return compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        assert index is not None, "RLOO requires index"
        return compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index
        )
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        return compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            gamma=gamma
        )
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        assert index is not None, "REINFORCE++ baseline requires index"
        return compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index
        )
    else:
        raise NotImplementedError(f"Unknown advantage estimator: {adv_estimator}")
