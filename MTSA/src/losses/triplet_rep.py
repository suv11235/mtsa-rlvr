"""
Triplet representation losses for safety defense (CRL / arXiv:2506.11938).

Reference: https://github.com/samuelsimko/crl-llm-defense (code/methods/triplet/)
Paper: https://arxiv.org/pdf/2506.11938
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple, List

import torch
import torch.nn.functional as F

DistType = Literal["norm", "cosine", "mix"]


@dataclass(frozen=True)
class TripletLossConfig:
    """Weights and margins for safe/unsafe triplet + benign KL retain."""

    alpha_safe: float = 0.5
    beta_unsafe: float = 0.6
    gamma_kl: float = 0.7
    margin_safe: float = 2.0
    margin_unsafe: float = 3.0
    rep_layers: Tuple[int, ...] = tuple(range(20, 31))
    alpha_mode: Literal["all", "target"] = "all"
    cosine_scale: float = 10.0
    kl_temperature: float = 2.0
    # Per-pair distance metrics (crl-llm-defense train.sh defaults)
    safe_dist_pos: DistType = "norm"
    safe_dist_neg: DistType = "cosine"
    unsafe_dist_pos: DistType = "cosine"
    unsafe_dist_neg: DistType = "cosine"


@contextmanager
def adapters_disabled_ctx(model):
    """Context manager: run forward on base weights (LoRA adapters off)."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def get_layer_hidden_states(
    model,
    batch: dict,
    layers: Sequence[int],
) -> torch.Tensor:
    """
    Stack hidden states for selected transformer layers.

    Returns:
        Tensor [L, B, T, D] where L = len(layers).
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    return torch.stack([hidden_states[layer + 1] for layer in layers])


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool [L,B,T,D] over batch and sequence using [B,T] mask -> [L,D]."""
    mask = mask.unsqueeze(0).unsqueeze(-1).float()
    num = (h * mask).sum(dim=(1, 2))
    den = mask.sum(dim=(1, 2)).clamp_min(1.0)
    return num / den


def pairwise_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    category: DistType,
    cosine_scale: float = 10.0,
) -> torch.Tensor:
    """Token-level distance [L,B,T] (reference trainer ``dist`` categories)."""
    if category == "norm":
        return torch.norm(x - y, dim=-1, p=2)
    if category == "cosine":
        xn = F.normalize(x, dim=-1)
        yn = F.normalize(y, dim=-1)
        return 1.0 - (xn * yn).sum(dim=-1)
    if category == "mix":
        return d_mix(x, y, cosine_scale=cosine_scale)
    raise ValueError(f"Unknown distance category: {category}")


def d_mix(x: torch.Tensor, y: torch.Tensor, cosine_scale: float = 10.0) -> torch.Tensor:
    """Euclidean + cosine distance (legacy combined metric)."""
    xn = F.normalize(x, dim=-1)
    yn = F.normalize(y, dim=-1)
    return torch.norm(x - y, dim=-1) + cosine_scale * (
        1.0 - F.cosine_similarity(xn, yn, dim=-1).relu()
    )


def _masked_hinge_mean(
    hinge: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean hinge over layer x batch x token (mask [B,T] broadcast to [L,B,T])."""
    m = mask.unsqueeze(0).expand(hinge.shape[0], -1, -1).float()
    return (hinge * m).sum() / m.sum().clamp_min(1.0)


def compute_safe_triplet(
    h_safe: torch.Tensor,
    h_safe_base: torch.Tensor,
    unsafe_centroid: torch.Tensor,
    mask_safe: torch.Tensor,
    margin: float,
    dist_pos_type: DistType = "norm",
    dist_neg_type: DistType = "cosine",
    cosine_scale: float = 10.0,
) -> torch.Tensor:
    """Keep benign representations near base, far from harmful centroid."""
    d_pos = pairwise_dist(h_safe, h_safe_base, dist_pos_type, cosine_scale)
    d_neg = pairwise_dist(h_safe, unsafe_centroid, dist_neg_type, cosine_scale)
    hinge = F.relu(d_pos - d_neg + margin)
    return _masked_hinge_mean(hinge, mask_safe)


def compute_unsafe_triplet(
    h_unsafe: torch.Tensor,
    h_unsafe_base: torch.Tensor,
    unsafe_centroid: torch.Tensor,
    mask_unsafe: torch.Tensor,
    margin: float,
    dist_pos_type: DistType = "cosine",
    dist_neg_type: DistType = "cosine",
    cosine_scale: float = 10.0,
) -> torch.Tensor:
    """Cluster harmful representations; push away from base harmful."""
    d_pos = pairwise_dist(h_unsafe, unsafe_centroid, dist_pos_type, cosine_scale)
    d_neg = pairwise_dist(h_unsafe, h_unsafe_base, dist_neg_type, cosine_scale)
    hinge = F.relu(d_pos - d_neg + margin)
    return _masked_hinge_mean(hinge, mask_unsafe)


def kl_retain_on_logits(
    lora_logits: torch.Tensor,
    base_logits: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(base || policy) averaged over masked response tokens; T^2-scaled.

    Direction matches CRL Eq. 12 (KL of base model against the defense policy).
    Index-selects response tokens so the reduction is a MEAN over tokens, keeping
    the term O(1) regardless of response length. The previous implementation
    multiplied by the mask then used ``batchmean`` over [B, T, V], which divides
    only by batch size and therefore SUMS per-token KL across the response — an
    ~O(response_len) inflation that let the retain term dominate the triplet loss.
    """
    mask = response_mask.bool()
    if not mask.any():
        return (lora_logits * 0.0).sum()
    lora = lora_logits[mask]  # [N_response_tokens, V]
    base = base_logits[mask].detach()
    kl = F.kl_div(
        F.log_softmax(lora / temperature, dim=-1),
        F.softmax(base / temperature, dim=-1),
        reduction="batchmean",
    )
    return kl * (temperature**2)


def kl_retain_loss(
    model,
    benign_batch: dict,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Per-response-token KL(base || policy) on the benign batch; T^2-scaled."""
    mask = benign_batch["labels"] != -100
    out_safe = model(**benign_batch, use_cache=False)
    with adapters_disabled_ctx(model):
        base_logits = model(**benign_batch, use_cache=False).logits
    return kl_retain_on_logits(out_safe.logits, base_logits, mask, temperature)


def _resolve_rep_layers(model, cfg: TripletLossConfig) -> Tuple[List[int], List[int]]:
    """Safe layers (alpha_mode) vs unsafe target layers (CRL train.sh)."""
    n_blocks = model.config.num_hidden_layers
    unsafe_layers = list(cfg.rep_layers)
    if cfg.alpha_mode == "all":
        safe_layers = list(range(n_blocks))
    else:
        safe_layers = unsafe_layers
    return safe_layers, unsafe_layers


def compute_triplet_and_kl_losses(
    model,
    benign_batch: dict,
    harmful_batch: dict,
    cfg: TripletLossConfig,
) -> Tuple[torch.Tensor, dict]:
    """
    Full triplet + KL objective on paired benign/harmful batches.

    Returns:
        (total_loss, metrics_dict) with unscaled component losses for balancing.
    """
    safe_layers, unsafe_layers = _resolve_rep_layers(model, cfg)

    h_safe = get_layer_hidden_states(model, benign_batch, safe_layers)
    h_unsafe = get_layer_hidden_states(model, harmful_batch, unsafe_layers)
    h_harmful_for_safe = get_layer_hidden_states(model, harmful_batch, safe_layers)

    with adapters_disabled_ctx(model):
        h_safe_base = get_layer_hidden_states(model, benign_batch, safe_layers)
        h_unsafe_base = get_layer_hidden_states(model, harmful_batch, unsafe_layers)

    mask_safe = benign_batch["labels"] != -100
    mask_unsafe = harmful_batch["labels"] != -100

    harmful_centroid_safe = masked_mean(h_harmful_for_safe, mask_unsafe).unsqueeze(1).unsqueeze(2)
    unsafe_centroid = masked_mean(h_unsafe, mask_unsafe).unsqueeze(1).unsqueeze(2)

    l_safe = compute_safe_triplet(
        h_safe,
        h_safe_base,
        harmful_centroid_safe,
        mask_safe,
        cfg.margin_safe,
        dist_pos_type=cfg.safe_dist_pos,
        dist_neg_type=cfg.safe_dist_neg,
        cosine_scale=cfg.cosine_scale,
    )
    l_unsafe = compute_unsafe_triplet(
        h_unsafe,
        h_unsafe_base,
        unsafe_centroid,
        mask_unsafe,
        cfg.margin_unsafe,
        dist_pos_type=cfg.unsafe_dist_pos,
        dist_neg_type=cfg.unsafe_dist_neg,
        cosine_scale=cfg.cosine_scale,
    )
    kl = kl_retain_loss(model, benign_batch, temperature=cfg.kl_temperature)

    total = cfg.alpha_safe * l_safe + cfg.beta_unsafe * l_unsafe + cfg.gamma_kl * kl
    metrics = {
        "loss/safe_triplet": l_safe.detach(),
        "loss/unsafe_triplet": l_unsafe.detach(),
        "loss/kl_retain": kl.detach(),
        "loss/triplet_total": total.detach(),
    }
    return total, metrics
