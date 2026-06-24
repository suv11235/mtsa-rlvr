"""Unit tests for triplet representation loss (no GPU / no HF downloads)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.data.triplet_sft_data import normalize_harmful_record
from src.losses.triplet_rep import (
    TripletLossConfig,
    _masked_hinge_mean,
    compute_safe_triplet,
    compute_unsafe_triplet,
    d_mix,
    kl_retain_on_logits,
    masked_mean,
)


def test_normalize_harmful_record_tar_fields():
    p, o = normalize_harmful_record({"goal": "g", "target_response": "t"})
    assert p == "g" and o == "t"


def test_normalize_harmful_record_circuit_breakers_fields():
    p, o = normalize_harmful_record({"prompt": "p", "output": "o"})
    assert p == "p" and o == "o"


def test_d_mix_zero_for_identical_vectors():
    x = torch.randn(2, 4, 8)
    assert torch.allclose(d_mix(x, x), torch.zeros(2, 4), atol=1e-5)


def test_masked_mean_respects_mask():
    h = torch.ones(1, 2, 4, 3)  # L=1, B=2, T=4, D=3
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float)
    out = masked_mean(h, mask)
    assert out.shape == (1, 3)
    assert out[0, 0] == pytest.approx(1.0)


def test_masked_hinge_mean_averages_over_layers():
    """
    Regression: old code divided only by batch x tokens, inflating loss ~L×.

    Official repo: safe_loss.sum() / layers_safe_mask.sum() (all layers in denom).
    """
    L, B, T = 11, 2, 4
    hinge = torch.full((L, B, T), 3.0)
    mask = torch.ones(B, T)
    mean = _masked_hinge_mean(hinge, mask)
    assert mean.item() == pytest.approx(3.0)
    buggy_denom = mask.sum().item()
    assert (hinge.sum() / buggy_denom).item() == pytest.approx(3.0 * L)


def test_safe_triplet_positive_when_anchor_far_from_centroid():
    torch.manual_seed(0)
    h_safe = torch.randn(2, 1, 4, 8)
    h_base = h_safe.clone()
    centroid = torch.zeros_like(h_safe) + 10.0
    mask = torch.ones(1, 4)
    loss_near = compute_safe_triplet(
        h_safe, h_base, centroid, mask, margin=0.0, cosine_scale=1.0
    )
    loss_far = compute_safe_triplet(
        h_safe + 5.0, h_base + 5.0, centroid, mask, margin=0.0, cosine_scale=1.0
    )
    assert loss_near.item() >= 0.0
    assert loss_far.item() >= loss_near.item()


def test_kl_retain_is_per_token_mean_not_length_scaled():
    """
    Regression: old KL used mask*logits + batchmean over [B,T,V], dividing only by
    batch -> SUM of per-token KL across the response (~O(response_len) inflation).
    Per-token mean must be invariant to the number of response tokens.
    """
    torch.manual_seed(0)
    V = 32
    base_tok = torch.randn(V)
    lora_tok = base_tok + 0.5  # fixed per-token divergence

    def kl_for_len(n_tokens):
        base = base_tok.unsqueeze(0).unsqueeze(0).expand(1, n_tokens, V).clone()
        lora = lora_tok.unsqueeze(0).unsqueeze(0).expand(1, n_tokens, V).clone()
        mask = torch.ones(1, n_tokens)
        return kl_retain_on_logits(lora, base, mask, temperature=1.0).item()

    short = kl_for_len(4)
    long = kl_for_len(128)
    assert short == pytest.approx(long, rel=1e-4)  # length-invariant
    assert short > 0.0


def test_kl_retain_ignores_masked_prompt_tokens():
    torch.manual_seed(0)
    V = 16
    lora = torch.randn(1, 6, V)
    base = torch.randn(1, 6, V)
    full = torch.ones(1, 6)
    # Only the last 3 tokens are response tokens.
    partial = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.float)
    kl_partial = kl_retain_on_logits(lora, base, partial, temperature=1.0)
    # Equivalent to computing on just those 3 columns.
    kl_slice = kl_retain_on_logits(
        lora[:, 3:], base[:, 3:], torch.ones(1, 3), temperature=1.0
    )
    assert kl_partial.item() == pytest.approx(kl_slice.item(), rel=1e-5)


def test_kl_retain_zero_when_no_response_tokens():
    lora = torch.randn(1, 4, 8)
    base = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4)
    assert kl_retain_on_logits(lora, base, mask).item() == pytest.approx(0.0)


def test_triplet_config_defaults_match_crl_repo():
    """Defaults aligned with crl-llm-defense code/train.sh (triplet method)."""
    cfg = TripletLossConfig()
    assert cfg.alpha_safe == 0.5
    assert cfg.beta_unsafe == 0.6
    assert cfg.gamma_kl == 0.7
    assert cfg.margin_safe == 2.0
    assert cfg.margin_unsafe == 3.0
    assert cfg.alpha_mode == "all"
    assert cfg.safe_dist_pos == "norm"
    assert cfg.unsafe_dist_neg == "cosine"
