# MTSA training and loss objectives

Reference for loss-like terms across `MTSA/src`. Paths are relative to the `MTSA/` directory unless noted.

---

## 1. Multi-turn RLVR (`src/rlvr/mt_rlvr.py` + `src/rlvr/core_algos.py`)

### Optimizer loss in `ppo_update` (scalar minimized each PPO minibatch)

When components are enabled:

\[
L = L_{\text{pg}} - \lambda_{\text{ent}}\, L_{\text{ent}} + \lambda_{\text{rep}}\, L_{\text{rep}} + \lambda_{\text{cap}}\, L_{\text{cap}}
\]

| Term | Description | Code |
|------|-------------|------|
| **\(L_{\text{pg}}\)** — PPO / clipped policy gradient | Clipped surrogate on log-ratio × advantages; dual-clip for negative advantages | `compute_policy_loss` in `src/rlvr/core_algos.py` |
| **\(L_{\text{ent}}\)** — entropy (optional regularizer) | Mean token entropy on the response slice; subtracted when `entropy_coeff > 0` (higher entropy lowers the loss) | `compute_entropy_loss` in `src/rlvr/core_algos.py` |
| **\(L_{\text{rep}}\)** — representation closeness | Mean over layers of mean L2 norm \(\|h_{\text{policy}} - h_{\text{ref}}\|\) per token; reference = same model with adapters disabled (LoRA) or separate `ref_model` | `src/rlvr/mt_rlvr.py` when `use_rep_loss` |
| **\(L_{\text{cap}}\)** — capability regularizer | HuggingFace causal LM `loss` on a separate control batch (e.g. GSM8K collate in `src/utils/capability_regularizer.py`) — masked CE / NLL on assistant tokens | `src/rlvr/mt_rlvr.py` when `capability_weight > 0` and capability iterator is set |

Defaults often leave \(L_{\text{rep}}\) and \(L_{\text{cap}}\) off (`use_rep_loss=False`, `capability_weight=0`).

### Reward shaping (not the same tensor as `L`, but part of the RL signal)

- **`compute_rewards`** in `src/rlvr/core_algos.py`:  
  `token_level_scores - kl_ratio * (old_log_prob - ref_log_prob)`  
  Adjusts token-level rewards before advantages; not added as an extra term to the PPO backward loss in `ppo_update`.

### Helpers in `core_algos.py` not wired into RLVR today

- **`kl_penalty`** — variants on logprob vs ref (`kl`, `abs`, `mse`, `low_var_kl`). Imported in `mt_rlvr.py` but **not used** there for the main loss.
- **`compute_value_loss`** — clipped value loss for a critic; **not used** in `mt_rlvr` (GAE / value head path not implemented).

---

## 2. TAR inner loop inside RLVR (`src/algorithm/objectives.py`)

Used when `use_tamper_resistance` is on and the trainer runs the inner tampering step:

| Objective | Loss | Role |
|-----------|------|------|
| **`obj_max_entropy_next_token`** | Negative mean entropy of next-token distribution (minimize → maximize entropy) | “Scramble” tampering |
| **`obj_standard_max_next_token`** | **`F.cross_entropy`** on shifted logits vs `labels` (`ignore_index=-100`) | Simulated SFT-style tampering on harmful continuation |

These are **inner-loop** objectives; the outer PPO update still uses `ppo_update` on restored weights.

---

## 3. Standalone TAR meta-training (`src/algorithm/tar_vanilla_train.py`)

| Phase | Loss | Notes |
|-------|------|--------|
| Inner (`tar_type == "entropy"`) | Negative mean entropy from `softmax` × `log_softmax` | Maximize uncertainty |
| Inner (attack path) | `outputs.loss` — HF **causal LM cross-entropy** on harmful labels | Standard NLL |
| Outer | `outer_outputs.loss` — HF **causal LM CE** on refusal labels, scaled by `tar_loss_scale` / grad accumulation | Meta “retain safety” |

---

## 4. Preference learning — MTSA RLHF (`src/trainer/mt_rlhf.py` + `src/algorithm/mt-rlhf.py`)

`MTRLTrainer.get_batch_loss_metrics` combines:

| Component | Description |
|-----------|-------------|
| **Custom DPO-style loss** | Squared error on a logit derived from chosen/rejected log-prob ratios vs reference, with reward adjustment: \((\text{logit} - \frac{1}{2\beta}(\text{chosen\_reward}-\text{rejected\_reward}))^2\) — see `dpo_loss` in `src/trainer/mt_rlhf.py` |
| **Optional RPO term** | `+ rpo_alpha * nll_loss` when `rpo_alpha` is set |
| **Optional weighting** | `losses * policy_weights` when `use_weighting` |
| **Optional aux** | `+ aux_loss_coef * aux_loss` when `aux_loss_enabled` |

---

## 5. Red-team DPO (`src/algorithm/red_team_dpo.py`)

Uses **`trl.DPOTrainer`** — standard **TRL DPO** loss (preference / log-probability contrast vs reference). No custom `dpo_loss` in that entry script.

---

## 6. Red-team SFT (`src/algorithm/red_team_sft.py`)

Uses **`trl.SFTTrainer`** — usual **causal LM cross-entropy** on masked labels (implementation inside Transformers/TRL).

---

## 7. Capability batch construction (`src/utils/capability_regularizer.py`)

No standalone loss function: builds `input_ids` / `attention_mask` / `labels` so that **`model(**batch).loss`** is masked **CE / NLL** on GSM8K-style assistant targets. Consumed from RLVR as \(L_{\text{cap}}\) above.

---

## 8. Reward / scoring (not optimizer losses by default)

These produce **scores** for RLVR or evaluation; they are **not** added to `backward()` in the training paths above unless explicitly wired elsewhere:

| Module | Role |
|--------|------|
| `src/reward/cosine_sim.py` | Cosine similarity, diversity-style rewards, etc. |
| `src/reward/selfbule.py` | Self-BLEU–style n-gram scores |
| `src/reward/armreward.py` | Classifier **score** (inference), not a training CE here |
| `src/rlvr/reward_manager/multiturn_reward.py` | Multi-turn judge / entropy / combined **rewards** for RLVR |

---

## 9. RLVR one-liner

With all optional terms on, the **scalar backpropped** in RLVR `ppo_update` is:

**PPO clipped policy loss − (entropy coeff × mean response entropy) + (rep weight × L2 hidden drift vs ref) + (capability weight × control-task CE).**

KL from policy vs reference enters **reward / advantage** construction via `compute_rewards`, not as an additional term in that same `loss` unless you add it.

---

---

## 10. Config sources for RLVR loss weights (`mt_rlvr_train.py`)

Loss-related flags live on **`RLVRScriptArguments`** and are copied into **`RLVRConfig`**.

1. **TRL / HuggingFace `TrlParser`** — defaults + optional main training YAML + **CLI** (same as before).
2. **Dedicated loss file** — `--loss_config_file path.yaml` (or `.json`). After parsing, `src/utils/loss_config_merge.py` loads the file and sets **any allowed key that does not appear on the command line** (so **CLI overrides the loss file**). The main training YAML and CLI are merged first by `TrlParser`; the loss file then fills remaining keys (so **loss file overrides parser defaults / main yaml** for those keys unless CLI set them).

Example: `MTSA/config/loss_defaults.example.yaml`.

*Last aligned with codebase layout under `MTSA/src`. Update this file when adding new training scripts or loss terms.*
