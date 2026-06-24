"""
TAR meta-training (inner tamper simulation) + triplet representation defense (outer).

Combines:
  - MTSA ``tar_vanilla_train.py`` first-order MAML retain loop
  - Circuit Breakers triplet loss (arXiv:2506.11938) via ``src.losses.triplet_rep``

Paper references:
  - TAR: tamper-resistance-repo / MTSA TAR scripts
  - Triplet: https://arxiv.org/pdf/2506.11938
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from src.data.triplet_sft_data import (
    ChatTensorDataset,
    collate_chat_batch,
    load_harmful_pairs,
    load_ultrachat_pairs,
    move_batch_to_device,
    tokenize_chat_batch,
    tokenize_chat_pair,
    tokenize_refusal_pair,
)
from src.losses.triplet_rep import (
    TripletLossConfig,
    compute_triplet_and_kl_losses,
    kl_retain_loss,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TARTripletArguments:
    model_name_or_path: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    harmful_path: str = field(
        default="datasets/attack_target/train_attack_target_labels.json",
        metadata={"help": "Harmful JSON/JSONL (goal+target_response or prompt+output)."},
    )
    output_dir: str = field(default="./outputs/tar_triplet_sft")
    ultrachat_samples: int = field(default=2000)
    limit_harmful: Optional[int] = field(default=None)

    inner_lr: float = field(default=1e-4)
    inner_lr_samples: str = field(
        default="2e-5,4e-5,1e-4",
        metadata={
            "help": "Comma-separated inner (adversary) LRs; one is sampled per outer "
            "accumulation step (matches official TAR repo adversary_lr_samples)."
        },
    )
    outer_lr: float = field(default=2e-5)
    inner_steps: int = field(default=8)
    max_steps: int = field(default=750)
    batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    max_length: int = field(default=256)

    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    init_adapter: str = field(
        default="",
        metadata={
            "help": "Path to an existing LoRA adapter to continue training (Phase 2 of "
            "the two-phase pipeline: TAR on a Phase-1 triplet-SFT checkpoint). Must use "
            "the same LoRA config (r / target_modules / layers_to_transform). When empty, "
            "a fresh adapter is initialized."
        },
    )

    bf16: bool = field(default=True, metadata={"help": "Load model in bfloat16 (Lambda/CRL default)."})
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "QLoRA 4-bit load. Default off; bf16 full LoRA matches crl-llm-defense."},
    )
    gradient_checkpointing: bool = field(default=True)

    tar_type: str = field(
        default="attack",
        metadata={
            "help": "Inner-loop adversary. 'attack' = directed harmful SFT "
            "(obj_standard_max_next_token on the harmful target, matches the TAR "
            "paper's adversary). 'entropy' = undirected entropy-maximization "
            "(weak/degenerate, kept only for ablation)."
        },
    )
    tar_loss_scale: float = field(default=4.0)
    tar_tr_objective: str = field(
        default="max_entropy",
        metadata={
            "help": "Tamper-resistance (defender) objective. 'max_entropy' = official "
            "TAR repo (trajectory-accumulated max-entropy on heldout forget, gated at "
            "tar_tr_lower_bound). 'refusal_ce' = legacy endpoint refusal CE (ablation)."
        },
    )
    tar_tr_lower_bound: float = field(
        default=-11.76,
        metadata={
            "help": "Only apply accumulated TR meta-grads when summed tr_loss >= this "
            "value (= -log|V| for Llama-3, official repo default)."
        },
    )
    use_tr_weight_schedule: bool = field(
        default=False,
        metadata={
            "help": "If True, weight each inner TR step by exp(0.5*(i-(K-1))) / K; "
            "else uniform 1/K (official default when use_weighting_schedule=False)."
        },
    )
    retain_kl_scale: float = field(
        default=1.0,
        metadata={
            "help": "Weight on the KL-vs-base retain term in the outer step (TAR paper's "
            "original lambda_retain capability anchor = KL(base||policy) on benign "
            "response tokens, computed on the tampered weights). Only applied in "
            "pure-TAR mode (use_triplet_loss=False); when the triplet bundle is on, its "
            "own kl_retain component already provides this. 0 disables."
        },
    )
    use_triplet_loss: bool = field(default=True)
    triplet_loss_scale: float = field(
        default=1.0,
        metadata={
            "help": "Scale CRL triplet+KL vs TAR retain. "
            "Outer step minimizes (tar_loss_scale * CE + triplet_loss_scale * triplet). "
            "Use 1.0 with tar_loss_scale=1.0 for 1:1 weighted outer terms."
        },
    )
    triplet_timing: str = field(
        default="pre_tamper",
        metadata={
            "help": "When to evaluate triplet: pre_tamper (CRL-style, on θ before inner "
            "loop) or post_tamper (after inner loop, with retain CE)."
        },
    )

    alpha_mode: str = field(
        default="all",
        metadata={
            "help": "CRL alpha_mode: 'all' = safe triplet over all transformer layers, "
            "'target' = safe and unsafe both use rep_layers only."
        },
    )

    alpha_safe: float = field(default=0.5)
    beta_unsafe: float = field(default=0.6)
    gamma_kl: float = field(default=0.7)
    margin_safe: float = field(default=2.0)
    margin_unsafe: float = field(default=3.0)
    kl_temperature: float = field(default=1.0)
    rep_layers: List[int] = field(default_factory=lambda: list(range(20, 31)))

    max_grad_norm: float = field(default=1.0)

    log_every: int = field(default=5)
    save_every: int = field(default=50)

    run_name: str = field(default="tar-triplet-sft")
    wandb_project: str = field(default="mtsa-tar-triplet")
    wandb_mode: str = field(
        default="online",
        metadata={"help": "online | offline | disabled"},
    )


def _inner_entropy_loss(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(-1)
    if attention_mask is not None:
        entropy = entropy * attention_mask
        return -(entropy.sum() / attention_mask.sum().clamp_min(1.0))
    return -entropy.mean()


def _max_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Negative mean entropy (official tamper-resistance-repo objectives.max_entropy_loss)."""
    softmax = F.softmax(logits, dim=-1)
    log_softmax = F.log_softmax(logits, dim=-1)
    entropy = torch.sum(-softmax * log_softmax, dim=-1).mean()
    return entropy * -1


def _tr_step_weight(inner_i: int, inner_steps: int, use_schedule: bool, schedule_lambda: float = 0.5) -> float:
    if use_schedule:
        return math.exp(schedule_lambda * (inner_i - (inner_steps - 1))) / inner_steps
    return 1.0 / inner_steps


def _sample_inner_lr(samples: str, fallback: float) -> float:
    try:
        choices = [float(x.strip()) for x in samples.split(",") if x.strip()]
    except ValueError:
        return fallback
    return random.choice(choices) if choices else fallback


def _accumulate_param_grads(
    buffer: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if name in buffer:
                buffer[name] = buffer[name] + param.grad
            else:
                buffer[name] = param.grad.clone()


def _add_buffer_to_grads(model: torch.nn.Module, buffer: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in buffer:
            continue
        if param.grad is None:
            param.grad = buffer[name].clone()
        else:
            param.grad = param.grad + buffer[name]


def _run_official_tar_inner_loop(
    model: torch.nn.Module,
    adversary_batch: dict,
    meta_forget_batch: dict,
    inner_steps: int,
    inner_lr: float,
    tar_loss_scale: float,
    tar_tr_lower_bound: float,
    use_tr_weight_schedule: bool,
    gradient_accumulation_steps: int,
) -> Tuple[float, float, float, Dict[str, torch.Tensor], bool]:
    """
    Official TAR inner loop (tamper-resistance-repo inner_loop_step + tar_training_loop).

    Per inner step:
      1. Adversary: directed harmful SFT (AdamW step).
      2. Defender: max-entropy TR loss on heldout forget; accumulate meta-grads.

    Returns (total_tr_loss_metric, last_inner_adv_loss, last_tr_diagnostic_ce,
             accumulated_tr_grads, apply_tr_grad).
    """
    original_state = {
        n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
    }
    accumulated_grads: Dict[str, torch.Tensor] = {}
    inner_optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=inner_lr,
    )

    total_tr_loss = 0.0
    last_adv_loss = 0.0
    last_diagnostic_ce = 0.0

    for inner_i in range(inner_steps):
        model.zero_grad(set_to_none=True)
        adv_out = model(**adversary_batch)
        adv_loss = adv_out.loss
        last_adv_loss = adv_loss.item()
        adv_loss.backward()
        inner_optimizer.step()
        model.zero_grad(set_to_none=True)

        step_weight = _tr_step_weight(inner_i, inner_steps, use_tr_weight_schedule)
        tr_scale = tar_loss_scale * step_weight

        tr_out = model(**meta_forget_batch)
        me_loss = _max_entropy_loss(tr_out.logits)
        last_diagnostic_ce = tr_out.loss.item()

        tr_term = (me_loss * tr_scale) / gradient_accumulation_steps
        total_tr_loss += tr_term.item()
        tr_term.backward()
        _accumulate_param_grads(accumulated_grads, model)
        model.zero_grad(set_to_none=True)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(original_state[name])

    apply_tr_grad = total_tr_loss >= tar_tr_lower_bound
    return total_tr_loss, last_adv_loss, last_diagnostic_ce, accumulated_grads, apply_tr_grad


def _validate_triplet_timing(timing: str) -> str:
    timing = timing.lower().strip()
    if timing not in ("post_tamper", "pre_tamper"):
        raise ValueError(
            f"triplet_timing must be 'post_tamper' or 'pre_tamper', got {timing!r}"
        )
    return timing


def _validate_alpha_mode(mode: str) -> str:
    mode = mode.lower().strip()
    if mode not in ("all", "target"):
        raise ValueError(f"alpha_mode must be 'all' or 'target', got {mode!r}")
    return mode


def _validate_tar_tr_objective(obj: str) -> str:
    obj = obj.lower().strip()
    if obj not in ("max_entropy", "refusal_ce"):
        raise ValueError(f"tar_tr_objective must be 'max_entropy' or 'refusal_ce', got {obj!r}")
    return obj


def _maybe_init_wandb(args: TARTripletArguments):
    """Initialize wandb when API key is present and mode is not disabled."""
    if args.wandb_mode == "disabled" or not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_NAME", args.run_name)
    os.environ["WANDB_MODE"] = args.wandb_mode
    return wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=os.environ["WANDB_NAME"],
        config=asdict(args),
    )


def main():
    parser = HfArgumentParser(TARTripletArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.triplet_timing = _validate_triplet_timing(args.triplet_timing)
    args.alpha_mode = _validate_alpha_mode(args.alpha_mode)
    args.tar_tr_objective = _validate_tar_tr_objective(args.tar_tr_objective)
    os.makedirs(args.output_dir, exist_ok=True)
    wandb_run = _maybe_init_wandb(args)

    logger.info(
        "TAR+triplet config: tar_tr_objective=%s tar_type=%s inner_steps=%d "
        "triplet_timing=%s triplet_loss_scale=%.3f tar_loss_scale=%.1f "
        "retain_kl_scale=%.1f bf16=%s load_in_4bit=%s grad_ckpt=%s",
        args.tar_tr_objective,
        args.tar_type,
        args.inner_steps,
        args.triplet_timing,
        args.triplet_loss_scale,
        args.tar_loss_scale,
        args.retain_kl_scale,
        args.bf16,
        args.load_in_4bit,
        args.gradient_checkpointing,
    )

    triplet_cfg = TripletLossConfig(
        alpha_safe=args.alpha_safe,
        beta_unsafe=args.beta_unsafe,
        gamma_kl=args.gamma_kl,
        margin_safe=args.margin_safe,
        margin_unsafe=args.margin_unsafe,
        kl_temperature=args.kl_temperature,
        rep_layers=tuple(args.rep_layers),
        alpha_mode=args.alpha_mode,
    )

    logger.info("Loading tokenizer: %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_token = os.environ.get("HF_TOKEN")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        logger.info("Loading model (4-bit QLoRA)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Loading model (bf16 LoRA, no quant)")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    if args.init_adapter:
        from peft import PeftModel

        logger.info("Resuming from Phase-1 adapter (trainable): %s", args.init_adapter)
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        rep_max = max(args.rep_layers)
        lora_layers = list(range(rep_max + 1))
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            layers_to_transform=lora_layers,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    harmful_prompts, harmful_outputs = load_harmful_pairs(args.harmful_path, args.limit_harmful)
    benign_prompts, benign_responses = load_ultrachat_pairs(args.ultrachat_samples)

    harmful_data = tokenize_chat_batch(
        harmful_prompts, harmful_outputs, tokenizer, max_length=args.max_length
    )
    benign_data = tokenize_chat_batch(
        benign_prompts, benign_responses, tokenizer, max_length=args.max_length
    )

    harmful_loader = DataLoader(
        ChatTensorDataset(harmful_data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_chat_batch,
    )
    benign_loader = DataLoader(
        ChatTensorDataset(benign_data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_chat_batch,
    )
    harmful_iter = iter(harmful_loader)
    benign_iter = iter(benign_loader)

    optimizer = AdamW(model.parameters(), lr=args.outer_lr)
    model.train()
    device = model.device

    pair_index = 0

    for step in range(args.max_steps):
        optimizer.zero_grad()
        total_meta = 0.0
        total_triplet = 0.0
        total_triplet_raw = 0.0
        total_retain = 0.0
        total_safe = 0.0
        total_unsafe = 0.0
        total_kl = 0.0
        total_retain_kl = 0.0
        total_retain_kl_raw = 0.0
        total_tr_loss = 0.0
        total_inner_adv = 0.0
        total_tr_diagnostic = 0.0
        tr_grad_applied_count = 0

        for accum_i in range(args.gradient_accumulation_steps):
            try:
                harmful_batch = next(harmful_iter)
            except StopIteration:
                harmful_iter = iter(harmful_loader)
                harmful_batch = next(harmful_iter)
            try:
                meta_forget_batch = next(harmful_iter)
            except StopIteration:
                harmful_iter = iter(harmful_loader)
                meta_forget_batch = next(harmful_iter)
            try:
                benign_batch = next(benign_iter)
            except StopIteration:
                benign_iter = iter(benign_loader)
                benign_batch = next(benign_iter)

            harmful_batch = move_batch_to_device(harmful_batch, device)
            meta_forget_batch = move_batch_to_device(meta_forget_batch, device)
            benign_batch = move_batch_to_device(benign_batch, device)

            tri_loss = None
            tri_metrics = {}
            triplet_raw = 0.0
            if args.use_triplet_loss and args.triplet_timing == "pre_tamper":
                tri_loss, tri_metrics = compute_triplet_and_kl_losses(
                    model, benign_batch, harmful_batch, triplet_cfg
                )
                triplet_raw = tri_loss.item()

            if args.tar_tr_objective == "max_entropy":
                inner_lr = _sample_inner_lr(args.inner_lr_samples, args.inner_lr)
                tr_loss_metric, inner_adv, tr_diag, tr_grads, apply_tr = (
                    _run_official_tar_inner_loop(
                        model=model,
                        adversary_batch=harmful_batch,
                        meta_forget_batch=meta_forget_batch,
                        inner_steps=args.inner_steps,
                        inner_lr=inner_lr,
                        tar_loss_scale=args.tar_loss_scale,
                        tar_tr_lower_bound=args.tar_tr_lower_bound,
                        use_tr_weight_schedule=args.use_tr_weight_schedule,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                    )
                )
                total_tr_loss += tr_loss_metric
                total_inner_adv += inner_adv
                total_tr_diagnostic += tr_diag
                if apply_tr:
                    tr_grad_applied_count += 1

                # Retain KL first, then add accumulated TR meta-grads (official repo order:
                # retain backward -> add TR grads from storage -> optimizer.step).
                retain_kl_raw = 0.0
                retain_kl_term = torch.tensor(0.0, device=device)
                if not args.use_triplet_loss and args.retain_kl_scale > 0:
                    kl_val = kl_retain_loss(
                        model, benign_batch, temperature=args.kl_temperature
                    )
                    retain_kl_raw = kl_val.item()
                    retain_kl_term = (
                        kl_val * args.retain_kl_scale
                    ) / args.gradient_accumulation_steps
                    if retain_kl_term.requires_grad:
                        retain_kl_term.backward()

                if apply_tr:
                    _add_buffer_to_grads(model, tr_grads)

                triplet_loss = torch.tensor(0.0, device=device)
                if args.use_triplet_loss:
                    if args.triplet_timing == "pre_tamper":
                        triplet_loss = (
                            tri_loss * args.triplet_loss_scale
                        ) / args.gradient_accumulation_steps
                        triplet_loss.backward()
                    elif args.triplet_timing == "post_tamper":
                        tri_loss, tri_metrics = compute_triplet_and_kl_losses(
                            model, benign_batch, harmful_batch, triplet_cfg
                        )
                        triplet_raw = tri_loss.item()
                        triplet_loss = (
                            tri_loss * args.triplet_loss_scale
                        ) / args.gradient_accumulation_steps
                        triplet_loss.backward()
                    if tri_metrics:
                        total_safe += tri_metrics["loss/safe_triplet"].item()
                        total_unsafe += tri_metrics["loss/unsafe_triplet"].item()
                        total_kl += tri_metrics["loss/kl_retain"].item()

                total_meta += tr_loss_metric
                total_triplet += triplet_loss.item() if args.use_triplet_loss else 0.0
                total_triplet_raw += triplet_raw
                total_retain_kl += retain_kl_term.item()
                total_retain_kl_raw += retain_kl_raw

            else:
                # Legacy endpoint refusal-CE path (ablation).
                original_state = {
                    n: p.data.clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                }

                inner_lr = _sample_inner_lr(args.inner_lr_samples, args.inner_lr)
                inner_optimizer = AdamW(
                    (p for p in model.parameters() if p.requires_grad),
                    lr=inner_lr,
                )
                for _inner in range(args.inner_steps):
                    if args.tar_type == "entropy":
                        out = model(
                            input_ids=harmful_batch["input_ids"],
                            attention_mask=harmful_batch["attention_mask"],
                        )
                        inner_loss = _inner_entropy_loss(out, harmful_batch["attention_mask"])
                    else:
                        out = model(**harmful_batch)
                        inner_loss = out.loss
                    inner_optimizer.zero_grad()
                    inner_loss.backward()
                    inner_optimizer.step()

                goal = harmful_prompts[pair_index % len(harmful_prompts)]
                pair_index += 1
                refusal_batch = tokenize_refusal_pair(goal, tokenizer, args.max_length)
                refusal_batch = {
                    k: v.unsqueeze(0).to(device) for k, v in refusal_batch.items()
                }

                ref_out = model(**refusal_batch)
                retain_loss = (ref_out.loss * args.tar_loss_scale) / args.gradient_accumulation_steps

                triplet_loss = torch.tensor(0.0, device=device)
                triplet_raw = 0.0
                if args.use_triplet_loss:
                    if args.triplet_timing == "post_tamper":
                        tri_loss, tri_metrics = compute_triplet_and_kl_losses(
                            model, benign_batch, harmful_batch, triplet_cfg
                        )
                    triplet_raw = tri_loss.item()
                    triplet_loss = (
                        tri_loss * args.triplet_loss_scale
                    ) / args.gradient_accumulation_steps
                    if tri_metrics:
                        total_safe += tri_metrics["loss/safe_triplet"].item()
                        total_unsafe += tri_metrics["loss/unsafe_triplet"].item()
                        total_kl += tri_metrics["loss/kl_retain"].item()

                retain_kl_term = torch.tensor(0.0, device=device)
                retain_kl_raw = 0.0
                if not args.use_triplet_loss and args.retain_kl_scale > 0:
                    kl_val = kl_retain_loss(
                        model, benign_batch, temperature=args.kl_temperature
                    )
                    retain_kl_raw = kl_val.item()
                    retain_kl_term = (
                        kl_val * args.retain_kl_scale
                    ) / args.gradient_accumulation_steps

                outer_loss = retain_loss + triplet_loss + retain_kl_term
                outer_loss.backward()

                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(original_state[n])

                total_meta += retain_loss.item()
                total_triplet += triplet_loss.item() if args.use_triplet_loss else 0.0
                total_triplet_raw += triplet_raw
                total_retain += ref_out.loss.item()
                total_retain_kl += retain_kl_term.item()
                total_retain_kl_raw += retain_kl_raw

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        if step % args.log_every == 0:
            n = args.gradient_accumulation_steps
            triplet_raw_avg = total_triplet_raw / n
            retain_kl_raw_avg = total_retain_kl_raw / n
            outer_total = total_meta + total_triplet + total_retain_kl
            triplet_pct = 100.0 * total_triplet / outer_total if outer_total > 0 else 0.0
            if args.tar_tr_objective == "max_entropy":
                logger.info(
                    "step %d/%d | tar_tr=max_entropy inner_steps=%d | "
                    "tr_loss=%.4f inner_adv=%.4f tr_diag_ce=%.4f tr_grad=%d/%d | "
                    "retain_kl=%.4f (w=%.4f) | triplet_raw=%.4f | outer=%.4f",
                    step,
                    args.max_steps,
                    args.inner_steps,
                    total_tr_loss / n,
                    total_inner_adv / n,
                    total_tr_diagnostic / n,
                    tr_grad_applied_count,
                    n,
                    retain_kl_raw_avg,
                    total_retain_kl,
                    triplet_raw_avg,
                    outer_total,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log(
                        {
                            "tr_max_entropy_loss": total_tr_loss / n,
                            "inner_adv_loss": total_inner_adv / n,
                            "tr_diagnostic_ce": total_tr_diagnostic / n,
                            "tr_grad_applied": tr_grad_applied_count / n,
                            "retain_kl_raw": retain_kl_raw_avg,
                            "w_retain_kl": total_retain_kl,
                            "triplet_raw": triplet_raw_avg,
                            "outer_total": outer_total,
                            "outer_lr": args.outer_lr,
                        },
                        step=step,
                    )
            else:
                logger.info(
                    "step %d/%d | tar_type=%s timing=%s | refusal_ce=%.4f | "
                    "retain_kl=%.4f (w=%.4f) | triplet_raw=%.4f safe=%.4f unsafe=%.4f kl=%.4f | "
                    "w_tar=%.4f w_triplet=%.4f | outer=%.4f | triplet%%=%.1f",
                    step,
                    args.max_steps,
                    args.tar_type,
                    args.triplet_timing,
                    total_retain / n,
                    retain_kl_raw_avg,
                    total_retain_kl,
                    triplet_raw_avg,
                    total_safe / n if args.use_triplet_loss else 0.0,
                    total_unsafe / n if args.use_triplet_loss else 0.0,
                    total_kl / n if args.use_triplet_loss else 0.0,
                    total_meta,
                    total_triplet,
                    outer_total,
                    triplet_pct,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log(
                        {
                            "retain_ce": total_retain / n,
                            "retain_kl_raw": retain_kl_raw_avg,
                            "w_retain_kl": total_retain_kl,
                            "triplet_raw": triplet_raw_avg,
                            "loss/safe_triplet": total_safe / n if args.use_triplet_loss else 0.0,
                            "loss/unsafe_triplet": total_unsafe / n if args.use_triplet_loss else 0.0,
                            "loss/kl_retain": total_kl / n if args.use_triplet_loss else 0.0,
                            "w_tar": total_meta,
                            "w_triplet": total_triplet,
                            "outer_total": outer_total,
                            "triplet_pct": triplet_pct,
                            "outer_lr": args.outer_lr,
                        },
                        step=step,
                    )
            if step == 0 and args.use_triplet_loss and triplet_raw_avg > 0:
                balanced_scale = total_meta / triplet_raw_avg
                logger.info(
                    "step 0 balance hint: triplet_loss_scale=%.2f for ~equal w_tar/w_triplet "
                    "(current=%.2f)",
                    balanced_scale,
                    args.triplet_loss_scale,
                )

        if step > 0 and step % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt)
            logger.info("saved %s", ckpt)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete -> %s", args.output_dir)
    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
