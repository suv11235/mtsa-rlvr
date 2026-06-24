"""
Standalone triplet defense SFT (no TAR inner loop).

Implementation lives in modular ``src.losses`` + ``src.data``; this script is the
paper-faithful entry point (arXiv:2506.11938).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.data.triplet_sft_data import (
    ChatTensorDataset,
    collate_chat_batch,
    load_harmful_pairs,
    load_ultrachat_pairs,
    move_batch_to_device,
    tokenize_chat_batch,
)
from src.losses.triplet_rep import TripletLossConfig, compute_triplet_and_kl_losses


class PairBatchLoader:
    def __init__(self, benign_loader, harmful_loader):
        self.benign_loader = benign_loader
        self.harmful_loader = harmful_loader
        self._len = min(len(benign_loader), len(harmful_loader))

    def __iter__(self):
        return zip(self.benign_loader, self.harmful_loader)

    def __len__(self):
        return self._len


class TripletTrainer(Trainer):
    def __init__(self, benign_ds, harmful_ds, triplet_cfg: TripletLossConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.harmful_ds = harmful_ds
        self.triplet_cfg = triplet_cfg

    def get_train_dataloader(self):
        args = self.args
        benign_loader = DataLoader(
            self.benign_ds,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_chat_batch,
        )
        harmful_loader = DataLoader(
            self.harmful_ds,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_chat_batch,
        )
        return PairBatchLoader(benign_loader, harmful_loader)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign, harmful = inputs
        device = model.device
        benign = move_batch_to_device(benign, device)
        harmful = move_batch_to_device(harmful, device)

        loss, metrics = compute_triplet_and_kl_losses(
            model, benign, harmful, self.triplet_cfg
        )
        self.log({k: v.item() for k, v in metrics.items()})
        return (loss, None) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--harmful_path", type=str, required=True)
    parser.add_argument("--ultrachat_samples", type=int, default=5000)
    parser.add_argument("--limit_harmful", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--mb", type=float, default=2.0)
    parser.add_argument("--mh", type=float, default=3.0)
    parser.add_argument("--rep_layers", type=int, nargs="*", default=list(range(20, 31)))
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="triplet-defense")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    triplet_cfg = TripletLossConfig(
        alpha_safe=args.alpha,
        beta_unsafe=args.beta,
        gamma_kl=args.gamma,
        margin_safe=args.mb,
        margin_unsafe=args.mh,
        rep_layers=tuple(args.rep_layers),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    benign_prompts, benign_responses = load_ultrachat_pairs(args.ultrachat_samples)
    harmful_prompts, harmful_outputs = load_harmful_pairs(args.harmful_path, args.limit_harmful)

    benign_data = tokenize_chat_batch(
        benign_prompts, benign_responses, tokenizer, max_length=args.max_length
    )
    harmful_data = tokenize_chat_batch(
        harmful_prompts, harmful_outputs, tokenizer, max_length=args.max_length
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=1.0,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    trainer = TripletTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        benign_ds=ChatTensorDataset(benign_data),
        harmful_ds=ChatTensorDataset(harmful_data),
        triplet_cfg=triplet_cfg,
    )
    trainer.train()

    model.save_pretrained(output_dir / "lora_adapter")
    with open(output_dir / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    (output_dir / "READY").touch()


if __name__ == "__main__":
    main()
