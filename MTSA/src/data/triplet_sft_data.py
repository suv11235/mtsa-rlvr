"""
Data loading and chat tokenization for triplet / TAR+triplet SFT.

Harmful JSON formats supported:
  - TAR: ``goal`` + ``target_response``
  - Circuit Breakers: ``prompt`` + ``output``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _chat_template_tensor(tokenizer, messages, *, max_length: int) -> torch.Tensor:
    """Return 2D input_ids from apply_chat_template (transformers 4/5 compatible)."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=messages[-1]["role"] == "user" if messages else True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    if isinstance(encoded, torch.Tensor):
        return encoded
    return encoded["input_ids"]


def tokenize_chat_pair(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """Single (user, assistant) example with prompt masked in labels."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_tokens = _chat_template_tensor(
        tokenizer, messages, max_length=max_length
    )
    prompt_only = [{"role": "user", "content": prompt}]
    prompt_tokens = _chat_template_tensor(
        tokenizer, prompt_only, max_length=max_length
    )

    labels = full_tokens.clone()
    prompt_len = prompt_tokens.shape[-1]
    labels[0, :prompt_len] = -100

    seq_len = full_tokens.shape[-1]
    pad_len = max_length - seq_len

    input_ids = F.pad(full_tokens, (0, pad_len), value=tokenizer.pad_token_id).squeeze(0)
    labels = F.pad(labels, (0, pad_len), value=-100).squeeze(0)
    attention_mask = F.pad(
        torch.ones_like(full_tokens),
        (0, pad_len),
        value=0,
    ).squeeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_chat_batch(
    prompts: Sequence[str],
    responses: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> List[Dict[str, torch.Tensor]]:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    return [
        tokenize_chat_pair(p, r, tokenizer, max_length=max_length)
        for p, r in zip(prompts, responses)
    ]


def _load_json_or_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(8192)
        f.seek(0)
        first_non_ws = next((ch for ch in head if not ch.isspace()), None)

        def parse_jsonl(file_obj):
            out = []
            for line in file_obj:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                try:
                    out.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
            return out

        if first_non_ws == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                f.seek(0)
                return parse_jsonl(f)
        return parse_jsonl(f)


def normalize_harmful_record(record: dict) -> Tuple[str, str]:
    """Map TAR or Circuit Breakers fields to (prompt, harmful_output)."""
    prompt = record.get("prompt") or record.get("goal") or ""
    output = record.get("output") or record.get("target_response") or ""
    return prompt, output


def load_harmful_pairs(path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    records = _load_json_or_jsonl(path)
    if limit is not None:
        records = records[:limit]
    prompts, outputs = [], []
    for r in records:
        p, o = normalize_harmful_record(r)
        if p and o:
            prompts.append(p)
            outputs.append(o)
    return prompts, outputs


def load_ultrachat_pairs(num_samples: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    from datasets import load_dataset

    ds = (
        load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        .shuffle(seed=seed)
        .select(range(num_samples))
    )
    prompts, responses = [], []
    for item in ds:
        messages = item["messages"]
        if not messages:
            continue
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant = next(
            (m["content"] for m in messages if m["role"] == "assistant"),
            "",
        )
        if user and assistant:
            prompts.append(user)
            responses.append(assistant)
    return prompts, responses


class ChatTensorDataset(Dataset):
    def __init__(self, examples: List[Dict[str, torch.Tensor]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def collate_chat_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def get_refusal_text(goal: str) -> str:
    return (
        "I'm sorry, but I cannot fulfill this request. "
        "I am programmed to be a helpful and harmless AI assistant."
    )


def tokenize_refusal_pair(
    goal: str,
    tokenizer,
    max_length: int,
    refusal_text: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    text = refusal_text or get_refusal_text(goal)
    return tokenize_chat_pair(goal, text, tokenizer, max_length=max_length)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}
