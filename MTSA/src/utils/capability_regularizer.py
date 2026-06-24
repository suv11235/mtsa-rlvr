import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


def extract_gsm8k_final_answer(answer: str) -> str:
    """
    GSM8K answers often contain a final marker like: '#### 42'.
    Fallback: last integer/float-like token in the string.
    """
    if not answer:
        return ""

    marker_match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    if marker_match:
        return marker_match.group(1).strip()

    candidates = re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    return candidates[-1].strip() if candidates else ""


@dataclass(frozen=True)
class CapabilityBatchConfig:
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    max_length: int = 768
    answer_prefix: str = ""
    answer_mode: str = "final"  # "final" or "full"


def load_capability_dataset(cfg: CapabilityBatchConfig):
    # Lazy import: avoids importing pyarrow-heavy `datasets` when only collate / GSM8K
    # helpers are needed (e.g. unit tests, lightweight scripts).
    from datasets import load_dataset

    if cfg.dataset_name == "openai/gsm8k":
        return load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    # Generic datasets fallback
    return load_dataset(cfg.dataset_name, split=cfg.split)


def make_gsm8k_messages(question: str, final_only: bool = True) -> List[Dict[str, str]]:
    instruction = (
        "Solve the following grade-school math problem.\n\n"
        f"{question}\n\n"
        + ("Return only the final numeric answer." if final_only else "Show your work and end with the final numeric answer.")
    )
    return [{"role": "user", "content": instruction}]


def collate_gsm8k_capability_batch(
    tokenizer,
    examples: Sequence[Dict[str, Any]],
    cfg: CapabilityBatchConfig,
) -> Dict[str, Any]:
    """
    Builds a supervised NLL batch where loss is applied only to the assistant answer tokens.
    Uses the tokenizer chat template when available.
    """
    input_id_seqs: List[List[int]] = []
    attention_seqs: List[List[int]] = []
    label_seqs: List[List[int]] = []

    for ex in examples:
        question = ex.get("question", "")
        answer = ex.get("answer", "")

        if cfg.answer_mode == "full":
            assistant_answer = answer.strip()
        else:
            final = extract_gsm8k_final_answer(answer)
            assistant_answer = f"{cfg.answer_prefix}{final}".strip()

        messages = make_gsm8k_messages(question, final_only=(cfg.answer_mode != "full"))

        if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
            # Prompt text ends with assistant header (generation prompt).
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Full text includes the assistant response, used for labels.
            full_text = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": assistant_answer}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Plain-text fallback for non-chat tokenizers.
            prompt_text = f"Question: {question}\nAnswer:"
            full_text = f"{prompt_text} {assistant_answer}"

        prompt_enc = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_length,
        )
        full_enc = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_length,
        )

        prompt_len = len(prompt_enc["input_ids"])
        full_ids = full_enc["input_ids"]
        full_attn = full_enc["attention_mask"]

        labels = list(full_ids)
        for j in range(min(prompt_len, len(labels))):
            labels[j] = -100

        input_id_seqs.append(full_ids)
        attention_seqs.append(full_attn)
        label_seqs.append(labels)

    max_len = max(len(x) for x in input_id_seqs) if input_id_seqs else 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad(seq: List[int], pad_value: int) -> List[int]:
        if len(seq) >= max_len:
            return seq
        return seq + [pad_value] * (max_len - len(seq))

    batch_input_ids = [pad(s, pad_id) for s in input_id_seqs]
    batch_attention = [pad(s, 0) for s in attention_seqs]
    batch_labels = [pad(s, -100) for s in label_seqs]

    import torch

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }
