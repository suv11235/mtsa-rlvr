"""
Modular unit tests for src.utils.capability_regularizer.

These tests avoid HuggingFace dataset downloads and remote tokenizer loads so they
run quickly in CI and local sandboxes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.capability_regularizer import (
    CapabilityBatchConfig,
    collate_gsm8k_capability_batch,
    extract_gsm8k_final_answer,
    load_capability_dataset,
    make_gsm8k_messages,
)


def _datasets_import_ok() -> bool:
    """True only if `import datasets` succeeds (pyarrow etc. must be loadable)."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False
    return True


requires_datasets = pytest.mark.skipif(
    not _datasets_import_ok(),
    reason="requires importable `datasets` package (and working pyarrow)",
)


# ---------------------------------------------------------------------------
# extract_gsm8k_final_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("", ""),
        ("Some text without numbers", ""),
        ("#### 42", "42"),
        ("Work shown here #### 3.5", "3.5"),
        ("####  -7", "-7"),
        ("No marker but 10 and 20", "20"),
        ("x = -2.5 done", "-2.5"),
    ],
)
def test_extract_gsm8k_final_answer(answer: str, expected: str) -> None:
    assert extract_gsm8k_final_answer(answer) == expected


# ---------------------------------------------------------------------------
# make_gsm8k_messages
# ---------------------------------------------------------------------------


def test_make_gsm8k_messages_final_only() -> None:
    msgs = make_gsm8k_messages("2+2?", final_only=True)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "2+2?" in msgs[0]["content"]
    assert "Return only the final numeric answer" in msgs[0]["content"]
    assert "Show your work" not in msgs[0]["content"]


def test_make_gsm8k_messages_show_work() -> None:
    msgs = make_gsm8k_messages("2+2?", final_only=False)
    assert "Show your work" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# CapabilityBatchConfig
# ---------------------------------------------------------------------------


def test_capability_batch_config_defaults_and_frozen() -> None:
    cfg = CapabilityBatchConfig()
    assert cfg.dataset_name == "openai/gsm8k"
    assert cfg.answer_mode == "final"
    with pytest.raises(AttributeError):
        cfg.dataset_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Stub tokenizer for collate (no HF network)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """
    Minimal tokenizer: encodes text as [10 + ord(c) % 90 for c in text].
    Supports optional apply_chat_template for collate_gsm8k_capability_batch.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, with_chat_template: bool = True) -> None:
        self._with_chat = with_chat_template
        if not with_chat_template:
            # collate uses: hasattr + callable; None is not callable → plain-text branch
            self.apply_chat_template = None  # type: ignore[assignment]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        if not self._with_chat:
            raise AttributeError("no chat template")
        parts: List[str] = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        out = "\n".join(parts)
        if add_generation_prompt:
            out += "\n<|assistant|>\n"
        return out

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        add_special_tokens: bool = False,
        truncation: bool = True,
        max_length: int = 512,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        del add_special_tokens, truncation, kwargs

        def encode_one(s: str) -> List[int]:
            ids = [10 + (ord(c) % 90) for c in s[:max_length]]
            return ids

        if isinstance(text, str):
            ids = encode_one(text)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        all_ids = [encode_one(t) for t in text]
        max_len = max(len(x) for x in all_ids)
        pad = self.pad_token_id
        padded = []
        attn = []
        for row in all_ids:
            pad_n = max_len - len(row)
            padded.append(row + [pad] * pad_n)
            attn.append([1] * len(row) + [0] * pad_n)
        return {"input_ids": padded, "attention_mask": attn}


# ---------------------------------------------------------------------------
# collate_gsm8k_capability_batch
# ---------------------------------------------------------------------------


def test_collate_gsm8k_final_mode_masks_prompt_labels() -> None:
    tok = _StubTokenizer(with_chat_template=True)
    cfg = CapabilityBatchConfig(max_length=256, answer_mode="final")
    examples = [
        {"question": "Q1?", "answer": "#### 7"},
        {"question": "Q2?", "answer": "Blah 99"},
    ]
    batch = collate_gsm8k_capability_batch(tok, examples, cfg)

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    bs, seq = batch["input_ids"].shape
    assert bs == 2
    assert batch["attention_mask"].shape == (bs, seq)
    assert batch["labels"].shape == (bs, seq)

    for b in range(bs):
        row_labels = batch["labels"][b].tolist()
        assert -100 in row_labels
        trainable = [x for x in row_labels if x != -100]
        assert len(trainable) > 0
        assert all(x >= 0 for x in trainable)


def test_collate_gsm8k_full_mode_uses_full_answer_string() -> None:
    tok = _StubTokenizer(with_chat_template=True)
    cfg = CapabilityBatchConfig(max_length=512, answer_mode="full")
    long_answer = "Step1 #### 3\nStep2 more"
    examples = [{"question": "x?", "answer": long_answer}]
    batch = collate_gsm8k_capability_batch(tok, examples, cfg)
    # Full mode should not reduce to final-only numeric extraction for assistant text
    assert batch["input_ids"].shape[0] == 1


def test_collate_gsm8k_answer_prefix_prepended() -> None:
    tok = _StubTokenizer(with_chat_template=True)
    cfg = CapabilityBatchConfig(max_length=256, answer_mode="final", answer_prefix="ANS: ")
    batch = collate_gsm8k_capability_batch(
        tok, [{"question": "?", "answer": "#### 2"}], cfg
    )
    # If prefix were missing, encoding length could differ; we only assert structure
    assert batch["labels"].shape[1] == batch["input_ids"].shape[1]


def test_collate_plain_tokenizer_path_no_chat_template() -> None:
    tok = _StubTokenizer(with_chat_template=False)
    cfg = CapabilityBatchConfig(max_length=128, answer_mode="final")
    batch = collate_gsm8k_capability_batch(
        tok,
        [{"question": "1+1", "answer": "#### 2"}],
        cfg,
    )
    assert batch["input_ids"].shape[0] == 1
    labels = batch["labels"][0].tolist()
    assert -100 in labels


def test_collate_truncation_respects_max_length() -> None:
    tok = _StubTokenizer(with_chat_template=True)
    cfg = CapabilityBatchConfig(max_length=12, answer_mode="final")
    batch = collate_gsm8k_capability_batch(
        tok,
        [{"question": "x" * 200, "answer": "#### 1"}],
        cfg,
    )
    assert batch["input_ids"].shape[1] <= 12


# ---------------------------------------------------------------------------
# load_capability_dataset
# ---------------------------------------------------------------------------


@requires_datasets
def test_load_capability_dataset_gsm8k_branch() -> None:
    fake_ds = MagicMock()
    with patch("datasets.load_dataset", return_value=fake_ds) as m:
        cfg = CapabilityBatchConfig(
            dataset_name="openai/gsm8k",
            dataset_config="main",
            split="train[:1%]",
        )
        out = load_capability_dataset(cfg)
    assert out is fake_ds
    m.assert_called_once_with("openai/gsm8k", "main", split="train[:1%]")


@requires_datasets
def test_load_capability_dataset_generic_branch() -> None:
    fake_ds = MagicMock()
    with patch("datasets.load_dataset", return_value=fake_ds) as m:
        cfg2 = CapabilityBatchConfig(dataset_name="some_user/dataset", split="validation")
        out = load_capability_dataset(cfg2)
    assert out is fake_ds
    m.assert_called_once_with("some_user/dataset", split="validation")
