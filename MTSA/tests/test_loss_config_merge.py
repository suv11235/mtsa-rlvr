"""Tests for src.utils.loss_config_merge."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.utils.loss_config_merge import (
    explicit_cli_longoption_names,
    load_loss_config_file,
    merge_loss_config_into_args,
)


@dataclass
class _DummyArgs:
    loss_config_file: str | None = None
    rep_loss_weight: float = 1.0
    use_rep_loss: bool = False
    kl_coef: float = 0.001
    entropy_coeff: float = 0.0


def test_load_loss_config_json_flat(tmp_path: Path) -> None:
    p = tmp_path / "loss.json"
    p.write_text(json.dumps({"rep_loss_weight": 0.25, "use_rep_loss": True}), encoding="utf-8")
    d = load_loss_config_file(str(p))
    assert d["rep_loss_weight"] == 0.25
    assert d["use_rep_loss"] is True


def test_load_loss_config_yaml_nested_loss(tmp_path: Path) -> None:
    p = tmp_path / "loss.yaml"
    p.write_text(
        "loss:\n  rep_loss_weight: 0.9\n  kl_coef: 0.002\ntop_ignored: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys"):
        load_loss_config_file(str(p))

    p2 = tmp_path / "loss2.yaml"
    p2.write_text("loss:\n  rep_loss_weight: 0.9\n  kl_coef: 0.002\n", encoding="utf-8")
    d = load_loss_config_file(str(p2))
    assert d["rep_loss_weight"] == 0.9
    assert d["kl_coef"] == 0.002


def test_merge_cli_overrides_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"rep_loss_weight": 0.3, "entropy_coeff": 0.05}), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--rep_loss_weight", "2.0"])

    # Simulate TrlParser already having applied CLI: rep_loss_weight=2.0 on args
    args = _DummyArgs(rep_loss_weight=2.0, entropy_coeff=0.0)
    merge_loss_config_into_args(args, str(cfg), explicit_cli=explicit_cli_longoption_names())
    assert args.rep_loss_weight == 2.0  # unchanged: CLI blocked file overwrite
    assert args.entropy_coeff == 0.05  # from file (not on CLI)


def test_merge_file_fills_when_cli_silent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"rep_loss_weight": 0.4}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog"])

    args = _DummyArgs(rep_loss_weight=1.0)
    merge_loss_config_into_args(args, str(cfg), explicit_cli=explicit_cli_longoption_names())
    assert args.rep_loss_weight == 0.4
