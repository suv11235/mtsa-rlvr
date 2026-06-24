"""
Optional loss / PPO / reward-weight YAML or JSON for mt_rlvr_train.py.

Precedence: TrlParser result (defaults + main training YAML + CLI), then this file
fills any *allowed* keys that were **not** set on the CLI. So CLI always wins.

Supported formats: .yaml / .yml (OmegaConf if installed, else PyYAML) and .json (stdlib).
Optional nesting: a top-level ``loss:`` mapping is merged into the flat key space
(values under ``loss`` override same-named top-level keys if both exist).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Set, Tuple, Union, get_args, get_origin

# Keys we allow from a dedicated loss file (must exist on RLVRScriptArguments or be
# consumed when building RLVRConfig in mt_rlvr_train.py).
_LOSS_FILE_ALLOWED_KEYS: Tuple[str, ...] = (
    # Reward composition (not the PPO tensor, but loss-adjacent)
    "use_entropy_reward",
    "entropy_reward_weight",
    "judge_reward_weight",
    # KL in reward signal
    "use_kl_in_reward",
    "kl_coef",
    # Advantage / policy algorithm
    "adv_estimator",
    # PPO / RLVRConfig (previously not all wired from CLI)
    "entropy_coeff",
    "cliprange",
    "cliprange_value",
    "kl_penalty_type",
    "kl_ctrl_type",
    "target_kl",
    "kl_horizon",
    # Representation + capability regularizers
    "use_rep_loss",
    "rep_loss_weight",
    "use_capability_regularizer",
    "capability_dataset_name",
    "capability_dataset_config",
    "capability_split",
    "capability_weight",
    "capability_batch_size",
    "capability_max_length",
    "capability_answer_mode",
    "capability_answer_prefix",
    # TAR inner loop (affects which inner objective runs)
    "use_tamper_resistance",
    "tar_type",
    "tar_inner_loop_steps",
    "tar_inner_lr",
    "tar_inner_goal_sampling",
)

_ALLOWED = frozenset(_LOSS_FILE_ALLOWED_KEYS)


def _load_yaml_as_dict(path: Path) -> Any:
    """Load YAML to plain dict/list (OmegaConf preferred; PyYAML fallback)."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    except ImportError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "YAML loss config requires `omegaconf` (see MTSA/requirements.txt) or `pyyaml`."
            ) from e
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def load_loss_config_file(path: str) -> Dict[str, Any]:
    """Load a dict from YAML (.yaml / .yml) or JSON (.json)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"loss config file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = _load_yaml_as_dict(p)

    if not isinstance(raw, MutableMapping):
        raise ValueError(f"loss config root must be a mapping, got {type(raw)}")

    merged: Dict[str, Any] = dict(raw)
    if "loss" in merged and isinstance(merged["loss"], Mapping):
        loss_block = dict(merged.pop("loss"))
        merged.update(loss_block)

    unknown = sorted(set(merged.keys()) - _ALLOWED)
    if unknown:
        raise ValueError(
            f"Unknown keys in loss config {p}: {unknown}. "
            f"Allowed: {sorted(_ALLOWED)}"
        )
    return {k: merged[k] for k in merged if k in _ALLOWED}


def explicit_cli_longoption_names() -> Set[str]:
    """
    Collect ``--name`` / ``--name=value`` keys from sys.argv (normalized to
    dataclass field style with underscores).
    """
    names: Set[str] = set()
    for tok in sys.argv[1:]:
        if not tok.startswith("--"):
            continue
        body = tok[2:]
        if "=" in body:
            key = body.split("=", 1)[0]
        else:
            key = body
        names.add(key.replace("-", "_"))
    return names


def _unwrap_optional(field_type: Any) -> Any:
    if get_origin(field_type) is Union:
        args = tuple(a for a in get_args(field_type) if a is not type(None))
        if len(args) == 1:
            return args[0]
    return field_type


def _coerce_for_dataclass(field_type: Any, value: Any) -> Any:
    """Best-effort coerce JSON/YAML scalars into bool / int / float."""
    if value is None:
        return None
    field_type = _unwrap_optional(field_type)
    if field_type is bool or field_type == bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"Cannot coerce {value!r} to bool")
    if field_type is int or field_type == int:
        return int(value)
    if field_type is float or field_type == float:
        return float(value)
    if field_type is str or field_type == str:
        return str(value)
    return value


def merge_loss_config_into_args(args: Any, path: Optional[str], explicit_cli: Optional[Set[str]] = None) -> None:
    """
    Load ``path`` and set attributes on ``args`` for keys not present in ``explicit_cli``.

    Mutates ``args`` in place. No-op if path is None or empty.
    """
    if not path:
        return
    explicit = explicit_cli if explicit_cli is not None else explicit_cli_longoption_names()
    data = load_loss_config_file(path)
    from dataclasses import fields

    field_by_name = {f.name: f for f in fields(args.__class__)}

    for key, raw_val in data.items():
        if key in explicit:
            continue
        if key not in field_by_name:
            continue
        f = field_by_name[key]
        coerced = _coerce_for_dataclass(f.type, raw_val)
        setattr(args, key, coerced)
