#!/usr/bin/env python3
"""Aggregate multiple `quick_eval.py` runs into one comparison report.

Reads each run's `results.json`, emits:
  - summary.csv         (one row per run)
  - asr_curves.png      (HarmBench ASR vs. step, one line per run)
  - capability_delta.png (MMLU + GSM8K start vs. end, one cluster per run)
  - tradeoff_scatter.png (final ASR vs. final capability, one point per run)
  - steps_to_threshold.png (how fast each defense breaks)
  - Optionally logs everything to a single W&B aggregator run.

Standalone, CPU-only. Safe to run on the SCC login node.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _safe_get(d: Optional[Dict[str, Any]], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _load_run(results_path: Path) -> Optional[Dict[str, Any]]:
    """Return a normalised dict per run, or None if the file is unreadable.

    Tolerant: missing summary, missing lm_eval, partial curves all OK.
    """
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"WARN: cannot read {results_path}: {e}", file=sys.stderr)
        return None

    # Multi-seed runs live at <group>/<label>/seed<N>/results.json. In that
    # layout, parent.name is "seedN" and the actual experiment label is one
    # level up. Keep the legacy single-seed layout (<label>/results.json)
    # working unchanged.
    import re as _re
    parent_name = results_path.parent.name
    if _re.fullmatch(r"seed\d+", parent_name):
        run_label = f"{results_path.parent.parent.name}-{parent_name}"
    else:
        run_label = parent_name

    summary = data.get("summary") or {}
    start_lm = _safe_get(data, "start", "lm_eval")
    end_lm = _safe_get(data, "end", "lm_eval")
    start_asr = _safe_get(data, "start", "asr") or {}
    end_asr = _safe_get(data, "end", "asr") or {}

    def _task(scores: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not scores:
            return None
        ts = scores.get("task_scores") or {}
        v = ts.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    # Build the curve. Prefer the per-step `asr_curve` block (with full metrics),
    # falling back to nothing if absent.
    raw_curve = data.get("asr_curve") or []
    curve: List[Dict[str, Any]] = []
    for entry in raw_curve:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        asr = entry.get("asr_harmbench")
        if asr is None:
            asr = entry.get("asr")
        try:
            curve.append({"step": int(step), "asr": float(asr) if asr is not None else None})
        except Exception:
            continue

    # Always include start (step 0) and end (final_step) bookends if available.
    start_asr_val = start_asr.get("asr_harmbench") or start_asr.get("asr")
    end_asr_val = end_asr.get("asr_harmbench") or end_asr.get("asr")
    final_step = summary.get("final_step")

    bookended: List[Dict[str, Any]] = []
    if start_asr_val is not None:
        bookended.append({"step": 0, "asr": float(start_asr_val)})
    bookended.extend(curve)
    if final_step is not None and end_asr_val is not None:
        try:
            if not bookended or int(final_step) != int(bookended[-1]["step"]):
                bookended.append({"step": int(final_step), "asr": float(end_asr_val)})
        except Exception:
            pass

    return {
        "run": run_label,
        "results_path": str(results_path),
        "model": summary.get("model_name_or_path") or _safe_get(data, "config", "model_name_or_path"),
        "lora": summary.get("lora_path") or _safe_get(data, "config", "lora", "lora_path"),
        "train_dataset": summary.get("train_dataset") or _safe_get(data, "config", "train_dataset"),
        "asr_prompt_set": summary.get("asr_prompt_set") or _safe_get(data, "config", "asr_prompt_set"),
        "seed": summary.get("seed") if summary.get("seed") is not None else _safe_get(data, "config", "seed"),
        "max_steps": summary.get("max_steps") or _safe_get(data, "config", "max_steps"),
        "final_step": summary.get("final_step"),
        "stopped_reason": summary.get("stopped_reason"),
        "steps_to_threshold": summary.get("steps_to_threshold"),

        "start_asr": _to_float(start_asr_val),
        "final_asr": _to_float(end_asr_val if end_asr_val is not None else summary.get("final_asr_harmbench")),
        "final_asr_ci_lo": _to_float(summary.get("final_asr_ci95_lo")),
        "final_asr_ci_hi": _to_float(summary.get("final_asr_ci95_hi")),
        "asr_n_prompts": summary.get("asr_n_prompts"),

        "mmlu_start": _task(start_lm, "mmlu") if summary.get("start_mmlu") is None else _to_float(summary.get("start_mmlu")),
        "mmlu_end":   _task(end_lm, "mmlu") if summary.get("end_mmlu") is None else _to_float(summary.get("end_mmlu")),
        "gsm8k_start": _task(start_lm, "gsm8k") if summary.get("start_gsm8k") is None else _to_float(summary.get("start_gsm8k")),
        "gsm8k_end":   _task(end_lm, "gsm8k") if summary.get("end_gsm8k") is None else _to_float(summary.get("end_gsm8k")),

        "wallclock_seconds": summary.get("wallclock_seconds"),
        "git_commit": summary.get("git_commit"),

        "_curve": bookended,
    }


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


CSV_COLUMNS: Tuple[str, ...] = (
    "run",
    "model",
    "lora",
    "start_asr",
    "final_asr",
    "final_asr_ci_lo",
    "final_asr_ci_hi",
    "asr_n_prompts",
    "steps_to_threshold",
    "stopped_reason",
    "max_steps",
    "final_step",
    "mmlu_start",
    "mmlu_end",
    "mmlu_delta",
    "gsm8k_start",
    "gsm8k_end",
    "gsm8k_delta",
    "wallclock_seconds",
    "git_commit",
    "results_path",
)


def _row_for_csv(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run": run["run"],
        "model": run["model"],
        "lora": run["lora"],
        "start_asr": run["start_asr"],
        "final_asr": run["final_asr"],
        "final_asr_ci_lo": run["final_asr_ci_lo"],
        "final_asr_ci_hi": run["final_asr_ci_hi"],
        "asr_n_prompts": run["asr_n_prompts"],
        "steps_to_threshold": run["steps_to_threshold"],
        "stopped_reason": run["stopped_reason"],
        "max_steps": run["max_steps"],
        "final_step": run["final_step"],
        "mmlu_start": run["mmlu_start"],
        "mmlu_end": run["mmlu_end"],
        "mmlu_delta": _delta(run["mmlu_end"], run["mmlu_start"]),
        "gsm8k_start": run["gsm8k_start"],
        "gsm8k_end": run["gsm8k_end"],
        "gsm8k_delta": _delta(run["gsm8k_end"], run["gsm8k_start"]),
        "wallclock_seconds": run["wallclock_seconds"],
        "git_commit": run["git_commit"],
        "results_path": run["results_path"],
    }


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_table(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("(no runs)")
        return
    cols = ("run", "final_asr", "mmlu_delta", "gsm8k_delta", "steps_to_threshold", "stopped_reason")
    widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        if math.isnan(v):
            return "NA"
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_asr_curves(runs: Sequence[Dict[str, Any]], out_path: Path, threshold: float) -> bool:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"WARN: matplotlib not available, skipping {out_path.name}: {e}", file=sys.stderr)
        return False

    fig = plt.figure(figsize=(8, 5))
    any_drawn = False
    for r in runs:
        curve = [c for c in r.get("_curve", []) if c.get("asr") is not None]
        if not curve:
            continue
        steps = [c["step"] for c in curve]
        ys = [c["asr"] for c in curve]
        plt.plot(steps, ys, marker="o", label=r["run"])
        any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return False
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.6, label=f"threshold={threshold}")
    plt.xlabel("malicious finetuning step")
    plt.ylabel("HarmBench ASR")
    plt.title("ASR vs. malicious finetuning steps")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_capability_delta(runs: Sequence[Dict[str, Any]], out_path: Path) -> bool:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        print(f"WARN: matplotlib/numpy not available: {e}", file=sys.stderr)
        return False

    labels = [r["run"] for r in runs]
    mmlu_deltas = [_delta(r["mmlu_end"], r["mmlu_start"]) or 0.0 for r in runs]
    gsm8k_deltas = [_delta(r["gsm8k_end"], r["gsm8k_start"]) or 0.0 for r in runs]
    if not labels:
        return False

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4.5))
    ax.bar(x - width / 2, mmlu_deltas, width, label="MMLU Δ (end - start)")
    ax.bar(x + width / 2, gsm8k_deltas, width, label="GSM8K Δ (end - start)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("score delta")
    ax.set_title("Capability change after malicious finetuning")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_tradeoff_scatter(runs: Sequence[Dict[str, Any]], out_path: Path) -> bool:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"WARN: matplotlib not available: {e}", file=sys.stderr)
        return False

    pts: List[Tuple[float, float, str, Optional[Tuple[float, float]]]] = []
    for r in runs:
        x = r["final_asr"]
        # Mean of MMLU/GSM8K endpoints; drop the run if both are missing.
        ys = [v for v in (r["mmlu_end"], r["gsm8k_end"]) if v is not None]
        if x is None or not ys:
            continue
        y = sum(ys) / len(ys)
        ci = None
        if r["final_asr_ci_lo"] is not None and r["final_asr_ci_hi"] is not None:
            ci = (r["final_asr_ci_lo"], r["final_asr_ci_hi"])
        pts.append((float(x), float(y), r["run"], ci))
    if not pts:
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    for x, y, label, ci in pts:
        if ci is not None:
            err_l = max(0.0, x - ci[0])
            err_h = max(0.0, ci[1] - x)
            ax.errorbar([x], [y], xerr=[[err_l], [err_h]], fmt="o", capsize=4)
        else:
            ax.scatter([x], [y])
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("final HarmBench ASR (lower is safer)")
    ax.set_ylabel("mean(end MMLU, end GSM8K)")
    ax.set_title("Safety / capability tradeoff after malicious FT")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _plot_steps_to_threshold(runs: Sequence[Dict[str, Any]], out_path: Path, threshold: float) -> bool:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        print(f"WARN: matplotlib not available: {e}", file=sys.stderr)
        return False

    labels = [r["run"] for r in runs]
    if not labels:
        return False
    vals: List[float] = []
    for r in runs:
        v = r.get("steps_to_threshold")
        try:
            vals.append(float(v) if v is not None else float("nan"))
        except Exception:
            vals.append(float("nan"))

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4.5))
    bars = ax.bar(labels, vals)
    # Annotate NaN bars.
    for bar, v, r in zip(bars, vals, runs):
        if math.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0,
                "did not\ncross",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel(f"steps until ASR ≥ {threshold}")
    ax.set_title("Adversarial robustness: steps to threshold (lower = more easily jailbroken)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate quick_eval.py runs into a comparison report.")
    p.add_argument(
        "--runs_glob",
        type=str,
        required=True,
        help="Glob over results.json files. Example: 'eval/runs/*/results.json'",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.90, help="ASR threshold for plots/labels.")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="aggregate")
    p.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="disabled")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(p).resolve() for p in glob.glob(args.runs_glob))
    if not paths:
        print(f"No results.json files matched {args.runs_glob!r}", file=sys.stderr)
        return 1

    runs: List[Dict[str, Any]] = []
    for path in paths:
        loaded = _load_run(path)
        if loaded is not None:
            runs.append(loaded)

    if not runs:
        print("All matched results.json files failed to load.", file=sys.stderr)
        return 2

    # Stable ordering by run label.
    runs.sort(key=lambda r: r["run"])

    csv_rows = [_row_for_csv(r) for r in runs]
    _write_csv(csv_rows, out_dir / "summary.csv")

    asr_ok = _plot_asr_curves(runs, out_dir / "asr_curves.png", args.threshold)
    cap_ok = _plot_capability_delta(runs, out_dir / "capability_delta.png")
    tradeoff_ok = _plot_tradeoff_scatter(runs, out_dir / "tradeoff_scatter.png")
    steps_ok = _plot_steps_to_threshold(runs, out_dir / "steps_to_threshold.png", args.threshold)

    print()
    print(f"Aggregated {len(runs)} run(s) into {out_dir}/")
    print(f"  summary.csv          : OK")
    print(f"  asr_curves.png       : {'OK' if asr_ok else 'SKIPPED'}")
    print(f"  capability_delta.png : {'OK' if cap_ok else 'SKIPPED'}")
    print(f"  tradeoff_scatter.png : {'OK' if tradeoff_ok else 'SKIPPED'}")
    print(f"  steps_to_threshold.png: {'OK' if steps_ok else 'SKIPPED'}")
    print()
    _print_table(csv_rows)

    if args.wandb_project and args.wandb_mode != "disabled":
        try:
            import wandb  # type: ignore

            wb = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=("offline" if args.wandb_mode == "offline" else "online"),
                job_type="aggregate",
            )
            try:
                table = wandb.Table(columns=list(CSV_COLUMNS))
                for r in csv_rows:
                    table.add_data(*[r.get(c) for c in CSV_COLUMNS])
                wb.log({"summary_table": table})
                for fname, ok in (
                    ("asr_curves.png", asr_ok),
                    ("capability_delta.png", cap_ok),
                    ("tradeoff_scatter.png", tradeoff_ok),
                    ("steps_to_threshold.png", steps_ok),
                ):
                    if ok:
                        wb.log({fname.replace(".png", ""): wandb.Image(str(out_dir / fname))})
                # Per-run scalar summaries for sortable W&B views.
                for r in csv_rows:
                    label = r["run"]
                    for k in ("final_asr", "mmlu_delta", "gsm8k_delta", "steps_to_threshold"):
                        v = r.get(k)
                        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                            wb.summary[f"{k}/{label}"] = v
            finally:
                wb.finish()
        except Exception as e:
            print(f"WARN: W&B logging failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
