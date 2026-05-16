#!/usr/bin/env python3
"""Aggregate multi-seed results from a v2-style sweep.

Layout assumed:
    <runs_root>/<label>/seed<N>/results.json

For every label, computes mean ± std across seeds for the headline metrics
and writes:
    summary_seeds.csv  — one row per (label, seed)
    summary_means.csv  — one row per label (mean across seeds, with std)
    asr_curves_mean.png — ASR(step) per label, seed-mean line + shaded band
    capability_means.png — start vs. end MMLU/GSM8K, mean ± std error bars

Standalone, CPU-only. Doesn't replace `aggregate_runs.py` — it's a thin
companion that surfaces the across-seed aggregates the v2 report needs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NUMERIC_KEYS: Tuple[str, ...] = (
    "start_asr_harmbench",
    "final_asr_harmbench",
    "final_asr_ci95_lo",
    "final_asr_ci95_hi",
    "start_mmlu",
    "end_mmlu",
    "mmlu_delta",
    "start_gsm8k",
    "end_gsm8k",
    "gsm8k_delta",
    "start_response_empty_rate",
    "final_response_empty_rate",
    "wallclock_seconds",
    "final_step",
    "steps_to_threshold",
)


def _parse_label_seed(results_path: Path) -> Tuple[str, Optional[int]]:
    parent = results_path.parent.name
    m = re.fullmatch(r"seed(\d+)", parent)
    if m:
        return results_path.parent.parent.name, int(m.group(1))
    return parent, None


def _load(results_path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"WARN: cannot read {results_path}: {e}")
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    xs = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return None, None, 0
    if len(xs) == 1:
        return float(xs[0]), 0.0, 1
    return float(statistics.mean(xs)), float(statistics.pstdev(xs)), len(xs)


def _curve_from_results(data: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Return [(step, asr)] including bookended start (0) and final entries."""
    points: List[Tuple[int, float]] = []
    start_asr = ((data.get("start") or {}).get("asr") or {}).get("asr_harmbench")
    if start_asr is not None:
        try:
            points.append((0, float(start_asr)))
        except Exception:
            pass
    for entry in data.get("asr_curve") or []:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        asr = entry.get("asr_harmbench")
        if step is None or asr is None:
            continue
        try:
            points.append((int(step), float(asr)))
        except Exception:
            pass
    summary = data.get("summary") or {}
    final_step = summary.get("final_step")
    final_asr = summary.get("final_asr_harmbench")
    if final_step is not None and final_asr is not None:
        try:
            if not points or int(final_step) != points[-1][0]:
                points.append((int(final_step), float(final_asr)))
        except Exception:
            pass
    return points


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_glob", required=True,
                   help="glob for results.json — e.g. 'eval/runs/checkpoint_compare_v2/*/seed*/results.json'")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.runs_glob))]
    if not paths:
        print(f"no results.json matched {args.runs_glob}")
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label: Dict[str, List[Dict[str, Any]]] = {}
    seed_rows: List[Dict[str, Any]] = []

    for path in paths:
        data = _load(path)
        if data is None:
            continue
        label, seed = _parse_label_seed(path)
        s = data.get("summary") or {}
        row = {"label": label, "seed": seed, "results_path": str(path)}
        for k in NUMERIC_KEYS:
            row[k] = _to_float(s.get(k))
        # Cross-fill the few metrics that live elsewhere if missing in summary.
        if row.get("start_response_empty_rate") is None:
            row["start_response_empty_rate"] = _to_float(((data.get("start") or {}).get("asr") or {}).get("empty_rate"))
        if row.get("final_response_empty_rate") is None:
            row["final_response_empty_rate"] = _to_float(((data.get("end") or {}).get("asr") or {}).get("empty_rate"))
        row["lora"] = s.get("lora_path")
        row["model"] = s.get("model_name_or_path")
        row["_curve"] = _curve_from_results(data)
        by_label.setdefault(label, []).append(row)
        seed_rows.append({k: v for k, v in row.items() if k != "_curve"})

    # --- Per-seed CSV
    seed_csv = out_dir / "summary_seeds.csv"
    keys = sorted({k for r in seed_rows for k in r.keys()})
    with seed_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(seed_rows, key=lambda x: (str(x.get("label")), x.get("seed") or -1)):
            w.writerow(r)
    print(f"wrote {seed_csv} ({len(seed_rows)} rows)")

    # --- Per-label means CSV
    means_rows: List[Dict[str, Any]] = []
    for label, rows in sorted(by_label.items()):
        out_row: Dict[str, Any] = {"label": label, "n_seeds": len(rows)}
        out_row["model"] = rows[0].get("model")
        out_row["lora"] = rows[0].get("lora")
        for k in NUMERIC_KEYS:
            mean, std, n = _mean_std([r.get(k) for r in rows])
            out_row[f"{k}_mean"] = mean
            out_row[f"{k}_std"] = std
            out_row[f"{k}_n"] = n
        means_rows.append(out_row)
    means_csv = out_dir / "summary_means.csv"
    keys = sorted({k for r in means_rows for k in r.keys()})
    with means_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in means_rows:
            w.writerow(r)
    print(f"wrote {means_csv} ({len(means_rows)} labels)")

    # --- Plots (best-effort; skip silently if matplotlib missing)
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return 0

    # ASR curves: per-label seed-mean line with min/max shading.
    plt.figure(figsize=(8, 5))
    for label, rows in sorted(by_label.items()):
        steps_to_vals: Dict[int, List[float]] = {}
        for r in rows:
            for s, v in r["_curve"]:
                steps_to_vals.setdefault(s, []).append(v)
        if not steps_to_vals:
            continue
        steps = sorted(steps_to_vals.keys())
        means = [statistics.mean(steps_to_vals[s]) for s in steps]
        mins = [min(steps_to_vals[s]) for s in steps]
        maxs = [max(steps_to_vals[s]) for s in steps]
        plt.plot(steps, means, marker="o", label=label)
        plt.fill_between(steps, mins, maxs, alpha=0.15)
    plt.xlabel("step")
    plt.ylabel("HarmBench ASR")
    plt.title("ASR per defense (seed mean ± min/max)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "asr_curves_mean.png", dpi=150)
    plt.close()

    # Capability deltas
    labels = [r["label"] for r in means_rows]
    mmlu_d = [r.get("mmlu_delta_mean") or 0.0 for r in means_rows]
    mmlu_e = [r.get("mmlu_delta_std") or 0.0 for r in means_rows]
    gsm_d = [r.get("gsm8k_delta_mean") or 0.0 for r in means_rows]
    gsm_e = [r.get("gsm8k_delta_std") or 0.0 for r in means_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], mmlu_d, width, yerr=mmlu_e, capsize=3, label="MMLU Δ")
    ax.bar([i + width / 2 for i in x], gsm_d, width, yerr=gsm_e, capsize=3, label="GSM8K Δ")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("delta (end - start)")
    ax.set_title("Capability degradation under malicious FT (seed mean ± std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "capability_means.png", dpi=150)
    plt.close(fig)
    print(f"wrote plots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
