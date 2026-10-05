#!/usr/bin/env python3
"""
Recursively scan an eval output tree for final_stats.txt files and aggregate
"Inference Time" fields written by roboverse IL eval (default_runner / act_eval).

Example:
  python scripts/aggregate_inference_time_stats.py il_outputs/a2a_noise/stack_cube/default/eval
  python scripts/aggregate_inference_time_stats.py il_outputs/a2a_noise/
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RE_AVG = re.compile(r"^\s*Average Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE)
RE_STD_DEMO = re.compile(
    r"^\s*STD of Demo Avg Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE
)
RE_MIN = re.compile(r"^\s*Min Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE)
RE_MAX = re.compile(r"^\s*Max Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE)
RE_STEPS = re.compile(
    r"^\s*Total Inference Steps:\s*(\d+)\s*$", re.MULTILINE
)
RE_DEMOS = re.compile(
    r"^\s*Number of Demos Evaluated:\s*(\d+)\s*$", re.MULTILINE
)
RE_PI_AVG = re.compile(
    r"^\s*Average Per-Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE
)
RE_PI_STD_DEMO = re.compile(
    r"^\s*STD of Demo Avg Per-Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE
)
RE_PI_MIN = re.compile(
    r"^\s*Min Per-Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE
)
RE_PI_MAX = re.compile(
    r"^\s*Max Per-Inference Time:\s*([\d.]+)\s*ms\s*$", re.MULTILINE
)
RE_PI_TOTAL = re.compile(
    r"^\s*Total Actual Inferences:\s*(\d+)\s*$", re.MULTILINE
)


@dataclass
class Record:
    path: Path
    avg_ms: float
    std_demo_ms: float | None
    min_ms: float
    max_ms: float
    total_steps: int | None
    num_demos: int | None
    # Per-inference (model forward pass only) fields
    pi_avg_ms: float | None = None
    pi_std_demo_ms: float | None = None
    pi_min_ms: float | None = None
    pi_max_ms: float | None = None
    pi_total_inferences: int | None = None


def _first_float(pattern: re.Pattern, text: str) -> float | None:
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def _first_int(pattern: re.Pattern, text: str) -> int | None:
    m = pattern.search(text)
    return int(m.group(1)) if m else None


def parse_final_stats(path: Path) -> Record | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"WARN: could not read {path}: {e}", file=sys.stderr)
        return None

    avg = _first_float(RE_AVG, text)
    if avg is None:
        return None

    std_demo = _first_float(RE_STD_DEMO, text)
    min_ms = _first_float(RE_MIN, text)
    max_ms = _first_float(RE_MAX, text)
    if min_ms is None or max_ms is None:
        return None

    return Record(
        path=path,
        avg_ms=avg,
        std_demo_ms=std_demo,
        min_ms=min_ms,
        max_ms=max_ms,
        total_steps=_first_int(RE_STEPS, text),
        num_demos=_first_int(RE_DEMOS, text),
        pi_avg_ms=_first_float(RE_PI_AVG, text),
        pi_std_demo_ms=_first_float(RE_PI_STD_DEMO, text),
        pi_min_ms=_first_float(RE_PI_MIN, text),
        pi_max_ms=_first_float(RE_PI_MAX, text),
        pi_total_inferences=_first_int(RE_PI_TOTAL, text),
    )


def iter_final_stats(root: Path) -> Iterable[Path]:
    # Match both the new "00_" prefixed name and the legacy name for safety.
    if root.is_file() and root.name in ("00_final_stats.txt", "final_stats.txt"):
        yield root
        return
    if not root.is_dir():
        return
    yield from root.rglob("00_final_stats.txt")
    yield from root.rglob("final_stats.txt")


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    mean = statistics.fmean(s)
    stdev = statistics.stdev(s) if n > 1 else 0.0
    mid = n // 2
    if n % 2:
        median = s[mid]
    else:
        median = (s[mid - 1] + s[mid]) / 2.0

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "n": float(n),
        "mean": mean,
        "std": stdev,
        "min": s[0],
        "max": s[-1],
        "median": median,
        "p25": pct(25),
        "p75": pct(75),
    }


def weighted_mean(records: list[Record], weight_key: str) -> float | None:
    num = 0.0
    den = 0.0
    for r in records:
        w = getattr(r, weight_key)
        if w is None or w <= 0:
            continue
        num += r.avg_ms * w
        den += w
    if den <= 0:
        return None
    return num / den


def main() -> int:
    p = argparse.ArgumentParser(
        description="Aggregate Inference Time stats from final_stats.txt under a directory tree."
    )
    p.add_argument(
        "root",
        type=Path,
        help="Root directory (or a single final_stats.txt path)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print one line per parsed final_stats.txt",
    )
    args = p.parse_args()
    root = args.root.expanduser().resolve()

    paths = sorted(iter_final_stats(root))
    records: list[Record] = []
    skipped = 0
    for path in paths:
        rec = parse_final_stats(path)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    print(f"Root: {root}")
    print(f"Found final_stats.txt: {len(paths)} file(s)")
    print(f"Parsed with Inference Time block: {len(records)}")
    if skipped:
        print(f"Skipped (missing Inference Time fields): {skipped}")

    if not records:
        print("No valid records — nothing to aggregate.")
        return 1

    avgs = [r.avg_ms for r in records]
    sm = summarize(avgs)
    w_steps = weighted_mean(records, "total_steps")

    print()
    print("=== Per-eval-run Average Inference Time (ms) ===")
    print(
        f"  count: {int(sm['n'])}\n"
        f"  mean:  {sm['mean']:.4f}\n"
        f"  std:   {sm['std']:.4f}   (across eval runs)\n"
        f"  min:   {sm['min']:.4f}\n"
        f"  max:   {sm['max']:.4f}\n"
        f"  median:{sm['median']:.4f}\n"
        f"  p25:   {sm['p25']:.4f}\n"
        f"  p75:   {sm['p75']:.4f}"
    )
    if w_steps is not None:
        print(f"  weighted mean (by Total Inference Steps): {w_steps:.4f}")

    std_demos = [r.std_demo_ms for r in records if r.std_demo_ms is not None]
    if std_demos:
        sd = summarize(std_demos)
        print()
        print("=== STD of Demo Avg Inference Time per run (ms) ===")
        print(
            f"  mean: {sd['mean']:.4f}\n"
            f"  std:  {sd['std']:.4f}\n"
            f"  min:  {sd['min']:.4f}\n"
            f"  max:  {sd['max']:.4f}"
        )

    print()
    print("=== Across all runs: min of per-run mins / max of per-run maxes (ms) ===")
    print(f"  global min (smallest Min among files): {min(r.min_ms for r in records):.4f}")
    print(f"  global max (largest Max among files):  {max(r.max_ms for r in records):.4f}")

    steps_total = sum(r.total_steps or 0 for r in records)
    demos_total = sum(r.num_demos or 0 for r in records)
    if steps_total:
        print()
        print(f"Total Inference Steps (sum where present): {steps_total}")
    if demos_total:
        print(
            "Number of Demos Evaluated (sum over runs, not deduplicated): "
            f"{demos_total}"
        )

    # Per-Inference Time aggregation (model forward pass only)
    pi_records = [r for r in records if r.pi_avg_ms is not None]
    if pi_records:
        pi_avgs = [r.pi_avg_ms for r in pi_records]
        pi_sm = summarize(pi_avgs)
        print()
        print("=== Per-eval-run Average Per-Inference Time (model forward pass only, ms) ===")
        print(
            f"  count: {int(pi_sm['n'])}\n"
            f"  mean:  {pi_sm['mean']:.4f}\n"
            f"  std:   {pi_sm['std']:.4f}   (across eval runs)\n"
            f"  min:   {pi_sm['min']:.4f}\n"
            f"  max:   {pi_sm['max']:.4f}\n"
            f"  median:{pi_sm['median']:.4f}\n"
            f"  p25:   {pi_sm['p25']:.4f}\n"
            f"  p75:   {pi_sm['p75']:.4f}"
        )
        pi_mins = [r.pi_min_ms for r in pi_records if r.pi_min_ms is not None]
        pi_maxs = [r.pi_max_ms for r in pi_records if r.pi_max_ms is not None]
        if pi_mins and pi_maxs:
            print()
            print("=== Across all runs: Per-Inference min/max (ms) ===")
            print(f"  global min: {min(pi_mins):.4f}")
            print(f"  global max: {max(pi_maxs):.4f}")
        pi_inferences_total = sum(r.pi_total_inferences or 0 for r in pi_records)
        if pi_inferences_total:
            print(f"Total Actual Inferences (sum where present): {pi_inferences_total}")

    if args.list:
        print()
        print("=== Per file ===")
        for r in records:
            try:
                rel = r.path.relative_to(root)
            except ValueError:
                rel = r.path
            extra = ""
            if r.total_steps is not None:
                extra += f" steps={r.total_steps}"
            if r.pi_avg_ms is not None:
                extra += f" per_infer={r.pi_avg_ms:.4f}"
            print(f"  avg={r.avg_ms:.4f} ms{extra}  {rel}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
