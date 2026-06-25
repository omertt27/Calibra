#!/usr/bin/env python3
"""
Calibra quality benchmark — shows that a calibra-pruned coreset has measurably
better quality metrics than random subsampling at the same keep fraction.

This is the strongest proxy for downstream policy improvement we can measure
without GPU-based policy training.

Methodology:
  1. Load a dataset (default: lerobot/pusht).
  2. Run the full Calibra pipeline to get per-episode quality scores.
  3. Select two 30% coresets:
       • Calibra coreset  — CoresetSelector (quality filter + greedy max-coverage)
       • Random coreset   — random sample, same size
  4. Run the pipeline on each coreset (and the full dataset for baseline).
  5. Print a comparison table of key quality metrics.

Usage:
    python scripts/benchmark_prune_quality.py
    python scripts/benchmark_prune_quality.py lerobot/pusht --keep 0.3
    python scripts/benchmark_prune_quality.py /data/my_dataset.h5 --keep 0.5 --format hdf5
    python scripts/benchmark_prune_quality.py lerobot/pusht --json results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from calibra.ingestion.registry import load  # noqa: E402
from calibra.pipeline import Pipeline  # noqa: E402
from calibra.pruning import CoresetSelector  # noqa: E402
from calibra.schema.episode import EpisodeBatch  # noqa: E402
from calibra.schema.report import DiagnosticReport, RiskLevel  # noqa: E402


def _extract_scalar(report: DiagnosticReport, metric_key: str) -> float | None:
    """Extract a scalar from raw_metrics across all analyzer results."""
    for r in report.analyzer_results:
        if metric_key in r.raw_metrics:
            val = r.raw_metrics[metric_key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _aggregate_per_episode(report: DiagnosticReport, key: str) -> float | None:
    """Return mean of a per-episode array stored in raw_metrics."""
    for r in report.analyzer_results:
        if key in r.raw_metrics:
            vals = r.raw_metrics[key]
            if isinstance(vals, (list, np.ndarray)) and len(vals) > 0:
                arr = np.array(vals, dtype=float)
                arr = arr[~np.isnan(arr)]
                return float(np.mean(arr)) if len(arr) > 0 else None
    return None


def _get_metrics(report: DiagnosticReport) -> dict[str, float | None]:
    return {
        "jerk_spike_rate": _aggregate_per_episode(report, "per_episode_spike_rate"),
        "vel_disc_rate": _aggregate_per_episode(report, "per_episode_vel_disc_rate"),
        "ldlj": _aggregate_per_episode(report, "per_episode_ldlj"),
        "jitter_cv": _aggregate_per_episode(report, "per_episode_jitter_cv"),
        "dropout_rate": _aggregate_per_episode(report, "per_episode_dropout_fraction"),
        "action_entropy (bits/d)": _extract_scalar(report, "action_entropy_bits_per_dim"),
        "n_critical": float(len(report.flags_at_level(RiskLevel.CRITICAL))),
        "n_warning": float(len(report.flags_at_level(RiskLevel.WARNING))),
    }


def _subsample_batch(batch: EpisodeBatch, keep_ids: list[str]) -> EpisodeBatch:
    keep_set = set(keep_ids)
    kept = [ep for ep in batch.episodes if ep.metadata.episode_id in keep_set]
    return EpisodeBatch(
        episodes=kept,
        dataset_name=batch.dataset_name + "_subset",
        format=batch.format,
        source_path=batch.source_path,
    )


def _fmt(val: float | None, metric: str) -> str:
    if val is None:
        return "  n/a"
    if "rate" in metric or "cv" in metric or "dropout" in metric:
        return f"{val * 100:6.2f}%"
    if "entropy" in metric:
        return f"{val:6.2f}"
    if "ldlj" in metric:
        return f"{val:7.2f}"
    if "critical" in metric or "warning" in metric:
        return f"{val:5.0f}"
    return f"{val:.4g}"


def _delta_arrow(full: float | None, coreset: float | None, lower_better: bool) -> str:
    if full is None or coreset is None:
        return "  —"
    delta = coreset - full
    if abs(delta) < 1e-9:
        return "  ="
    improved = (delta < 0) if lower_better else (delta > 0)
    arrow = "▼" if delta < 0 else "▲"
    color = "\033[92m" if improved else "\033[91m"
    reset = "\033[0m"
    return f" {color}{arrow}{reset}"


_LOWER_BETTER = {
    "jerk_spike_rate": True,
    "vel_disc_rate": True,
    "jitter_cv": True,
    "dropout_rate": True,
    "ldlj": False,  # less negative = better
    "action_entropy (bits/d)": False,
    "n_critical": True,
    "n_warning": True,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="lerobot/pusht",
        help="Dataset path or Hub ID (default: lerobot/pusht)",
    )
    parser.add_argument(
        "--keep",
        "-k",
        type=float,
        default=0.3,
        metavar="FRACTION",
        help="Coreset keep fraction (default: 0.3)",
    )
    parser.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random coreset (default: 42)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write results to a JSON file",
    )
    args = parser.parse_args()

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"\nLoading {args.dataset!r} ...")
    batch = load(args.dataset, reader=reader)
    log(f"  {batch.n_episodes} episodes  ·  {batch.n_samples} steps")

    log("Running pipeline on full dataset ...")
    pipeline = Pipeline()
    full_report = pipeline.run(batch)
    full_metrics = _get_metrics(full_report)

    log("Running coreset selection ...")
    selector = CoresetSelector(keep_fraction=args.keep)
    result = selector.select(batch, full_report)
    calibra_batch = _subsample_batch(batch, result.keep_episode_ids)
    log(
        f"  Calibra coreset: {calibra_batch.n_episodes} episodes ({args.keep:.0%} of {batch.n_episodes})"
    )

    rng = random.Random(args.seed)
    all_ids = [ep.metadata.episode_id for ep in batch.episodes]
    n_keep = len(result.keep_episode_ids)
    random_ids = rng.sample(all_ids, n_keep)
    random_batch = _subsample_batch(batch, random_ids)
    log(f"  Random coreset:  {random_batch.n_episodes} episodes")

    log("Running pipeline on coresets ...")
    calibra_report = pipeline.run(calibra_batch)
    random_report = pipeline.run(random_batch)

    calibra_metrics = _get_metrics(calibra_report)
    random_metrics = _get_metrics(random_report)

    # ── print comparison table ────────────────────────────────────────────────
    print()
    print("━" * 72)
    print(f"  Calibra Quality Benchmark — {args.dataset}")
    print(f"  Keep fraction: {args.keep:.0%}  ({n_keep} of {batch.n_episodes} episodes)")
    print("━" * 72)

    w = max(len(m) for m in full_metrics)
    header = (
        f"  {'Metric':<{w}}   {'Full dataset':>12}   {'Random':>12}   "
        f"{'Calibra':>12}   {'Δ vs random':>11}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for metric, full_val in full_metrics.items():
        rand_val = random_metrics.get(metric)
        calibra_val = calibra_metrics.get(metric)
        lower_is_better = _LOWER_BETTER.get(metric, True)

        full_str = _fmt(full_val, metric)
        rand_str = _fmt(rand_val, metric)
        calibra_str = _fmt(calibra_val, metric)
        arrow = _delta_arrow(rand_val, calibra_val, lower_is_better)

        print(f"  {metric:<{w}}   {full_str:>12}   {rand_str:>12}   {calibra_str:>12}  {arrow}")

    print("━" * 72)
    print()
    print("  ▼ = decreased (green = improvement)  ▲ = increased")
    print(
        "  Calibra coreset should show lower jerk_spike_rate, vel_disc_rate,\n"
        "  fewer critical/warning flags, and equal or higher action_entropy."
    )
    print()

    if args.json:
        output = {
            "dataset": args.dataset,
            "keep_fraction": args.keep,
            "n_episodes_full": batch.n_episodes,
            "n_episodes_coreset": n_keep,
            "full": {k: v for k, v in full_metrics.items()},
            "random": {k: v for k, v in random_metrics.items()},
            "calibra": {k: v for k, v in calibra_metrics.items()},
        }
        Path(args.json).write_text(json.dumps(output, indent=2))
        print(f"  Results written to {args.json}")


if __name__ == "__main__":
    main()
