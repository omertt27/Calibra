"""
calibra benchmark — Closed-loop policy training benchmark and simulation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import List


from calibra.pipeline import Pipeline
from calibra.pruning import CoresetSelector
from calibra.predict import predict_outcome
from calibra.schema.episode import EpisodeBatch


_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN = "─" * _WIDTH


def run_benchmark(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra benchmark",
        description=(
            "Simulates and benchmarks training metrics for the raw dataset, "
            "a randomly-pruned baseline, and the Calibra coreset. "
            "Outputs expected compute savings and predicted success rates."
        ),
    )
    p.add_argument("path", help="Path of the source dataset to benchmark")
    p.add_argument(
        "--keep",
        "-k",
        type=float,
        default=0.3,
        help="Fraction of episodes to retain in the pruned coresets (default: 0.3)",
    )
    p.add_argument(
        "--policy",
        metavar="FAMILY",
        default="diffusion",
        help="Target policy family for success prediction (default: diffusion)",
    )
    p.add_argument(
        "--format",
        "-f",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Print raw metrics in JSON format to stdout.",
    )
    p.add_argument(
        "--base-gpu-hours",
        type=float,
        default=24.0,
        help="GPU-hours required to train on the full (100%) dataset (default: 24.0)",
    )
    args = p.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading dataset from {args.path!r} ...")

    # 1. Load dataset
    try:
        from calibra.ingestion.registry import load

        batch = load(args.path, reader=args.format)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    n_total = batch.n_episodes
    if n_total < 5:
        print(
            f"error: Dataset has only {n_total} episodes. Need at least 5 to run benchmarks.",
            file=sys.stderr,
        )
        sys.exit(1)

    k_size = max(1, round(n_total * args.keep))
    log(f"Dataset loaded: {n_total} episodes, {batch.n_samples} steps.")
    log("Running diagnostics on full (Raw) dataset ...")

    # 2. Raw dataset diagnostics and prediction
    pipeline = Pipeline()
    raw_report = pipeline.run(batch, policy_family=args.policy)
    raw_pred = predict_outcome(raw_report, policy_family=args.policy)
    raw_score = max(0.0, raw_pred.get("predicted_score", 100.0))

    # 3. Calibra coreset curation
    log(f"Running Calibra coreset selection (keep fraction: {args.keep:.2f}) ...")
    selector = CoresetSelector(keep_fraction=args.keep)
    prune_res = selector.select(batch, raw_report)

    calibra_episodes = [
        ep for ep in batch.episodes if ep.metadata.episode_id in prune_res.keep_episode_ids
    ]
    calibra_batch = EpisodeBatch(
        episodes=calibra_episodes,
        dataset_name=f"{batch.dataset_name}_calibra_coreset",
        format=batch.format,
        source_path=batch.source_path,
    )

    log("Running diagnostics on Calibra coreset ...")
    calibra_report = pipeline.run(calibra_batch, policy_family=args.policy)
    calibra_pred = predict_outcome(calibra_report, policy_family=args.policy)
    calibra_score = max(0.0, calibra_pred.get("predicted_score", 100.0))

    # 4. Random pruned baseline
    log("Running diagnostics on Randomly pruned baseline ...")
    random.seed(42)
    random_ids = random.sample([ep.metadata.episode_id for ep in batch.episodes], k_size)
    random_episodes = [ep for ep in batch.episodes if ep.metadata.episode_id in random_ids]
    random_batch = EpisodeBatch(
        episodes=random_episodes,
        dataset_name=f"{batch.dataset_name}_random_pruned",
        format=batch.format,
        source_path=batch.source_path,
    )
    random_report = pipeline.run(random_batch, policy_family=args.policy)
    random_pred = predict_outcome(random_report, policy_family=args.policy)
    random_score = max(0.0, random_pred.get("predicted_score", 100.0))

    # Compute calculations
    raw_hours = args.base_gpu_hours
    calibra_hours = raw_hours * (len(calibra_episodes) / n_total)
    random_hours = raw_hours * (k_size / n_total)
    compute_savings = 100.0 * (1.0 - (len(calibra_episodes) / n_total))

    # Compile result summary
    summary = {
        "dataset_name": batch.dataset_name,
        "policy_family": args.policy,
        "n_original": n_total,
        "keep_fraction": args.keep,
        "results": {
            "raw": {
                "n_episodes": n_total,
                "gpu_hours": round(raw_hours, 1),
                "predicted_success_rate": round(raw_score, 1),
            },
            "random": {
                "n_episodes": k_size,
                "gpu_hours": round(random_hours, 1),
                "predicted_success_rate": round(random_score, 1),
            },
            "calibra": {
                "n_episodes": len(calibra_episodes),
                "gpu_hours": round(calibra_hours, 1),
                "predicted_success_rate": round(calibra_score, 1),
            },
        },
        "compute_savings_pct": round(compute_savings, 1),
    }

    # Render training integration code block recommendations
    kept_indices_str = ",".join(
        str(i)
        for i, ep in enumerate(batch.episodes)
        if ep.metadata.episode_id in prune_res.keep_episode_ids
    )
    if batch.format == "lerobot":
        lerobot_cmd = (
            f"lerobot-train --dataset.repo_id {batch.dataset_name} "
            f'--dataset.episodes "[{kept_indices_str}]"'
        )
    else:
        lerobot_cmd = (
            f'python train.py --dataset {args.path} --coreset-indices "{kept_indices_str}"'
        )

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    # Render comparison report
    print(
        f"\n{_THICK}\n"
        f"  CALIBRA CLOSED-LOOP TRAINING BENCHMARK SIMULATION\n"
        f"{_THICK}\n"
        f"  Dataset: {batch.dataset_name}  ({n_total} episodes)\n"
        f"  Policy : {args.policy.upper()}\n"
        f"{_THIN}\n"
        f"  CURATION STRATEGY COMPARISON:\n"
        f"\n"
        f"  1. RAW DATASET (100%)\n"
        f"     - Size: {n_total} episodes\n"
        f"     - Training Time: {raw_hours:.1f} GPU-hours\n"
        f"     - Predicted Success Rate: {raw_score:.1f}%\n"
        f"\n"
        f"  2. RANDOM PRUNED ({(k_size / n_total):.0%})\n"
        f"     - Size: {k_size} episodes\n"
        f"     - Training Time: {random_hours:.1f} GPU-hours\n"
        f"     - Predicted Success Rate: {random_score:.1f}%\n"
        f"\n"
        f"  3. CALIBRA CORESET ({(len(calibra_episodes) / n_total):.0%})\n"
        f"     - Size: {len(calibra_episodes)} episodes\n"
        f"     - Training Time: {calibra_hours:.1f} GPU-hours\n"
        f"     - Predicted Success Rate: {calibra_score:.1f}%\n"
        f"{_THIN}\n"
        f"  🚀  Compute Cost Savings: {compute_savings:.1f}% saved\n"
        f"  🎯  Predicted Performance Delta: {calibra_score - random_score:+.1f}% vs. Random\n"
        f"{_THIN}\n"
        f"  RECOMMENDED TRAINING COMMAND BRIDGES:\n"
        f"  Copy and run this command to train using Calibra's coreset:\n"
        f"\n"
        f"  $ {lerobot_cmd}\n"
        f"{_THICK}"
    )
