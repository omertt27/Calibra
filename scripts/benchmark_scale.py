#!/usr/bin/env python3
"""
Calibra scale benchmark — measures pipeline and pruning wall-clock time
at synthetic dataset sizes from 1k to 500k episodes.

No disk I/O: episodes are generated as random numpy arrays in memory.
This isolates the compute cost of the analyzers and coreset selectors.

Usage:
    python scripts/benchmark_scale.py
    python scripts/benchmark_scale.py --sizes 1000,10000,100000
    python scripts/benchmark_scale.py --steps 50 --action-dim 7
    python scripts/benchmark_scale.py --json benchmark_results.json
    python scripts/benchmark_scale.py --no-pipeline   # pruning only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from calibra.pruning import ApproximateCoresetSelector, CoresetSelector  # noqa: E402
from calibra.pipeline import Pipeline  # noqa: E402
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata  # noqa: E402


def _make_batch(
    n_episodes: int,
    steps_per_ep: int,
    action_dim: int,
    rng: np.random.Generator,
) -> EpisodeBatch:
    """Generate a synthetic EpisodeBatch with random actions (no disk I/O)."""
    dt = 1.0 / 50.0  # 50 Hz
    episodes = []
    for i in range(n_episodes):
        ts = np.arange(steps_per_ep, dtype=np.float64) * dt
        actions = rng.standard_normal((steps_per_ep, action_dim)).astype(np.float32)
        # Smooth actions slightly so jerk metrics are non-degenerate.
        for d in range(action_dim):
            actions[:, d] = np.convolve(actions[:, d], np.ones(5) / 5, mode="same")
        obs = {"proprio": rng.standard_normal((steps_per_ep, action_dim)).astype(np.float32)}
        ep = Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i:07d}"),
            timestamps=ts,
            observations=obs,
            actions=actions,
        )
        episodes.append(ep)
    return EpisodeBatch(
        episodes=episodes,
        dataset_name=f"synthetic_{n_episodes}",
        format="synthetic",
        source_path="<in-memory>",
    )


def _time_pipeline(batch: EpisodeBatch) -> tuple[float, dict[str, float]]:
    t0 = time.perf_counter()
    report = Pipeline().run(batch)
    total = time.perf_counter() - t0
    return total, report.timing


def _time_prune(batch, report, selector_cls, **kwargs):
    selector = selector_cls(keep_fraction=0.3, **kwargs)
    t0 = time.perf_counter()
    selector.select(batch, report)
    return time.perf_counter() - t0


def _fmt(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        default="1000,5000,10000,50000,100000",
        help="Comma-separated episode counts to benchmark "
        "(default: 1000,5000,10000,50000,100000). "
        "Warning: 500k may take several minutes for the exact selector.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Steps per episode (default: 200)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimensionality (default: 7, matching GR00T single-arm)",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Skip Pipeline.run() timing (pruning-only benchmark)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write results to a JSON file in addition to stdout",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    rng = np.random.default_rng(seed=0)

    rows = []
    print("\nCalibra Scale Benchmark")
    print(f"  steps_per_episode={args.steps}  action_dim={args.action_dim}")
    print()

    header = f"  {'Episodes':>10}  {'Pipeline':>10}  {'Prune (exact)':>14}  {'Prune (approx)':>15}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for n in sizes:
        batch = _make_batch(n, args.steps, args.action_dim, rng)

        pipeline_s = None
        per_analyzer: dict[str, float] = {}
        report = None

        if not args.no_pipeline:
            pipeline_s, per_analyzer = _time_pipeline(batch)
            from calibra.pipeline import Pipeline as _P

            report = _P().run(batch)
        else:
            from calibra.pipeline import Pipeline as _P

            report = _P().run(batch)

        # Exact selector (skip for N > 100k — too slow to be useful as benchmark)
        if n <= 100_000:
            exact_s = _time_prune(batch, report, CoresetSelector)
        else:
            exact_s = None

        approx_s = _time_prune(batch, report, ApproximateCoresetSelector, batch_size=1000)

        exact_str = _fmt(exact_s) if exact_s is not None else "  (skipped)"
        pipeline_str = _fmt(pipeline_s) if pipeline_s is not None else "    (skipped)"

        print(f"  {n:>10,}  {pipeline_str:>10}  {exact_str:>14}  {_fmt(approx_s):>15}")

        rows.append(
            {
                "n_episodes": n,
                "steps_per_episode": args.steps,
                "action_dim": args.action_dim,
                "pipeline_s": pipeline_s,
                "per_analyzer_s": per_analyzer,
                "exact_prune_s": exact_s,
                "approx_prune_s": approx_s,
            }
        )

    print()

    if rows:
        # Show at what size approximate beats exact
        crossover = next(
            (
                r["n_episodes"]
                for r in rows
                if r["exact_prune_s"] is not None
                and r["approx_prune_s"] is not None
                and r["approx_prune_s"] < r["exact_prune_s"]
            ),
            None,
        )
        if crossover:
            print(f"  Approximate selector faster than exact starting at {crossover:,} episodes.")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"  Results written to {args.json}")


if __name__ == "__main__":
    main()
