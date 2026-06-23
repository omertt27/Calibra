"""
Scale benchmark: ApproximateCoresetSelector at 1k–500k episode scale.

Measures wall-clock time and memory for both the exact (O(N×K)) and
approximate (O(N×B)) selectors as N grows, confirming that Calibra handles
Open X-Embodiment / DROID-scale datasets.

Usage:
    python experiments/scale_benchmark.py
    python experiments/scale_benchmark.py --max-n 100000
    python experiments/scale_benchmark.py --save-fig
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from calibra.pruning import CoresetSelector, ApproximateCoresetSelector
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import DiagnosticReport, AnalyzerResult

FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _make_synthetic_batch(n: int, n_steps: int = 50, action_dim: int = 7) -> EpisodeBatch:
    """Generate a synthetic EpisodeBatch of n episodes for benchmarking."""
    rng = np.random.default_rng(seed=42)
    episodes = []
    for i in range(n):
        # Mix clean + corrupted episodes (20% corrupted)
        if rng.random() < 0.20:
            actions = rng.normal(0, 3.0, (n_steps, action_dim))  # noisy
        else:
            actions = rng.normal(0, 0.5, (n_steps, action_dim))  # clean
        ep = Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i:07d}"),
            timestamps=np.linspace(0.0, n_steps * 0.02, n_steps),
            observations={"proprio": rng.normal(0, 1.0, (n_steps, action_dim))},
            actions=actions,
        )
        episodes.append(ep)
    return EpisodeBatch(
        episodes=episodes,
        dataset_name=f"synthetic_{n}",
        format="synthetic",
        source_path="synthetic",
    )


def _make_mock_report(batch: EpisodeBatch) -> DiagnosticReport:
    """Create a minimal DiagnosticReport with per-episode smoothness metrics."""
    rng = np.random.default_rng(seed=0)
    n = batch.n_episodes

    spike_rates = np.abs(rng.normal(0.04, 0.03, n)).tolist()
    vel_disc = np.abs(rng.normal(0.05, 0.04, n)).tolist()
    ldlj = (-rng.exponential(8.0, n)).tolist()

    ep_metrics: list[dict] = []
    for i in range(n):
        ep_metrics.append({
            "episode_id": f"ep_{i:07d}",
            "spike_rate": spike_rates[i],
            "vel_disc_rate": vel_disc[i],
            "ldlj": ldlj[i],
            "dropout_rate": 0.0,
            "jitter_cv": 1e-5,
        })

    ar = AnalyzerResult(
        analyzer_name="control_smoothness",
        raw_metrics={
            "per_episode_metrics": ep_metrics,
            "jerk_spikes": {"mean_spike_fraction": float(np.mean(spike_rates))},
            "vel_discontinuities": {"mean_disc_fraction": float(np.mean(vel_disc))},
            "ldlj": {"mean_ldlj": float(np.mean(ldlj))},
        },
    )
    return DiagnosticReport(
        dataset_name=batch.dataset_name,
        source_path=batch.source_path,
        format=batch.format,
        n_episodes=n,
        n_samples=n * 50,
        analyzer_results=[ar],
    )


def _benchmark_n(
    n: int,
    keep_fraction: float = 0.30,
    run_exact: bool = True,
    batch_size: int = 1000,
) -> dict:
    gc.collect()

    batch = _make_synthetic_batch(n)
    report = _make_mock_report(batch)

    result = {"n": n, "keep_fraction": keep_fraction, "batch_size": batch_size}

    # ── Approximate selector ──────────────────────────────────────────────────
    approx_sel = ApproximateCoresetSelector(
        keep_fraction=keep_fraction, batch_size=batch_size
    )
    tracemalloc.start()
    t0 = time.perf_counter()
    approx_res = approx_sel.select(batch, report)
    approx_time = time.perf_counter() - t0
    _, approx_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result["approx_time_s"] = round(approx_time, 3)
    result["approx_mem_mb"] = round(approx_mem / 1e6, 1)
    result["approx_kept"] = approx_res.n_kept
    result["approx_quality_removed"] = approx_res.n_quality_failures

    # ── Exact selector (only for small N, O(N×K) blows up) ───────────────────
    if run_exact and n <= 10_000:
        exact_sel = CoresetSelector(keep_fraction=keep_fraction)
        tracemalloc.start()
        t0 = time.perf_counter()
        exact_res = exact_sel.select(batch, report)
        exact_time = time.perf_counter() - t0
        _, exact_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result["exact_time_s"] = round(exact_time, 3)
        result["exact_mem_mb"] = round(exact_mem / 1e6, 1)
        result["exact_kept"] = exact_res.n_kept

        # Overlap: how many episodes the two selectors agree on
        overlap = len(
            set(approx_res.keep_episode_ids) & set(exact_res.keep_episode_ids)
        )
        result["overlap_pct"] = round(100.0 * overlap / max(exact_res.n_kept, 1), 1)

    return result


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'N':>8}  {'Approx(s)':>10}  {'Mem(MB)':>8}  "
        f"{'Exact(s)':>9}  {'Overlap%':>9}  {'Kept':>6}"
    )
    print()
    print("Scale Benchmark: ApproximateCoresetSelector vs Exact")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for r in rows:
        exact_s = f"{r['exact_time_s']:.3f}" if "exact_time_s" in r else "N/A (too large)"
        overlap = f"{r['overlap_pct']:.1f}%" if "overlap_pct" in r else "—"
        print(
            f"{r['n']:>8,}  {r['approx_time_s']:>10.3f}  {r['approx_mem_mb']:>8.1f}  "
            f"{exact_s:>9}  {overlap:>9}  {r['approx_kept']:>6}"
        )
    print("─" * len(header))


def _save_figure(rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure generation")
        return

    ns = [r["n"] for r in rows]
    approx_times = [r["approx_time_s"] for r in rows]
    exact_ns = [r["n"] for r in rows if "exact_time_s" in r]
    exact_times = [r["exact_time_s"] for r in rows if "exact_time_s" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#ffffff")

    # Time scaling
    ax1.loglog(ns, approx_times, "o-", color="#2196F3", linewidth=2, label="Approximate (O(N·B))")
    if exact_ns:
        ax1.loglog(exact_ns, exact_times, "s--", color="#F44336", linewidth=2, label="Exact (O(N·K))")
    ax1.set_xlabel("Dataset size (episodes)", fontsize=11)
    ax1.set_ylabel("Wall-clock time (s)", fontsize=11)
    ax1.set_title("Runtime Scaling", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Memory scaling
    approx_mems = [r["approx_mem_mb"] for r in rows]
    ax2.semilogx(ns, approx_mems, "o-", color="#4CAF50", linewidth=2)
    ax2.set_xlabel("Dataset size (episodes)", fontsize=11)
    ax2.set_ylabel("Peak memory (MB)", fontsize=11)
    ax2.set_title("Memory Usage (Approximate Selector)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Calibra Coreset Selector Scale Benchmark", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "fig_scale_benchmark.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_scale_benchmark.png", bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Calibra scale benchmark")
    p.add_argument("--max-n", type=int, default=500_000, metavar="N",
                   help="Largest dataset size to test (default: 500000)")
    p.add_argument("--keep", type=float, default=0.30,
                   help="Coreset keep fraction (default: 0.30)")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="ApproximateCoresetSelector batch size (default: 1000)")
    p.add_argument("--save-fig", action="store_true",
                   help="Save timing/memory plots to experiments/figures/")
    args = p.parse_args()

    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    sizes = [s for s in sizes if s <= args.max_n]
    if not sizes:
        sizes = [args.max_n]

    rows = []
    for n in sizes:
        print(f"  Benchmarking N={n:,} episodes ...", end=" ", flush=True)
        row = _benchmark_n(
            n,
            keep_fraction=args.keep,
            run_exact=True,
            batch_size=args.batch_size,
        )
        rows.append(row)
        print(f"approx={row['approx_time_s']:.2f}s  mem={row['approx_mem_mb']:.0f}MB")

    _print_table(rows)

    if args.save_fig:
        _save_figure(rows)

    # Summary assertion
    max_row = rows[-1]
    n = max_row["n"]
    t = max_row["approx_time_s"]
    print(f"\nAt N={n:,}: approximate selector finished in {t:.1f}s — "
          f"{'✅ under 60s' if t < 60 else '⚠️ over 60s (consider larger batch_size)'}")
    print(
        "\nNote: memory figures reflect fully in-memory synthetic numpy arrays. "
        "Real usage with lazy Parquet/HDF5 loading uses significantly less RAM."
    )


if __name__ == "__main__":
    main()