"""
calibra corrupt — apply controlled corruptions to a dataset and measure metric response.

This command answers: "Does this metric actually detect this failure mode?"

It loads a dataset, applies one or more synthetic corruptions, runs the full
Calibra pipeline on both original and corrupted versions, then prints a
side-by-side metric comparison showing exactly which metrics react and by how much.

Supported corruption modes
--------------------------
--drop-frames RATE     Randomly remove RATE fraction of steps per episode.
                       Simulates camera dropout, network packet loss, or
                       recording gaps. Expected signal: dropout_rate ↑

--add-jitter-ms STD    Add Gaussian noise (std=STD ms) to all timestamps.
                       Simulates clock drift, USB polling jitter, or async
                       sensor logging. Expected signal: jitter_cv ↑

--inject-spikes RATE   Insert abrupt velocity discontinuities into RATE
                       fraction of steps by swapping random action pairs.
                       Simulates dropped control commands or hardware resets.
                       Expected signal: spike_rate ↑, vel_disc_rate ↑

--delay-episode FRAC   Shift timestamps of a FRAC fraction of episodes
                       forward by a fixed offset. Simulates sync loss events.
                       Expected signal: jitter_cv ↑, dropout_rate ↑

--truncate-episodes FRAC  Remove the last 20% of steps from FRAC fraction of
                       episodes. Simulates early recording termination.
                       Expected signal: short_episode_fraction ↑

Corruptions are composable — pass multiple flags to apply several at once.

Usage
-----
    # Single corruption
    calibra corrupt /data/robot_demos.h5 --drop-frames 0.10

    # Combined corruptions
    calibra corrupt lerobot/pusht --add-jitter-ms 50 --inject-spikes 0.05

    # Force format
    calibra corrupt /data/my_dataset --format lerobot --drop-frames 0.15

Exit codes
----------
    0  All specified metrics showed a statistically detectable response.
    1  One or more expected metric responses were not detected (useful in CI).
    2  Dataset could not be loaded or corruptions could not be applied.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.schema.episode import Episode, EpisodeBatch


# ── corruption transforms ─────────────────────────────────────────────────────


@dataclass
class CorruptionConfig:
    """Parameters controlling what corruptions to apply."""

    drop_frames: Optional[float] = None  # fraction of steps to drop [0, 1)
    add_jitter_ms: Optional[float] = None  # timestamp noise std (ms)
    inject_spikes: Optional[float] = None  # fraction of steps to spike [0, 1)
    delay_episode: Optional[float] = None  # fraction of episodes to delay [0, 1)
    truncate_episodes: Optional[float] = None  # fraction of episodes to truncate [0, 1)
    seed: int = 42

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in [
                self.drop_frames,
                self.add_jitter_ms,
                self.inject_spikes,
                self.delay_episode,
                self.truncate_episodes,
            ]
        )

    def describe(self) -> list[str]:
        parts = []
        if self.drop_frames is not None:
            parts.append(f"drop_frames={self.drop_frames:.1%}")
        if self.add_jitter_ms is not None:
            parts.append(f"add_jitter_ms={self.add_jitter_ms:.1f}")
        if self.inject_spikes is not None:
            parts.append(f"inject_spikes={self.inject_spikes:.1%}")
        if self.delay_episode is not None:
            parts.append(f"delay_episode={self.delay_episode:.1%}")
        if self.truncate_episodes is not None:
            parts.append(f"truncate_episodes={self.truncate_episodes:.1%}")
        return parts


def apply_corruptions(batch: EpisodeBatch, cfg: CorruptionConfig) -> EpisodeBatch:
    """
    Return a new EpisodeBatch with corruptions applied.
    The original batch is not modified.
    """
    rng = np.random.default_rng(cfg.seed)
    episodes = [_copy_episode(ep) for ep in batch.episodes]

    if cfg.drop_frames is not None:
        episodes = [_drop_frames(ep, cfg.drop_frames, rng) for ep in episodes]

    if cfg.add_jitter_ms is not None:
        episodes = [_add_jitter(ep, cfg.add_jitter_ms, rng) for ep in episodes]

    if cfg.inject_spikes is not None:
        episodes = [_inject_spikes(ep, cfg.inject_spikes, rng) for ep in episodes]

    if cfg.delay_episode is not None:
        n_delay = max(1, int(len(episodes) * cfg.delay_episode))
        idxs = rng.choice(len(episodes), size=n_delay, replace=False)
        for i in idxs:
            episodes[i] = _delay_episode(episodes[i], rng)

    if cfg.truncate_episodes is not None:
        n_trunc = max(1, int(len(episodes) * cfg.truncate_episodes))
        idxs = rng.choice(len(episodes), size=n_trunc, replace=False)
        for i in idxs:
            episodes[i] = _truncate_episode(episodes[i])

    return EpisodeBatch(
        episodes=episodes,
        dataset_name=batch.dataset_name + " [corrupted]",
        format=batch.format,
        source_path=batch.source_path,
    )


def _copy_episode(ep: Episode) -> Episode:
    return Episode(
        metadata=copy.copy(ep.metadata),
        timestamps=ep.timestamps.copy(),
        observations={k: v.copy() for k, v in ep.observations.items()},
        actions=ep.actions.copy(),
        obs_timestamps={k: v.copy() for k, v in ep.obs_timestamps.items()},
        action_timestamps=(
            ep.action_timestamps.copy() if ep.action_timestamps is not None else None
        ),
    )


def _drop_frames(ep: Episode, rate: float, rng: np.random.Generator) -> Episode:
    """Remove `rate` fraction of steps uniformly at random."""
    n = ep.n_steps
    n_keep = max(2, int(n * (1 - rate)))
    keep = np.sort(rng.choice(n, size=n_keep, replace=False))
    ep.timestamps = ep.timestamps[keep]
    ep.actions = ep.actions[keep]
    ep.observations = {k: v[keep] for k, v in ep.observations.items()}
    if ep.action_timestamps is not None:
        ep.action_timestamps = ep.action_timestamps[keep]
    ep.obs_timestamps = {k: v[keep] for k, v in ep.obs_timestamps.items()}
    return ep


def _add_jitter(ep: Episode, std_ms: float, rng: np.random.Generator) -> Episode:
    """Add Gaussian noise (std=std_ms milliseconds) to all timestamps."""
    std_s = std_ms / 1000.0
    noise = rng.normal(0, std_s, size=ep.timestamps.shape)
    ep.timestamps = ep.timestamps + noise
    ep.timestamps = np.sort(ep.timestamps)  # keep monotonically increasing
    return ep


def _inject_spikes(ep: Episode, rate: float, rng: np.random.Generator) -> Episode:
    """
    Inject jerk spikes by swapping a random action step with a distant one.
    This creates sudden large velocity changes at `rate` fraction of steps.
    """
    n = ep.n_steps
    n_spikes = max(1, int(n * rate))
    if n < 4:
        return ep
    spike_idxs = rng.choice(n - 2, size=min(n_spikes, n // 4), replace=False) + 1
    for i in spike_idxs:
        # Swap with a random distant step (>10% away)
        gap = max(2, n // 10)
        j_candidates = [j for j in range(n) if abs(j - i) > gap]
        if not j_candidates:
            continue
        j = int(rng.choice(j_candidates))
        ep.actions[i], ep.actions[j] = ep.actions[j].copy(), ep.actions[i].copy()
    return ep


def _delay_episode(ep: Episode, rng: np.random.Generator) -> Episode:
    """
    Shift all timestamps forward by a random delay (50–200ms).
    This creates a synthetic sync-loss event when episodes are adjacent.
    """
    delay_s = float(rng.uniform(0.05, 0.20))
    ep.timestamps = ep.timestamps + delay_s
    return ep


def _truncate_episode(ep: Episode) -> Episode:
    """Remove the last 20% of steps, simulating early recording termination."""
    keep = max(2, int(ep.n_steps * 0.80))
    ep.timestamps = ep.timestamps[:keep]
    ep.actions = ep.actions[:keep]
    ep.observations = {k: v[:keep] for k, v in ep.observations.items()}
    if ep.action_timestamps is not None:
        ep.action_timestamps = ep.action_timestamps[:keep]
    ep.obs_timestamps = {k: v[:keep] for k, v in ep.obs_timestamps.items()}
    return ep


# ── metric extraction ─────────────────────────────────────────────────────────

_METRIC_LABELS: dict[str, tuple[str, str]] = {
    "jitter_cv": ("Timestamp jitter CV", "temporal_stability"),
    "dropout_rate": ("Timestamp dropout rate", "temporal_stability"),
    "spike_rate": ("Jerk spike rate", "control_smoothness"),
    "vel_disc_rate": ("Velocity discontinuity", "control_smoothness"),
    "ldlj": ("LDLJ smoothness", "control_smoothness"),
    "action_entropy": ("Action entropy (bits/dim)", "coverage_entropy"),
}

# Which direction is "worse" for each metric.
_WORSE_IS_HIGHER = {"jitter_cv", "dropout_rate", "spike_rate", "vel_disc_rate"}
_WORSE_IS_LOWER = {"ldlj", "action_entropy"}


def _extract_metrics(report) -> dict[str, Optional[float]]:
    from calibra.compare import metrics_from_report

    return metrics_from_report(report)


# ── rendering ─────────────────────────────────────────────────────────────────

_WIDTH = 60


def _fmt(v: Optional[float], key: str) -> str:
    if v is None:
        return "n/a"
    if key in ("jitter_cv",):
        return f"{v:.2e}"
    if key in ("dropout_rate", "spike_rate", "vel_disc_rate"):
        return f"{v:.1%}"
    if key == "ldlj":
        return f"{v:.2f}"
    return f"{v:.3f}"


def _react_symbol(delta: Optional[float], key: str) -> str:
    if delta is None:
        return "   "
    magnitude = abs(delta)
    threshold = 0.001 if key == "jitter_cv" else (0.005 if "rate" in key else 0.1)
    if magnitude < threshold:
        return "  —"  # no meaningful response
    worse = (key in _WORSE_IS_HIGHER and delta > 0) or (key in _WORSE_IS_LOWER and delta < 0)
    if worse:
        return " 🔴" if magnitude > threshold * 5 else " 🟡"
    return " 🟢"


def render_corruption_report(
    dataset_name: str,
    cfg: CorruptionConfig,
    orig_metrics: dict[str, Optional[float]],
    corrupt_metrics: dict[str, Optional[float]],
    orig_n_episodes: int,
    corrupt_n_episodes: int,
) -> str:
    thick = "━" * _WIDTH
    divider = "─" * _WIDTH
    corruption_str = "  ".join(cfg.describe()) or "(none)"

    lines = [
        thick,
        f"calibra corrupt — {dataset_name}",
        f"Corruptions: {corruption_str}",
        thick,
        "",
        f"  {'Metric':<30}  {'Original':>10}  {'Corrupted':>10}  {'Δ':>10}  React",
        divider,
    ]

    for key, (label, _) in _METRIC_LABELS.items():
        orig = orig_metrics.get(key)
        corr = corrupt_metrics.get(key)
        delta = (corr - orig) if (orig is not None and corr is not None) else None
        delta_str = (
            f"{delta:+.1%}"
            if delta is not None and "rate" in key
            else (
                f"{delta:+.2e}"
                if delta is not None and key == "jitter_cv"
                else (f"{delta:+.2f}" if delta is not None else "n/a")
            )
        )
        react = _react_symbol(delta, key)
        lines.append(
            f"  {label:<30}  {_fmt(orig, key):>10}  {_fmt(corr, key):>10}  {delta_str:>10}  {react}"
        )

    lines += [
        divider,
        "",
        "Legend:  🔴 strong response  🟡 weak response  🟢 improved  — no response",
        "",
    ]

    # Interpretation
    responded = []
    missed = []
    for key, (label, _) in _METRIC_LABELS.items():
        orig = orig_metrics.get(key)
        corr = corrupt_metrics.get(key)
        if orig is None or corr is None:
            continue
        delta = corr - orig
        threshold = 0.001 if key == "jitter_cv" else (0.005 if "rate" in key else 0.1)
        if abs(delta) >= threshold:
            responded.append(label)
        elif (
            (cfg.drop_frames and key == "dropout_rate")
            or (cfg.add_jitter_ms and key == "jitter_cv")
            or (cfg.inject_spikes and key in ("spike_rate", "vel_disc_rate"))
        ):
            missed.append(label)

    if responded:
        lines.append("Detected by Calibra:")
        for m in responded:
            lines.append(f"  ✓ {m}")
        lines.append("")
    if missed:
        lines.append("Expected to respond but didn't (review thresholds or corruption rate):")
        for m in missed:
            lines.append(f"  ✗ {m}")
        lines.append("")

    lines.append(thick)
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────


def run_corrupt(argv: list[str]) -> None:
    import argparse
    from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
    from calibra.analyzers.coverage import CoverageEntropyAnalyzer
    from calibra.analyzers.task_structure import TaskStructureAnalyzer
    from calibra.analyzers.temporal import TemporalAnalyzer
    from calibra.ingestion.registry import load
    from calibra.pipeline import Pipeline

    p = argparse.ArgumentParser(
        prog="calibra corrupt",
        description=(
            "Apply synthetic corruptions to a dataset and show which Calibra metrics respond.\n"
            "Use this to validate that metrics detect specific failure modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  calibra corrupt lerobot/pusht --drop-frames 0.10\n"
            "  calibra corrupt /data/robot.h5 --add-jitter-ms 50 --inject-spikes 0.05\n"
            "  calibra corrupt lerobot/pusht --inject-spikes 0.08 --drop-frames 0.05"
        ),
    )
    p.add_argument("path", help="Path or Hub ID of dataset to corrupt")
    p.add_argument(
        "--drop-frames",
        type=float,
        metavar="RATE",
        help="Fraction of steps to randomly drop (e.g. 0.10 = 10%%)",
    )
    p.add_argument(
        "--add-jitter-ms",
        type=float,
        metavar="STD",
        help="Std-dev of Gaussian timestamp noise in milliseconds",
    )
    p.add_argument(
        "--inject-spikes",
        type=float,
        metavar="RATE",
        help="Fraction of steps to inject as jerk spikes",
    )
    p.add_argument(
        "--delay-episode",
        type=float,
        metavar="FRAC",
        help="Fraction of episodes to apply a synthetic delay offset to",
    )
    p.add_argument(
        "--truncate-episodes",
        type=float,
        metavar="FRAC",
        help="Fraction of episodes to truncate at 80%% length",
    )
    p.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "lerobot", "rlds", "mcap"],
        help="Force a format adapter",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    p.add_argument(
        "--gripper-dims",
        metavar="DIMS",
        default=None,
        help="Comma-separated gripper dims to exclude from smoothness",
    )
    args = p.parse_args(argv)

    cfg = CorruptionConfig(
        drop_frames=args.drop_frames,
        add_jitter_ms=args.add_jitter_ms,
        inject_spikes=args.inject_spikes,
        delay_episode=args.delay_episode,
        truncate_episodes=args.truncate_episodes,
        seed=args.seed,
    )
    if cfg.is_empty():
        p.error("Specify at least one corruption flag (e.g. --drop-frames 0.10)")

    gripper_dims: list[int] = [-1]
    if args.gripper_dims is not None:
        raw = args.gripper_dims.strip()
        gripper_dims = [int(x) for x in raw.split(",") if x.strip()] if raw else []

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # Strip hf:// prefix
    path = args.path[len("hf://") :] if args.path.startswith("hf://") else args.path

    log(f"Loading {path!r} ...")
    try:
        orig_batch = load(path, reader=reader)
    except Exception as e:
        print(f"error loading dataset: {e}", file=sys.stderr)
        sys.exit(2)
    log(f"  {orig_batch.n_episodes} episodes  ·  {orig_batch.n_samples} steps")

    log(f"Applying corruptions: {', '.join(cfg.describe())} ...")
    corrupt_batch = apply_corruptions(orig_batch, cfg)

    analyzers = [
        TemporalAnalyzer(),
        ControlSmoothnessAnalyzer(gripper_dims=gripper_dims),
        CoverageEntropyAnalyzer(),
        TaskStructureAnalyzer(),
    ]
    pipeline = Pipeline(analyzers=analyzers)

    log("Running pipeline on original ...")
    orig_report = pipeline.run(orig_batch)
    log("Running pipeline on corrupted ...")
    corrupt_report = pipeline.run(corrupt_batch)

    orig_metrics = _extract_metrics(orig_report)
    corrupt_metrics = _extract_metrics(corrupt_report)

    output = render_corruption_report(
        dataset_name=orig_batch.dataset_name,
        cfg=cfg,
        orig_metrics=orig_metrics,
        corrupt_metrics=corrupt_metrics,
        orig_n_episodes=orig_report.n_episodes,
        corrupt_n_episodes=corrupt_report.n_episodes,
    )
    print(output)

    # Exit 1 if an expected metric didn't respond (useful for CI).
    missed_critical = False
    for key in ("dropout_rate",) if cfg.drop_frames else ():
        orig = orig_metrics.get(key)
        corr = corrupt_metrics.get(key)
        if orig is not None and corr is not None and (corr - orig) < 0.005:
            missed_critical = True
    sys.exit(1 if missed_critical else 0)
