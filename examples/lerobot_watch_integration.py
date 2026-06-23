"""
LeRobot teleoperation → calibra watch --stream integration.

Drop this file into your LeRobot data collection script to pipe real-time
quality metrics into `calibra watch --stream --remediate`.

Usage
-----
  # Terminal 1: start Calibra in stream mode
  calibra watch --stream --remediate --log-file session.jsonl

  # Terminal 2: run your collection script with this wrapper
  python lerobot_watch_integration.py

Or pipe directly:
  python your_collect_script.py 2>&1 | python lerobot_watch_integration.py | calibra watch --stream --remediate

How it works
------------
  After each episode is saved, this wrapper computes the same quality metrics
  Calibra uses (spike_rate, vel_disc_rate, dropout_rate, ldlj, jitter_cv)
  and emits a single JSON line to stdout. calibra watch --stream reads these
  lines and prints a verdict + remediation advice in real time.

Integration points
------------------
  The key function is `emit_episode_metrics(episode_id, actions, timestamps)`.
  Call it from your teleoperation loop after each episode is complete.

  If you use LeRobot's `record` command, use `MetricsEmitter` as a context
  manager around the episode-save step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ── metric computation (pure numpy, no Calibra pipeline dependency) ───────────

def _compute_spike_rate(actions: np.ndarray, dt: float = 0.02, sigma_limit: float = 5.0) -> float:
    """Fraction of timesteps where jerk exceeds sigma_limit * std."""
    if len(actions) < 3:
        return 0.0
    vel = np.diff(actions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    jerk_norm = np.linalg.norm(jerk, axis=-1) if jerk.ndim > 1 else np.abs(jerk)
    mu, sigma = jerk_norm.mean(), jerk_norm.std()
    if sigma < 1e-9:
        return 0.0
    return float(np.mean(jerk_norm > mu + sigma_limit * sigma))


def _compute_vel_disc_rate(actions: np.ndarray, dt: float = 0.02) -> float:
    """Fraction of consecutive step pairs with a velocity sign reversal."""
    if len(actions) < 2:
        return 0.0
    vel = np.diff(actions, axis=0) / dt
    if vel.ndim == 1:
        vel = vel[:, None]
    reversals = (vel[:-1] * vel[1:]) < 0
    return float(np.mean(reversals))


def _compute_dropout_rate(timestamps: np.ndarray) -> float:
    """Fraction of frames where the gap is >2x the median gap."""
    if len(timestamps) < 2:
        return 0.0
    gaps = np.diff(timestamps)
    median_gap = np.median(gaps)
    if median_gap <= 0:
        return 0.0
    return float(np.mean(gaps > 2.0 * median_gap))


def _compute_ldlj(positions: np.ndarray, dt: float = 0.02) -> float:
    """Log-Dimensionless Jerk score (lower/more negative = jerkier)."""
    if len(positions) < 4:
        return -5.0
    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    T = len(positions) * dt
    speed_norm = np.linalg.norm(vel, axis=-1) if vel.ndim > 1 else np.abs(vel)
    A = float(np.max(speed_norm))
    if A < 1e-9:
        return -5.0
    jerk_sq = np.sum(jerk ** 2, axis=-1) if jerk.ndim > 1 else jerk ** 2
    integral = float(np.trapz(jerk_sq, dx=dt))
    if integral <= 0:
        return -5.0
    return float(np.log((T ** 3 / A ** 2) * integral))


def _compute_jitter_cv(timestamps: np.ndarray) -> float:
    """Coefficient of variation of inter-frame intervals."""
    if len(timestamps) < 2:
        return 0.0
    gaps = np.diff(timestamps)
    mu = gaps.mean()
    if mu < 1e-9:
        return 0.0
    return float(gaps.std() / mu)


# ── emitter ───────────────────────────────────────────────────────────────────

def emit_episode_metrics(
    episode_id: str,
    actions: np.ndarray,
    timestamps: np.ndarray,
    states: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    file: object = None,
) -> dict:
    """
    Compute and emit quality metrics for one episode as a JSON line.

    Parameters
    ----------
    episode_id  : unique episode identifier (used as the 'file' field)
    actions     : shape (T, D) — action trajectory
    timestamps  : shape (T,) — wall-clock timestamps in seconds
    states      : shape (T, D) — proprioceptive state (optional; falls back to actions)
    dt          : control timestep in seconds (inferred from timestamps if None)
    file        : output stream (default: sys.stdout)

    Returns the metrics dict (also emitted to the stream).
    """
    if file is None:
        file = sys.stdout

    if dt is None and len(timestamps) >= 2:
        dt = float(np.median(np.diff(timestamps)))
    dt = dt or 0.02

    positions = states if states is not None else actions

    metrics = {
        "file":           str(episode_id),
        "spike_rate":     round(_compute_spike_rate(actions, dt), 6),
        "vel_disc_rate":  round(_compute_vel_disc_rate(actions, dt), 6),
        "dropout_rate":   round(_compute_dropout_rate(timestamps), 6),
        "ldlj":           round(_compute_ldlj(positions, dt), 4),
        "jitter_cv":      round(_compute_jitter_cv(timestamps), 6),
    }

    print(json.dumps(metrics), file=file, flush=True)
    return metrics


class MetricsEmitter:
    """
    Context manager for emitting episode metrics after each episode save.

    Usage in a LeRobot-style collection loop:

        emitter = MetricsEmitter()
        for ep_idx in range(n_episodes):
            actions, timestamps = collect_episode(robot)
            save_episode(ep_idx, actions)
            with emitter:
                emitter.emit(f"ep_{ep_idx:04d}.hdf5", actions, timestamps)
    """

    def __init__(self, output_file: object = None) -> None:
        self._file = output_file or sys.stdout

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def emit(
        self,
        episode_id: str,
        actions: np.ndarray,
        timestamps: np.ndarray,
        states: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> dict:
        return emit_episode_metrics(
            episode_id, actions, timestamps, states=states, dt=dt, file=self._file
        )


# ── demo: simulate a collection session ──────────────────────────────────────

def _simulate_collection_session(n_episodes: int = 10) -> None:
    """
    Simulate a teleoperation session with mixed clean and bad episodes.

    Run this to see what calibra watch --stream output looks like:

        python examples/lerobot_watch_integration.py | calibra watch --stream --remediate
    """
    rng = np.random.default_rng(42)
    emitter = MetricsEmitter()

    for i in range(n_episodes):
        T = rng.integers(80, 150)
        dt = 0.02
        timestamps = np.arange(T) * dt

        if i % 4 == 3:
            # Simulate a bad episode (abrupt jerk spikes)
            actions = rng.normal(0, 0.1, (T, 7))
            actions[T // 2 : T // 2 + 3] += rng.normal(0, 5.0, (3, 7))
        else:
            actions = rng.normal(0, 0.1, (T, 7))
            for d in range(7):
                t = np.linspace(0, 2 * np.pi, T)
                actions[:, d] += 0.3 * np.sin(t + d * 0.5)

        emitter.emit(f"ep_{i:04d}.hdf5", actions, timestamps)

        import time
        time.sleep(0.05)  # simulate episode duration


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "Simulate a teleoperation session and emit episode metrics to stdout. "
            "Pipe to `calibra watch --stream --remediate` to see live feedback.\n\n"
            "Example:\n"
            "  python examples/lerobot_watch_integration.py | "
            "calibra watch --stream --remediate"
        )
    )
    p.add_argument("--n-episodes", type=int, default=10,
                   help="Number of synthetic episodes to simulate (default: 10)")
    args = p.parse_args()
    _simulate_collection_session(args.n_episodes)
