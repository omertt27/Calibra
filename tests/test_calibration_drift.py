"""Tests for the Calibration Drift Analyzer."""

from __future__ import annotations

import zlib

import numpy as np

from calibra.analyzers.calibration_drift import CalibrationDriftAnalyzer
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_ep(
    episode_id: str,
    offset: np.ndarray,
    dims: int = 6,
    steps: int = 200,
    dt: float = 0.05,
    hold_frac: float = 0.4,
    noise_std: float = 0.002,
    seed: int = 0,
) -> Episode:
    """Smooth low-frequency state trajectory with a frozen mid-episode hold
    (a genuine stationary segment), offset by a per-motor constant bias in
    the action to simulate a stale leader/follower calibration."""
    rng = np.random.default_rng(seed)
    ts = np.arange(steps, dtype=np.float64) * dt
    state = np.zeros((steps, dims), dtype=np.float32)
    for d in range(dims):
        state[:, d] = 0.5 * np.sin(2 * np.pi * 0.05 * ts + d)
    hold_len = int(steps * hold_frac)
    mid = steps // 2
    state[mid : mid + hold_len] = state[mid]
    action = state + offset + rng.normal(0, noise_std, state.shape).astype(np.float32)
    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=ts,
        observations={"proprio": state},
        actions=action.astype(np.float32),
    )


def _make_no_hold_ep(episode_id: str, dims: int = 6, steps: int = 200, dt: float = 0.05) -> Episode:
    """State never holds still (fast-moving sinusoid throughout) — no
    stationary run should be found."""
    rng = np.random.default_rng(zlib.crc32(episode_id.encode()) % (2**31))
    ts = np.arange(steps, dtype=np.float64) * dt
    state = np.zeros((steps, dims), dtype=np.float32)
    for d in range(dims):
        state[:, d] = 0.5 * np.sin(2 * np.pi * 0.3 * ts + d)
    action = state + rng.normal(0, 0.002, state.shape).astype(np.float32)
    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=ts,
        observations={"proprio": state},
        actions=action.astype(np.float32),
    )


def _make_batch(episodes: list[Episode]) -> EpisodeBatch:
    return EpisodeBatch(
        episodes=episodes, dataset_name="calib_test", format="hdf5", source_path="/dummy/path.h5"
    )


class TestCalibrationDriftAnalyzer:
    def test_injected_offset_flagged_warning(self):
        offset = np.array([0.3, 0, 0, 0, 0, 0], dtype=np.float32)
        episodes = [_make_ep(f"ep_{i}", offset, seed=i) for i in range(10)]
        batch = _make_batch(episodes)
        result = CalibrationDriftAnalyzer().analyze(batch)
        flag = result.flags[0]
        assert flag.metric == "joint_offset_max_abs"
        assert flag.level == RiskLevel.WARNING
        worst = max(result.raw_metrics["per_motor_offset"], key=lambda m: m["offset_frac_of_range"])
        assert worst["dim"] == 0
        assert worst["systematic"] is True

    def test_never_reports_critical(self):
        """Thresholds are explicitly uncalibrated (see module docstring) —
        capped at WARNING even for a very large injected offset."""
        offset = np.array([5.0, 0, 0, 0, 0, 0], dtype=np.float32)
        episodes = [_make_ep(f"ep_{i}", offset, seed=i) for i in range(10)]
        batch = _make_batch(episodes)
        result = CalibrationDriftAnalyzer().analyze(batch)
        assert result.flags[0].level != RiskLevel.CRITICAL

    def test_zero_offset_is_ok(self):
        episodes = [_make_ep(f"ep_{i}", np.zeros(6, dtype=np.float32), seed=i) for i in range(10)]
        batch = _make_batch(episodes)
        result = CalibrationDriftAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.OK

    def test_no_sustained_hold_returns_info(self):
        """Fast-moving trajectory with no real stationary run must not
        false-positive on transient near-zero-velocity samples (e.g. a
        sinusoid's turning points)."""
        episodes = [_make_no_hold_ep(f"ep_{i}") for i in range(10)]
        batch = _make_batch(episodes)
        result = CalibrationDriftAnalyzer().analyze(batch)
        flag = result.flags[0]
        assert flag.level == RiskLevel.INFO
        assert "skipped" in result.raw_metrics

    def test_no_state_observation_returns_info(self):
        rng = np.random.default_rng(0)
        steps = 50
        ts = np.arange(steps, dtype=np.float64) * 0.05
        episodes = [
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=ts,
                observations={"camera_rgb": rng.integers(0, 255, (steps, 8, 8, 3), dtype=np.uint8)},
                actions=rng.random((steps, 6)).astype(np.float32),
            )
            for i in range(5)
        ]
        batch = _make_batch(episodes)
        result = CalibrationDriftAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.INFO
        assert result.raw_metrics["skipped"] == "no stable-frame action/state pairs"

    def test_empty_batch_returns_no_flags(self):
        batch = _make_batch([])
        result = CalibrationDriftAnalyzer().analyze(batch)
        assert result.flags == []

    def test_requires_proprio_capability(self):
        assert CalibrationDriftAnalyzer.requires == frozenset({"proprio"})

    def test_gripper_dim_excluded_by_default(self):
        """A large offset only on the excluded gripper dim (last column)
        must not be reported."""
        offset = np.array([0, 0, 0, 0, 0, 5.0], dtype=np.float32)
        episodes = [_make_ep(f"ep_{i}", offset, seed=i) for i in range(10)]
        batch = _make_batch(episodes)
        result = CalibrationDriftAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.OK
