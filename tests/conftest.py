"""
Shared fixtures for Calibra tests.

All fixtures produce synthetic EpisodeBatch objects so tests run without
any real dataset files or optional format dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest

from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata


def _make_episode(
    n_steps: int = 100,
    dt: float = 0.1,
    action_dim: int = 7,
    jitter_std: float = 0.0,
    dropout_indices: list[int] | None = None,
    episode_id: str = "ep_0",
    cam_lag_std: float = 0.0,
    action_offset: float = 0.0,
) -> Episode:
    """
    Synthesise a single episode.

    jitter_std       : std-dev of Gaussian noise added to step deltas (seconds).
    dropout_indices  : step indices where the gap is inflated to 5× dt.
    cam_lag_std      : std-dev of camera timestamp offset from master (seconds).
    action_offset    : constant offset of action_timestamps from obs_timestamps.
    """
    rng = np.random.default_rng(42)

    deltas = np.full(n_steps, dt, dtype=np.float64)
    if jitter_std > 0:
        deltas += rng.normal(0, jitter_std, size=n_steps)
        deltas = np.abs(deltas)

    if dropout_indices:
        for idx in dropout_indices:
            if 0 <= idx < n_steps:
                deltas[idx] = 5 * dt

    timestamps = np.concatenate([[0.0], np.cumsum(deltas[:-1])])

    obs_ts: dict[str, np.ndarray] = {}
    if cam_lag_std > 0:
        obs_ts["camera_rgb"] = timestamps + rng.normal(0, cam_lag_std, size=n_steps)

    action_ts = None
    if action_offset != 0.0:
        action_ts = timestamps + action_offset

    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=timestamps,
        observations={
            "camera_rgb": rng.random((n_steps, 64, 64, 3)).astype(np.float32),
            "proprio": rng.random((n_steps, 14)).astype(np.float32),
        },
        actions=rng.random((n_steps, action_dim)).astype(np.float32),
        obs_timestamps=obs_ts,
        action_timestamps=action_ts,
    )


@pytest.fixture
def clean_batch() -> EpisodeBatch:
    """10 episodes with perfectly uniform timestamps."""
    episodes = [_make_episode(episode_id=f"ep_{i}") for i in range(10)]
    return EpisodeBatch(episodes=episodes, dataset_name="clean",
                        format="hdf5", source_path="/tmp/clean.h5")


@pytest.fixture
def jittery_batch() -> EpisodeBatch:
    """Episodes with high timing jitter (CV > WARNING threshold)."""
    episodes = [
        _make_episode(jitter_std=0.025, episode_id=f"ep_{i}") for i in range(10)
    ]
    return EpisodeBatch(episodes=episodes, dataset_name="jittery",
                        format="hdf5", source_path="/tmp/jittery.h5")


@pytest.fixture
def dropout_batch() -> EpisodeBatch:
    """Episodes with 10% dropout rate."""
    episodes = [
        _make_episode(
            dropout_indices=list(range(0, 100, 10)),
            episode_id=f"ep_{i}",
        )
        for i in range(10)
    ]
    return EpisodeBatch(episodes=episodes, dataset_name="dropout",
                        format="hdf5", source_path="/tmp/dropout.h5")


@pytest.fixture
def cam_lag_batch() -> EpisodeBatch:
    """Episodes with camera lag std > CRITICAL threshold (25 ms)."""
    episodes = [
        _make_episode(cam_lag_std=0.025, episode_id=f"ep_{i}") for i in range(10)
    ]
    return EpisodeBatch(episodes=episodes, dataset_name="cam_lag",
                        format="hdf5", source_path="/tmp/cam_lag.h5")


@pytest.fixture
def misaligned_batch() -> EpisodeBatch:
    """Episodes with action timestamps offset by 10 ms (> WARNING threshold)."""
    episodes = [
        _make_episode(action_offset=0.010, episode_id=f"ep_{i}") for i in range(10)
    ]
    return EpisodeBatch(episodes=episodes, dataset_name="misaligned",
                        format="hdf5", source_path="/tmp/misaligned.h5")
