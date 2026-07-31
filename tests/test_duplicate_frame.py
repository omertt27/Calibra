"""Tests for the Duplicate Frame Analyzer."""

from __future__ import annotations

import numpy as np

from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_ep(episode_id: str = "ep_0", n_steps: int = 30, image_mode: str = "random") -> Episode:
    rng = np.random.default_rng(0)
    ts = np.arange(n_steps, dtype=np.float64) * 0.05
    obs: dict = {"proprio": rng.random((n_steps, 8)).astype(np.float32)}

    if image_mode == "random":
        obs["camera_rgb"] = rng.integers(0, 255, (n_steps, 8, 8, 3), dtype=np.uint8)
    elif image_mode == "identical":
        frame = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        obs["camera_rgb"] = np.tile(frame, (n_steps, 1, 1, 1))
    # "none" → no image key at all

    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=ts,
        observations=obs,
        actions=rng.random((n_steps, 6)).astype(np.float32),
    )


def _make_batch(episodes: list[Episode]) -> EpisodeBatch:
    return EpisodeBatch(
        episodes=episodes, dataset_name="dup_test", format="hdf5", source_path="/dummy/path.h5"
    )


# ── tests ────────────────────────────────────────────────────────────────────


class TestDuplicateFrameAnalyzer:
    def test_all_identical_frames_flagged(self):
        batch = _make_batch([_make_ep(image_mode="identical")])
        result = DuplicateFrameAnalyzer().analyze(batch)
        flag = result.flags[0]
        assert flag.metric == "duplicate_frame_rate"
        assert flag.level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
        assert flag.observed.value == 1.0

    def test_all_different_frames_ok(self):
        batch = _make_batch([_make_ep(image_mode="random")])
        result = DuplicateFrameAnalyzer().analyze(batch)
        flag = result.flags[0]
        assert flag.level == RiskLevel.OK
        assert flag.observed.value < 0.05

    def test_no_image_data_returns_info(self):
        batch = _make_batch([_make_ep(image_mode="none")])
        result = DuplicateFrameAnalyzer().analyze(batch)
        flag = result.flags[0]
        assert flag.level == RiskLevel.INFO
        assert result.raw_metrics["skipped"] == "no image observations"

    def test_empty_batch_returns_no_flags(self):
        batch = _make_batch([])
        result = DuplicateFrameAnalyzer().analyze(batch)
        assert result.flags == []

    def test_requires_images_capability(self):
        assert DuplicateFrameAnalyzer.requires == frozenset({"images"})
