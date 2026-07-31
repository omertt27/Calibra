"""Tests for the Camera Freeze Analyzer."""

from __future__ import annotations

import numpy as np

from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer, _max_run_length
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_ep(
    episode_id: str = "ep_0", n_steps: int = 30, freeze_run: int = 0, isolated_dup: bool = False
) -> Episode:
    """
    freeze_run   : if > 0, the first `freeze_run` frames are held identical
                   (a sustained freeze), remaining frames are random.
    isolated_dup : if True, exactly one frame is repeated (run length 1),
                   with no other duplicates — must NOT count as a freeze.
    """
    rng = np.random.default_rng(0)
    ts = np.arange(n_steps, dtype=np.float64) * 0.05
    obs: dict = {"proprio": rng.random((n_steps, 8)).astype(np.float32)}

    frames = rng.integers(0, 255, (n_steps, 8, 8, 3), dtype=np.uint8)
    if freeze_run > 0:
        frames[:freeze_run] = frames[0]
    if isolated_dup and n_steps > 5:
        frames[5] = frames[4]
    obs["camera_rgb"] = frames

    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=ts,
        observations=obs,
        actions=rng.random((n_steps, 6)).astype(np.float32),
    )


def _make_batch(episodes: list[Episode]) -> EpisodeBatch:
    return EpisodeBatch(
        episodes=episodes, dataset_name="freeze_test", format="hdf5", source_path="/dummy/path.h5"
    )


# ── _max_run_length helper ──────────────────────────────────────────────────


class TestMaxRunLength:
    def test_empty(self):
        assert _max_run_length(np.array([], dtype=bool)) == 0

    def test_no_run(self):
        assert _max_run_length(np.array([False, True, False, True])) == 1

    def test_single_run(self):
        assert _max_run_length(np.array([True, True, True, False, True])) == 3


# ── CameraFreezeAnalyzer ─────────────────────────────────────────────────────


class TestCameraFreezeAnalyzer:
    def test_sustained_freeze_flagged(self):
        batch = _make_batch([_make_ep(freeze_run=10)])
        result = CameraFreezeAnalyzer(min_freeze_run=5).analyze(batch)
        flag = result.flags[0]
        assert flag.metric == "camera_freeze_events"
        assert flag.level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
        assert result.raw_metrics["freeze_episodes"][0]["episode_id"] == "ep_0"
        assert result.raw_metrics["freeze_episodes"][0]["run_length"] >= 9

    def test_isolated_duplicate_not_a_freeze(self):
        batch = _make_batch([_make_ep(isolated_dup=True)])
        result = CameraFreezeAnalyzer(min_freeze_run=5).analyze(batch)
        flag = result.flags[0]
        assert flag.level == RiskLevel.OK
        assert result.raw_metrics["freeze_episodes"] == []

    def test_no_freeze_ok(self):
        batch = _make_batch([_make_ep()])
        result = CameraFreezeAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.OK

    def test_no_image_data_returns_info(self):
        ep = _make_ep()
        ep.observations = {"proprio": ep.observations["proprio"]}
        batch = _make_batch([ep])
        result = CameraFreezeAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.INFO

    def test_empty_batch_returns_no_flags(self):
        batch = _make_batch([])
        result = CameraFreezeAnalyzer().analyze(batch)
        assert result.flags == []
