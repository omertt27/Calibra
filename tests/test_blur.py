"""Tests for the Blur Analyzer."""

from __future__ import annotations

import numpy as np

from calibra.analyzers.blur import BlurAnalyzer, compute_laplacian_variance
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_ep(episode_id: str = "ep_0", n_steps: int = 10, image_mode: str = "sharp") -> Episode:
    rng = np.random.default_rng(hash(episode_id) % (2**31))
    ts = np.arange(n_steps, dtype=np.float64) * 0.05
    obs: dict = {"proprio": rng.random((n_steps, 8)).astype(np.float32)}

    if image_mode == "sharp":
        # High-frequency noise: lots of local pixel-to-pixel variation.
        obs["camera_rgb"] = rng.integers(0, 255, (n_steps, 32, 32, 3), dtype=np.uint8)
    elif image_mode == "blurry":
        # Smooth low-frequency gradient: minimal local variation.
        x = np.linspace(0, 1, 32)
        gradient = np.outer(x, x)
        frame = (gradient * 255).astype(np.uint8)
        frame = np.stack([frame] * 3, axis=-1)
        obs["camera_rgb"] = np.tile(frame, (n_steps, 1, 1, 1))
    # "none" → no image key

    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=ts,
        observations=obs,
        actions=rng.random((n_steps, 6)).astype(np.float32),
    )


def _make_batch(episodes: list[Episode]) -> EpisodeBatch:
    return EpisodeBatch(
        episodes=episodes, dataset_name="blur_test", format="hdf5", source_path="/dummy/path.h5"
    )


# ── compute_laplacian_variance ──────────────────────────────────────────────


class TestComputeLaplacianVariance:
    def test_sharp_has_higher_variance_than_blurry(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        blurry = np.full((5, 32, 32, 3), 128, dtype=np.uint8)
        sharp_var = compute_laplacian_variance(sharp).mean()
        blurry_var = compute_laplacian_variance(blurry).mean()
        assert sharp_var > blurry_var

    def test_output_shape(self):
        images = np.zeros((7, 16, 16, 3), dtype=np.uint8)
        result = compute_laplacian_variance(images)
        assert result.shape == (7,)

    def test_grayscale_input(self):
        images = np.zeros((7, 16, 16), dtype=np.uint8)
        result = compute_laplacian_variance(images)
        assert result.shape == (7,)


# ── BlurAnalyzer ─────────────────────────────────────────────────────────────


class TestBlurAnalyzer:
    def test_outlier_blurry_episode_flagged(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="sharp") for i in range(6)]
        episodes.append(_make_ep("ep_blurry", image_mode="blurry"))
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        flag = result.flags[0]
        assert flag.metric == "blurry_episode_fraction"
        assert flag.level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
        assert "ep_blurry" in result.raw_metrics["outlier_episode_ids"]

    def test_uniformly_sharp_dataset_ok(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="sharp") for i in range(6)]
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.OK

    def test_too_few_episodes_returns_info(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="sharp") for i in range(3)]
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.INFO
        assert result.raw_metrics["skipped"] == "too few episodes with image data"

    def test_no_image_data_returns_info(self):
        episodes = [_make_ep(f"ep_{i}", image_mode="none") for i in range(6)]
        batch = _make_batch(episodes)
        result = BlurAnalyzer().analyze(batch)
        assert result.flags[0].level == RiskLevel.INFO
        assert result.raw_metrics["skipped"] == "no image observations"

    def test_empty_batch_returns_no_flags(self):
        batch = _make_batch([])
        result = BlurAnalyzer().analyze(batch)
        assert result.flags == []

    def test_requires_images_capability(self):
        assert BlurAnalyzer.requires == frozenset({"images"})
