"""Unit tests for the SSLTrajectoryEmbedderAnalyzer."""

from __future__ import annotations

import numpy as np

from calibra.analyzers.ssl_embed import SSLTrajectoryEmbedderAnalyzer
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata


def _make_batch(
    n_eps: int = 5, n_steps: int = 50, action_dim: int = 4, inject_outlier: bool = False
) -> EpisodeBatch:
    rng = np.random.default_rng(42)
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        if inject_outlier and i == 0:
            # Inject a huge outlier trajectory (massive actions / state changes)
            acts = rng.uniform(100.0, 200.0, (n_steps, action_dim)).astype(np.float32)
            obs = {"proprio": rng.uniform(100.0, 200.0, (n_steps, 8)).astype(np.float32)}
        else:
            acts = rng.uniform(-1.0, 1.0, (n_steps, action_dim)).astype(np.float32)
            obs = {"proprio": rng.uniform(-1.0, 1.0, (n_steps, 8)).astype(np.float32)}

        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=ts,
                observations=obs,
                actions=acts,
            )
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="ssl_test",
        format="hdf5",
        source_path="/tmp/ssl_test.h5",
    )


class TestSSLTrajectoryEmbedderAnalyzer:
    def test_analyzer_runs_successfully(self):
        batch = _make_batch(n_eps=4)
        analyzer = SSLTrajectoryEmbedderAnalyzer(embedding_dim=16)
        result = analyzer.analyze(batch)

        assert result.analyzer_name == "ssl_embed"
        assert "mean_nearest_distance" in result.raw_metrics
        assert "per_episode_ssl_novelty" in result.raw_metrics
        assert len(result.raw_metrics["per_episode_ssl_novelty"]) == 4

    def test_handles_empty_or_single_episode_gracefully(self):
        empty_batch = EpisodeBatch(
            episodes=[],
            dataset_name="empty",
            format="hdf5",
            source_path="/tmp/empty.h5",
        )
        analyzer = SSLTrajectoryEmbedderAnalyzer()
        res = analyzer.analyze(empty_batch)
        assert res.flags == []
        assert res.raw_metrics == {}

    def test_detects_outlier_trajectory(self):
        # We need enough episodes so that 1 outlier exceeds the 5% threshold
        # Or let's test if the outlier is recorded in the outlier list
        batch = _make_batch(n_eps=10, inject_outlier=True)
        analyzer = SSLTrajectoryEmbedderAnalyzer(embedding_dim=16)
        result = analyzer.analyze(batch)

        # Outliers should be captured in outlier_indices
        outliers = result.raw_metrics.get("outlier_indices", [])
        assert 0 in outliers  # The 0th episode (injected outlier) should be flagged
