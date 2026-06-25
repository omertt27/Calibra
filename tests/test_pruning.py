"""Tests for calibra.pruning — CoresetSelector."""

from __future__ import annotations

import numpy as np
import pytest

from calibra.pipeline import Pipeline
from calibra.pruning import CoresetSelector, _greedy_max_coverage
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_ep(n_steps=80, action_scale=1.0, spike=False, episode_id="ep_0"):
    rng = np.random.default_rng(int(episode_id.split("_")[-1]) if "_" in episode_id else 0)
    ts = np.arange(n_steps) * 0.02
    actions = rng.normal(0, action_scale, (n_steps, 6)).astype(np.float32)

    if spike:
        # inject a hard jerk spike at step 20
        actions[20] += 100.0

    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=ts,
        observations={"state": rng.random((n_steps, 6)).astype(np.float32)},
        actions=actions,
    )


@pytest.fixture
def mixed_batch():
    """Batch with clean, diverse, and spikey episodes."""
    episodes = (
        [_make_ep(action_scale=0.5, episode_id=f"ep_{i}") for i in range(5)]  # low-range
        + [_make_ep(action_scale=2.0, episode_id=f"ep_{i + 5}") for i in range(5)]  # high-range
        + [_make_ep(spike=True, episode_id=f"ep_{i + 10}") for i in range(3)]  # spikey
    )
    return EpisodeBatch(
        episodes=episodes, dataset_name="mixed", format="hdf5", source_path="/tmp/mixed.h5"
    )


# ── greedy max-coverage ───────────────────────────────────────────────────────


class TestGreedyMaxCoverage:
    def test_returns_k_indices(self):
        rng = np.random.default_rng(0)
        features = rng.random((20, 5))
        selected = _greedy_max_coverage(features, k=7)
        assert len(selected) == 7
        assert len(set(selected)) == 7  # unique

    def test_all_selected_when_k_gte_n(self):
        features = np.eye(5)
        selected = _greedy_max_coverage(features, k=10)
        assert sorted(selected) == list(range(5))

    def test_maximises_spread(self):
        # Two clusters: 0–4 near origin, 5–9 far from origin
        rng = np.random.default_rng(42)
        near = rng.normal(0, 0.01, (5, 2))
        far = rng.normal(10, 0.01, (5, 2))
        features = np.vstack([near, far])

        selected = _greedy_max_coverage(features, k=2)
        # Should pick one from each cluster
        groups = {int(i >= 5) for i in selected}
        assert groups == {0, 1}

    def test_single_episode(self):
        selected = _greedy_max_coverage(np.array([[1.0, 2.0]]), k=1)
        assert selected == [0]


# ── CoresetSelector ───────────────────────────────────────────────────────────


class TestCoresetSelector:
    def _run(self, batch, **kwargs):
        report = Pipeline().run(batch)
        selector = CoresetSelector(**kwargs)
        return selector.select(batch, report)

    def test_keep_fraction_respected(self, mixed_batch):
        result = self._run(mixed_batch, keep_fraction=0.5)
        # Should keep ≤ 50% (may be less due to quality failures)
        assert result.keep_fraction_actual <= 0.55  # small tolerance
        assert result.n_kept <= 7  # 50% of 13

    def test_spikey_episodes_removed_in_stage1(self, mixed_batch):
        result = self._run(mixed_batch, keep_fraction=0.8, max_spike_rate=0.05)
        # The 3 spikey episodes should be quality-filtered
        spikey_ids = {f"ep_{i + 10}" for i in range(3)}
        kept_set = set(result.keep_episode_ids)
        # All spikey episodes should be removed (not kept)
        assert not (spikey_ids & kept_set), f"Spikey episodes in kept set: {spikey_ids & kept_set}"

    def test_quality_only_keeps_all_passing(self, mixed_batch):
        result = self._run(mixed_batch, quality_only=True, keep_fraction=0.5)
        # quality_only skips Stage 2 → all quality-passing episodes are kept
        total = result.n_kept + result.n_quality_failures
        assert total == mixed_batch.n_episodes
        assert result.n_diversity_pruned == 0

    def test_episode_ids_are_complete_partition(self, mixed_batch):
        result = self._run(mixed_batch, keep_fraction=0.5)
        all_ids = {ep.metadata.episode_id for ep in mixed_batch.episodes}
        result_ids = (
            set(result.keep_episode_ids)
            | set(result.quality_fail_ids)
            | set(result.diversity_pruned_ids)
        )
        assert all_ids == result_ids

    def test_empty_batch_returns_empty(self):
        empty = EpisodeBatch(
            episodes=[], dataset_name="empty", format="hdf5", source_path="/tmp/empty.h5"
        )
        report = Pipeline().run(empty)
        result = CoresetSelector().select(empty, report)
        assert result.n_kept == 0
        assert result.n_original == 0

    def test_everything_quality_fails(self):
        """When all episodes fail quality, result is empty coreset."""
        # Create spikey episodes that will fail the default spike threshold
        episodes = [_make_ep(spike=True, episode_id=f"ep_{i}") for i in range(5)]
        batch = EpisodeBatch(
            episodes=episodes, dataset_name="all_bad", format="hdf5", source_path="/tmp/bad.h5"
        )
        report = Pipeline().run(batch)
        result = CoresetSelector(max_spike_rate=0.001).select(batch, report)
        assert result.n_kept == 0
        assert result.n_quality_failures == 5

    def test_to_dict_has_required_keys(self, mixed_batch):
        result = self._run(mixed_batch, keep_fraction=0.5)
        d = result.to_dict()
        required = {
            "method",
            "n_original",
            "n_kept",
            "keep_episode_ids",
            "quality_fail_ids",
            "diversity_pruned_ids",
            "quality_scores",
        }
        assert required.issubset(d.keys())

    def test_summary_is_string(self, mixed_batch):
        result = self._run(mixed_batch, keep_fraction=0.5)
        s = result.summary()
        assert isinstance(s, str)
        assert "CALIBRA PRUNING SUMMARY" in s

    def test_diverse_selection_spreads_across_clusters(self):
        """
        Episodes from two well-separated action clusters should both appear in
        the coreset when the greedy max-coverage algorithm is free to choose.

        Uses smooth random-walk trajectories (cumsum) so quality filtering
        doesn't interfere, and lenient quality thresholds to isolate Stage 2.
        """
        rng = np.random.default_rng(7)
        # Cluster A: smooth trajectories with actions centred near 0
        eps_a = []
        for i in range(8):
            ts = np.arange(80) * 0.02
            # smooth random walk near 0
            acts = np.cumsum(rng.normal(0, 0.005, (80, 4)), axis=0).astype(np.float32)
            obs = rng.random((80, 4)).astype(np.float32)
            eps_a.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"a_{i}"),
                    timestamps=ts,
                    observations={"state": obs},
                    actions=acts,
                )
            )
        # Cluster B: smooth trajectories with actions centred near +5
        eps_b = []
        for i in range(8):
            ts = np.arange(80) * 0.02
            acts = (np.cumsum(rng.normal(0, 0.005, (80, 4)), axis=0) + 5.0).astype(np.float32)
            obs = rng.random((80, 4)).astype(np.float32)
            eps_b.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"b_{i}"),
                    timestamps=ts,
                    observations={"state": obs},
                    actions=acts,
                )
            )

        batch = EpisodeBatch(
            episodes=eps_a + eps_b,
            dataset_name="clusters",
            format="hdf5",
            source_path="/tmp/clusters.h5",
        )
        report = Pipeline().run(batch)

        # Lenient quality thresholds so Stage 1 passes everything through;
        # we're testing that Stage 2 (diversity) picks from both clusters.
        result = CoresetSelector(
            keep_fraction=0.25,  # select 4 out of 16
            max_spike_rate=1.0,
            max_vel_disc_rate=1.0,
            max_dropout_fraction=1.0,
            min_ldlj=-1000.0,
        ).select(batch, report)

        kept = set(result.keep_episode_ids)
        a_kept = sum(1 for k in kept if k.startswith("a_"))
        b_kept = sum(1 for k in kept if k.startswith("b_"))
        # Both clusters should be represented in the selected coreset
        assert a_kept > 0, f"No episodes from cluster A kept. Kept: {kept}"
        assert b_kept > 0, f"No episodes from cluster B kept. Kept: {kept}"

    def test_novelty_strategy(self, mixed_batch):
        # Run with the novelty strategy and lenient thresholds so nothing fails quality
        result = CoresetSelector(
            keep_fraction=0.3,
            strategy="novelty",
            max_spike_rate=1.0,
            max_vel_disc_rate=1.0,
            max_dropout_fraction=1.0,
            min_ldlj=-1000.0,
        ).select(mixed_batch, Pipeline().run(mixed_batch))

        # Should keep up to 30% of quality-passing episodes
        assert len(result.keep_episode_ids) > 0
        assert len(result.keep_episode_ids) <= 4
        # Assert novelty score keys exist in diversity_scores output
        for ep_id in result.keep_episode_ids:
            assert ep_id in result.diversity_scores
