"""Tests for calibra.pruning — CoresetSelector."""

from __future__ import annotations

import numpy as np
import pytest

from calibra.pipeline import Pipeline
from calibra.pruning import CoresetSelector, _build_feature_matrix, _greedy_max_coverage
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


# ── diversity_weight sensitivity ───────────────────────────────────────────────
#
# Regression coverage for a bug where _build_feature_matrix multiplied each
# feature block by its weight and then min-max-normalised each column
# independently — which is invariant to multiplying a whole column by any
# positive constant, so every positive diversity_weight produced the exact
# same normalised matrix (and therefore the same coreset) regardless of its
# value. The fix normalises each block *before* weighting.


def _conflicting_ep_data():
    """
    4 candidates where action-space diversity and quality-metric diversity
    disagree about which pair is "more different":
      ep_0, ep_1: identical action stats, very different spike_rate.
      ep_2, ep_3: identical spike_rate, very different action stats.
    """
    episodes = [
        _make_ep(action_scale=1.0, episode_id="ep_0"),
        _make_ep(action_scale=1.0, episode_id="ep_1"),
        _make_ep(action_scale=1.0, episode_id="ep_2"),
        _make_ep(action_scale=1.0, episode_id="ep_3"),
    ]
    # Force identical action arrays for ep_0/ep_1, and far-apart arrays for ep_2/ep_3.
    episodes[1].actions = episodes[0].actions.copy()
    episodes[2].actions = np.zeros_like(episodes[2].actions)
    episodes[3].actions = np.full_like(episodes[3].actions, 100.0)

    ep_data = {
        "per_episode_spike_rate": [0.0, 0.9, 0.1, 0.1],
        "per_episode_vel_disc_rate": [0.0, 0.0, 0.0, 0.0],
        "per_episode_length": [80, 80, 80, 80],
    }
    return episodes, ep_data


class TestDiversityWeightSensitivity:
    def test_feature_matrix_changes_with_weight(self):
        episodes, ep_data = _conflicting_ep_data()
        candidates = [0, 1, 2, 3]

        mat_low = _build_feature_matrix(episodes, candidates, ep_data, diversity_weight=0.1)
        mat_high = _build_feature_matrix(episodes, candidates, ep_data, diversity_weight=0.9)

        assert not np.allclose(mat_low, mat_high), (
            "diversity_weight=0.1 and diversity_weight=0.9 produced the same "
            "normalised feature matrix — the weight has no effect."
        )

    def test_selection_changes_with_weight(self):
        episodes, ep_data = _conflicting_ep_data()
        candidates = [0, 1, 2, 3]

        mat_low = _build_feature_matrix(episodes, candidates, ep_data, diversity_weight=0.1)
        mat_high = _build_feature_matrix(episodes, candidates, ep_data, diversity_weight=0.9)

        selected_low = set(_greedy_max_coverage(mat_low, k=2))
        selected_high = set(_greedy_max_coverage(mat_high, k=2))

        # Low diversity_weight (quality-dominated) should favor the pair that
        # differs in spike_rate (ep_0, ep_1); high diversity_weight (action-
        # dominated) should favor the pair that differs in action stats (ep_2, ep_3)
        # over the pair that's identical in action space (ep_0, ep_1).
        assert selected_low == {0, 1}, f"expected quality-diverse pair, got {selected_low}"
        assert selected_high != selected_low, (
            f"diversity_weight=0.1 and 0.9 produced the same selection {selected_low} — "
            "the weight has no effect on which episodes are kept."
        )
        assert {0, 1} != selected_high and (2 in selected_high or 3 in selected_high), (
            f"expected high diversity_weight to favor the action-diverse pair, got {selected_high}"
        )


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

    def test_world_model_strategy(self):
        torch = pytest.importorskip("torch")  # noqa: F841

        rng = np.random.default_rng(0)
        episodes = []
        for i in range(5):
            ts = np.arange(40) * 0.02
            acts = rng.normal(0, float(i + 1), (40, 4)).astype(np.float32)
            obs = rng.random((40, 6)).astype(np.float32)
            episodes.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"wm_{i}"),
                    timestamps=ts,
                    observations={"state": obs},
                    actions=acts,
                )
            )
        batch = EpisodeBatch(
            episodes=episodes,
            dataset_name="wm_test",
            format="hdf5",
            source_path="/tmp/wm_test.h5",
        )
        report = Pipeline().run(batch)
        result = CoresetSelector(
            keep_fraction=0.5,
            strategy="world-model",
            max_spike_rate=1.0,
            max_vel_disc_rate=1.0,
            max_dropout_fraction=1.0,
            min_ldlj=-1000.0,
        ).select(batch, report)

        assert len(result.keep_episode_ids) > 0


# ── ADR-011 disposition view ─────────────────────────────────────────────────


class TestPruningResultToCurationReport:
    def _batch(self, ids):
        episodes = [_make_ep(episode_id=eid) for eid in ids]
        return EpisodeBatch(
            episodes=episodes, dataset_name="d", format="hdf5", source_path="/tmp/d.h5"
        )

    def _result(self, **kw):
        from calibra.pruning import PruningResult

        defaults = dict(
            keep_episode_ids=[],
            quality_fail_ids=[],
            diversity_pruned_ids=[],
            quality_scores={},
            diversity_scores={},
            n_original=0,
            n_kept=0,
            n_quality_failures=0,
            n_diversity_pruned=0,
            keep_fraction_actual=0.0,
        )
        defaults.update(kw)
        return PruningResult(**defaults)

    def test_maps_three_buckets_to_dispositions(self):
        from calibra.pruning import pruning_result_to_curation_report
        from calibra.schema.comparison import Disposition

        batch = self._batch(["ep_0", "ep_1", "ep_2", "ep_3"])
        result = self._result(
            keep_episode_ids=["ep_0", "ep_1"],
            quality_fail_ids=["ep_2"],
            diversity_pruned_ids=["ep_3"],
            quality_scores={"ep_0": 0.02, "ep_1": 0.05, "ep_2": 0.9, "ep_3": 0.1},
            fail_reasons={"ep_2": ["jerk_spike"], "ep_3": ["diversity_pruned"]},
            n_original=4,
            n_kept=2,
            n_quality_failures=1,
            n_diversity_pruned=1,
            keep_fraction_actual=0.5,
        )

        report = pruning_result_to_curation_report(result, batch)

        by_id = {d.episode_id: d for d in report.dispositions}
        assert by_id["ep_0"].disposition is Disposition.KEEP
        assert by_id["ep_1"].disposition is Disposition.KEEP
        assert by_id["ep_2"].disposition is Disposition.DROP
        assert by_id["ep_3"].disposition is Disposition.DROP  # redundant → DROP by default

        assert by_id["ep_2"].integrity_flags == ["jerk_spike"]
        assert by_id["ep_2"].reasons == ["jerk_spike"]
        assert by_id["ep_3"].integrity_flags == []  # not an integrity failure
        assert by_id["ep_3"].reasons == ["diversity_pruned"]
        assert by_id["ep_0"].quality_risk == pytest.approx(0.02)
        assert by_id["ep_0"].n_steps == 80

        assert report.original_n_episodes == 4
        assert report.retained_n_episodes == 2
        assert sorted(report.retained_indices) == [0, 1]
        assert sorted(report.dropped_indices) == [2, 3]
        assert report.disposition_counts() == {"KEEP": 2, "DROP": 2}

    def test_redundant_disposition_override_counts_as_retained(self):
        from calibra.pruning import pruning_result_to_curation_report
        from calibra.schema.comparison import Disposition

        batch = self._batch(["ep_0", "ep_1", "ep_2"])
        result = self._result(
            keep_episode_ids=["ep_0"],
            diversity_pruned_ids=["ep_1", "ep_2"],
            quality_scores={"ep_0": 0.0, "ep_1": 0.0, "ep_2": 0.0},
            fail_reasons={"ep_1": ["diversity_pruned"], "ep_2": ["diversity_pruned"]},
            n_original=3,
            n_kept=1,
            n_diversity_pruned=2,
            keep_fraction_actual=1 / 3,
        )

        report = pruning_result_to_curation_report(
            result, batch, redundant_disposition=Disposition.DOWNWEIGHT
        )

        assert report.disposition_counts() == {"KEEP": 1, "DOWNWEIGHT": 2}
        # DOWNWEIGHT is KEEP-like → all three episodes are "in the training set"
        assert report.retained_n_episodes == 3
        assert sorted(report.retained_indices) == [0, 1, 2]
        assert report.dropped_indices == []

    def test_end_to_end_through_selector(self, mixed_batch):
        from calibra.pruning import pruning_result_to_curation_report
        from calibra.schema.comparison import Disposition

        report_diag = Pipeline().run(mixed_batch)
        result = CoresetSelector(keep_fraction=0.5, max_spike_rate=0.05).select(
            mixed_batch, report_diag
        )

        curation = pruning_result_to_curation_report(result, mixed_batch)

        assert len(curation.dispositions) == mixed_batch.n_episodes
        assert [d.episode_index for d in curation.dispositions] == list(
            range(mixed_batch.n_episodes)
        )
        keep_ids = {d.episode_id for d in curation.by_disposition(Disposition.KEEP)}
        assert keep_ids == set(result.keep_episode_ids)
        assert curation.original_n_episodes == result.n_original
        # every episode characterized exactly once
        assert {d.episode_id for d in curation.dispositions} == {
            ep.metadata.episode_id for ep in mixed_batch.episodes
        }

    def test_report_fills_anomaly_and_coverage(self, mixed_batch):
        """Passing report= populates anomaly_score / coverage_value from assessment."""
        from calibra.pruning import pruning_result_to_curation_report

        report_diag = Pipeline().run(mixed_batch)
        result = CoresetSelector(keep_fraction=0.5, max_spike_rate=0.05).select(
            mixed_batch, report_diag
        )

        without = pruning_result_to_curation_report(result, mixed_batch)
        with_report = pruning_result_to_curation_report(
            result, mixed_batch, report=report_diag
        )

        # No report → assessment axes stay None; quality_risk still set from result.
        assert all(d.anomaly_score is None for d in without.dispositions)
        assert all(d.coverage_value is None for d in without.dispositions)
        assert any(d.quality_risk is not None for d in without.dispositions)

        # With report → anomaly_score populated for every episode.
        assert all(d.anomaly_score is not None for d in with_report.dispositions)
        assert all(
            0.0 <= d.anomaly_score <= 1.0 for d in with_report.dispositions
        )
        # dispositions themselves are unchanged by enrichment
        assert [d.disposition for d in without.dispositions] == [
            d.disposition for d in with_report.dispositions
        ]

    def test_calibra_score_and_redundancy_derived(self, mixed_batch):
        from calibra.pruning import pruning_result_to_curation_report

        report_diag = Pipeline().run(mixed_batch)
        result = CoresetSelector(keep_fraction=0.5, max_spike_rate=0.05).select(
            mixed_batch, report_diag
        )
        cur = pruning_result_to_curation_report(result, mixed_batch, report=report_diag)

        for d in cur.dispositions:
            # calibra_score = 100·(1 - quality_risk), always available
            assert d.quality_risk is not None
            assert d.calibra_score == pytest.approx(100.0 * (1.0 - d.quality_risk), abs=0.05)
            # redundancy = 1 - coverage_value where coverage_value exists
            if d.coverage_value is not None:
                assert d.redundancy == pytest.approx(1.0 - d.coverage_value, abs=1e-4)
            else:
                assert d.redundancy is None
        assert any(d.redundancy is not None for d in cur.dispositions)

        # without a report there is no coverage_value → no redundancy, but
        # calibra_score still derives from result.quality_scores
        no_report = pruning_result_to_curation_report(result, mixed_batch)
        assert all(d.redundancy is None for d in no_report.dispositions)
        assert any(d.calibra_score is not None for d in no_report.dispositions)

    def test_pruningresult_to_curation_report_method(self, mixed_batch):
        from calibra.schema.comparison import Disposition

        report_diag = Pipeline().run(mixed_batch)
        result = CoresetSelector(keep_fraction=0.5, max_spike_rate=0.05).select(
            mixed_batch, report_diag
        )

        via_method = result.to_curation_report(
            mixed_batch, report=report_diag, redundant_disposition=Disposition.ANNOTATE
        )
        assert via_method.original_n_episodes == result.n_original
        assert {d.episode_id for d in via_method.dispositions} == {
            ep.metadata.episode_id for ep in mixed_batch.episodes
        }
        # default redundant_disposition is DROP
        default = result.to_curation_report(mixed_batch)
        assert "ANNOTATE" not in default.disposition_counts()
