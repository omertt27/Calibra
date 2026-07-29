"""Tests for calibra.assessment — the anomaly/quality_risk/coverage_value split."""

from __future__ import annotations

import numpy as np

from calibra.assessment import compute_episode_assessments, rank_for_review
from calibra.pipeline import Pipeline
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import DiagnosticReport

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_batch(n_eps: int = 8, n_steps: int = 100, outlier_idx: int | None = None) -> EpisodeBatch:
    """
    Smooth, low-jerk baseline trajectories (a random walk, like a real
    controlled motion) with an optional injected outlier episode that gets a
    handful of large, localized action spikes — a real jerk/spike-rate
    anomaly, unlike naive iid-per-step noise where every episode already
    saturates relative metrics like velocity-discontinuity rate.
    """
    rng = np.random.default_rng(0)
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        acts = np.cumsum(rng.normal(0, 0.02, (n_steps, 6)), axis=0).astype(np.float32)
        if outlier_idx is not None and i == outlier_idx:
            spike_steps = rng.choice(np.arange(10, n_steps - 10), size=8, replace=False)
            acts[spike_steps] += rng.normal(0, 3.0, (8, 6)).astype(np.float32)
        obs = {"proprio": rng.uniform(-1, 1, (n_steps, 8)).astype(np.float32)}
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=ts,
                observations=obs,
                actions=acts,
            )
        )
    return EpisodeBatch(
        episodes=episodes, dataset_name="assessment_test", format="hdf5", source_path="/tmp/test.h5"
    )


# ── tests ────────────────────────────────────────────────────────────────────


class TestComputeEpisodeAssessments:
    def test_one_assessment_per_episode(self):
        report = Pipeline().run(_make_batch(n_eps=6))
        assessments = compute_episode_assessments(report)
        assert {a.episode_id for a in assessments} == {f"ep_{i}" for i in range(6)}

    def test_empty_report_returns_empty(self):
        report = DiagnosticReport(
            dataset_name="empty", source_path="/tmp/x", format="hdf5", n_episodes=0, n_samples=0
        )
        assert compute_episode_assessments(report) == []

    def test_outlier_episode_flagged_with_highest_quality_risk(self):
        # anomaly_score alone can tie at 1.0 across several episodes (each is
        # the batch max on *some* metric); quality_risk is the more targeted
        # signal since the injected spikes hit jerk/spike-rate specifically.
        batch = _make_batch(n_eps=8, outlier_idx=3)
        report = Pipeline().run(batch)
        assessments = {a.episode_id: a for a in compute_episode_assessments(report)}
        outlier = assessments["ep_3"]
        others = [a.quality_risk for eid, a in assessments.items() if eid != "ep_3"]
        assert outlier.quality_risk > max(others)
        reason_metrics = {r.metric for r in outlier.reasons}
        assert "per_episode_spike_rate" in reason_metrics or "per_episode_ldlj" in reason_metrics

    def test_reasons_reference_real_metrics(self):
        batch = _make_batch(n_eps=8, outlier_idx=2)
        report = Pipeline().run(batch)
        assessments = {a.episode_id: a for a in compute_episode_assessments(report)}
        for reason in assessments["ep_2"].reasons:
            assert 0.0 <= reason.percentile <= 100.0
            assert reason.metric

    def test_coverage_value_present_when_influence_ran(self):
        report = Pipeline().run(_make_batch(n_eps=6))
        assessments = compute_episode_assessments(report)
        assert all(a.coverage_value is not None for a in assessments)
        assert all(0.0 <= a.coverage_value <= 1.0 for a in assessments)

    def test_coverage_value_none_without_influence_analyzer(self):
        from calibra.analyzers.temporal import TemporalAnalyzer

        report = Pipeline(analyzers=[TemporalAnalyzer()]).run(_make_batch(n_eps=6))
        assessments = compute_episode_assessments(report)
        assert all(a.coverage_value is None for a in assessments)

    def test_quality_risk_matches_pruning_formula(self):
        from calibra.comparison.comparator import _extract_ep_data
        from calibra.pruning import compute_quality_scores_for_ids

        report = Pipeline().run(_make_batch(n_eps=6))
        assessments = {a.episode_id: a for a in compute_episode_assessments(report)}
        expected = compute_quality_scores_for_ids(report.episode_ids, _extract_ep_data(report))
        for episode_id, exp_score in expected.items():
            assert assessments[episode_id].quality_risk == exp_score


def _make_synthetic_report(episode_ids: list[str], ldlj_values: list[float]) -> DiagnosticReport:
    """A hand-built report with one exact, fully-controlled metric — avoids
    depending on real analyzers' emergent behavior on contrived trajectories."""
    from calibra.schema.report import AnalyzerResult

    return DiagnosticReport(
        dataset_name="synthetic",
        source_path="/tmp/synthetic",
        format="hdf5",
        n_episodes=len(episode_ids),
        n_samples=len(episode_ids) * 10,
        episode_ids=episode_ids,
        analyzer_results=[
            AnalyzerResult(
                analyzer_name="control_smoothness",
                raw_metrics={"per_episode_ldlj": ldlj_values},
            )
        ],
    )


def _make_metadata_only_batch(episode_ids: list[str], tasks: list[str]) -> EpisodeBatch:
    """Episodes carrying only the metadata needed to resolve a group key."""
    episodes = [
        Episode(
            metadata=EpisodeMetadata(episode_id=eid, task_description=task),
            timestamps=np.arange(10, dtype=np.float64),
            observations={},
            actions=np.zeros((10, 1), dtype=np.float32),
        )
        for eid, task in zip(episode_ids, tasks)
    ]
    return EpisodeBatch(
        episodes=episodes, dataset_name="synthetic", format="hdf5", source_path="/tmp/synthetic"
    )


class TestContextualGrouping:
    # "hard" episodes cluster around a much worse (more negative) LDLJ than
    # "easy" ones, but are internally consistent (low variance) — a
    # legitimately harder task, not a broken one. hard_1 (-25.1) is only
    # slightly worse than its five "hard" peers, but is a clear tail outlier
    # once compared against the much-better "easy" population too.
    _EASY_IDS = [f"easy_{i}" for i in range(6)]
    _HARD_IDS = [f"hard_{i}" for i in range(6)]
    _EASY_LDLJ = [-5.0, -5.1, -4.9, -5.05, -4.95, -5.0]
    _HARD_LDLJ = [-25.0, -25.1, -24.9, -25.05, -24.95, -25.0]

    def test_global_ranking_flags_a_hard_episode_that_is_normal_for_its_task(self):
        report = _make_synthetic_report(
            self._EASY_IDS + self._HARD_IDS, self._EASY_LDLJ + self._HARD_LDLJ
        )
        assessments = {a.episode_id: a for a in compute_episode_assessments(report)}
        # hard_1 is unremarkable among other "hard" episodes but gets flagged
        # purely because it's compared against the easier population too.
        assert assessments["hard_1"].reasons
        assert assessments["hard_1"].reasons[0].metric == "per_episode_ldlj"

    def test_group_by_task_stops_flagging_that_same_episode(self):
        episode_ids = self._EASY_IDS + self._HARD_IDS
        report = _make_synthetic_report(episode_ids, self._EASY_LDLJ + self._HARD_LDLJ)
        batch = _make_metadata_only_batch(
            episode_ids, ["easy"] * len(self._EASY_IDS) + ["hard"] * len(self._HARD_IDS)
        )
        assessments = {
            a.episode_id: a
            for a in compute_episode_assessments(report, batch=batch, group_by=["task"])
        }
        # Compared only against its own task, hard_1 is unremarkable.
        assert assessments["hard_1"].reasons == []

    def test_group_by_without_batch_falls_back_to_global(self):
        easy_ids = [f"easy_{i}" for i in range(6)]
        hard_ids = [f"hard_{i}" for i in range(6)]
        report = _make_synthetic_report(
            easy_ids + hard_ids, [-5.0, -5.1, -4.9, -5.05, -4.95, -5.0, -25, -25, -25, -25, -25, -25]
        )
        with_group_but_no_batch = compute_episode_assessments(report, group_by=["task"])
        without_group = compute_episode_assessments(report)
        assert [a.anomaly_score for a in with_group_but_no_batch] == [
            a.anomaly_score for a in without_group
        ]


class TestRankForReview:
    def test_sorted_descending_by_review_priority(self):
        report = Pipeline().run(_make_batch(n_eps=10, outlier_idx=5))
        ranked = rank_for_review(compute_episode_assessments(report))
        priorities = [a.review_priority for a in ranked]
        assert priorities == sorted(priorities, reverse=True)

    def test_outlier_ranks_first(self):
        report = Pipeline().run(_make_batch(n_eps=10, outlier_idx=5))
        ranked = rank_for_review(compute_episode_assessments(report))
        assert ranked[0].episode_id == "ep_5"

    def test_higher_coverage_value_lowers_priority_at_fixed_risk(self):
        from calibra.assessment import EpisodeAssessment

        low_coverage = EpisodeAssessment(
            episode_id="a", anomaly_score=0.5, quality_risk=0.5, coverage_value=0.0
        )
        high_coverage = EpisodeAssessment(
            episode_id="b", anomaly_score=0.5, quality_risk=0.5, coverage_value=1.0
        )
        assert high_coverage.review_priority < low_coverage.review_priority
