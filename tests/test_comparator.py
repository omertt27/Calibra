"""
Tests for DatasetComparator (Phase 2).

All fixtures are synthetic — no real dataset files required.
Permutation tests use n_permutations=199 (default) and fixed seed=42.
"""
from __future__ import annotations

import numpy as np
import pytest

from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.pipeline import Pipeline
from calibra.comparison.comparator import DatasetComparator, _permutation_test, _infer_flag_direction


# ── fixtures ─────────────────────────────────────────────────────────────────

def _make_ep(
    n_steps: int = 100,
    dt: float = 0.1,
    action_dim: int = 4,
    jitter_std: float = 0.0,
    seed: int = 0,
) -> Episode:
    """Synthetic episode with optional timing jitter."""
    rng = np.random.default_rng(seed)
    deltas = np.full(n_steps, dt)
    if jitter_std > 0:
        deltas = np.abs(deltas + rng.normal(0, jitter_std, n_steps))
    timestamps = np.concatenate([[0.0], np.cumsum(deltas[:-1])])
    return Episode(
        metadata=EpisodeMetadata(episode_id=f"ep_{seed}"),
        timestamps=timestamps,
        observations={"proprio": rng.random((n_steps, 4)).astype(np.float32)},
        actions=rng.random((n_steps, action_dim)).astype(np.float32),
    )


def _make_smooth_ep(n_steps: int = 100, dt: float = 0.1, seed: int = 0) -> Episode:
    """Episode with smooth sine-wave actions (good LDLJ)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, n_steps)
    timestamps = np.linspace(0, n_steps * dt, n_steps)
    actions = np.column_stack([np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 0.1))] * 4)
    return Episode(
        metadata=EpisodeMetadata(episode_id=f"ep_{seed}"),
        timestamps=timestamps,
        observations={"proprio": rng.random((n_steps, 4)).astype(np.float32)},
        actions=actions.astype(np.float32),
    )


def _make_jerky_ep(n_steps: int = 100, dt: float = 0.1, seed: int = 0) -> Episode:
    """Episode with high-frequency random actions (bad LDLJ)."""
    rng = np.random.default_rng(seed)
    timestamps = np.linspace(0, n_steps * dt, n_steps)
    actions = rng.uniform(-1, 1, (n_steps, 4)).astype(np.float32)
    return Episode(
        metadata=EpisodeMetadata(episode_id=f"ep_{seed}"),
        timestamps=timestamps,
        observations={"proprio": rng.random((n_steps, 4)).astype(np.float32)},
        actions=actions,
    )


@pytest.fixture
def clean_batch_diverse() -> EpisodeBatch:
    """10 clean episodes with distinct seeds (low jitter, varied per-episode)."""
    episodes = [_make_ep(jitter_std=0.0, seed=i) for i in range(10)]
    return EpisodeBatch(episodes=episodes, dataset_name="clean",
                        format="hdf5", source_path="/tmp/clean.h5")


@pytest.fixture
def jittery_batch_diverse() -> EpisodeBatch:
    """10 jittery episodes with distinct seeds (high jitter per episode)."""
    episodes = [_make_ep(jitter_std=0.025, seed=i) for i in range(10)]
    return EpisodeBatch(episodes=episodes, dataset_name="jittery",
                        format="hdf5", source_path="/tmp/jittery.h5")


@pytest.fixture
def smooth_batch() -> EpisodeBatch:
    """10 smooth sine-wave episodes."""
    episodes = [_make_smooth_ep(seed=i) for i in range(10)]
    return EpisodeBatch(episodes=episodes, dataset_name="smooth",
                        format="hdf5", source_path="/tmp/smooth.h5")


@pytest.fixture
def jerky_batch() -> EpisodeBatch:
    """10 jerky random-action episodes."""
    episodes = [_make_jerky_ep(seed=i) for i in range(10)]
    return EpisodeBatch(episodes=episodes, dataset_name="jerky",
                        format="hdf5", source_path="/tmp/jerky.h5")


@pytest.fixture
def pipeline_temporal_only() -> Pipeline:
    return Pipeline(analyzers=[TemporalAnalyzer(n_bootstrap=200)])


@pytest.fixture
def pipeline_smoothness_only() -> Pipeline:
    return Pipeline(analyzers=[ControlSmoothnessAnalyzer(n_bootstrap=200)])


# ── permutation test unit tests ───────────────────────────────────────────────

def test_permutation_test_identical_groups():
    """Identical groups → delta=0, p_value≈1."""
    vals = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.19, 0.13, 0.16, 0.14]
    delta, p = _permutation_test(vals, vals, n_permutations=199)
    assert delta == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_permutation_test_large_effect_significant():
    """Large mean shift → significant (p < 0.05)."""
    a = [0.0] * 10
    b = [1.0] * 10
    delta, p = _permutation_test(a, b, n_permutations=199)
    assert delta == pytest.approx(1.0)
    assert p < 0.05


def test_permutation_test_insufficient_data():
    """Fewer than 4 values per group → p_value=1.0 (no reliable test)."""
    delta, p = _permutation_test([0.1, 0.2, 0.3], [1.0, 2.0, 3.0], n_permutations=199)
    assert p == pytest.approx(1.0)
    assert delta == pytest.approx(2.0 - 0.2)


def test_permutation_test_filters_none_values():
    """None values are skipped; test still runs on valid entries."""
    a = [0.1, None, 0.15, None, 0.12, 0.18, 0.11, 0.19, 0.13, 0.16]
    b = [1.0, None, 1.05, None, 1.02, 1.08, 1.01, 1.09, 1.03, 1.06]
    delta, p = _permutation_test(a, b, n_permutations=199)
    assert delta > 0.8
    assert p < 0.05


# ── direction inference ───────────────────────────────────────────────────────

def test_direction_jitter_increase_is_degraded():
    assert _infer_flag_direction("timestamp_jitter_cv", delta=0.1) == "degraded"


def test_direction_jitter_decrease_is_improved():
    assert _infer_flag_direction("timestamp_jitter_cv", delta=-0.1) == "improved"


def test_direction_ldlj_increase_is_improved():
    assert _infer_flag_direction("ldlj", delta=2.0) == "improved"


def test_direction_ldlj_decrease_is_degraded():
    assert _infer_flag_direction("ldlj", delta=-2.0) == "degraded"


def test_direction_contact_density_is_ambiguous():
    assert _infer_flag_direction("contact_density", delta=0.1) == "ambiguous"


def test_direction_zero_delta_is_ambiguous():
    assert _infer_flag_direction("timestamp_jitter_cv", delta=0.0) == "ambiguous"


# ── comparator integration tests ──────────────────────────────────────────────

def test_same_report_no_significant_degradation(clean_batch_diverse, pipeline_temporal_only):
    """Comparing a report against itself: no significant degraded flags."""
    report = pipeline_temporal_only.run(clean_batch_diverse)
    comparator = DatasetComparator()
    result = comparator.compare(report, report)
    assert len(result.degraded) == 0


def test_jitter_degradation_detected(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """Jittery candidate vs clean baseline: timestamp_jitter_cv flagged as degraded."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    comparator = DatasetComparator(alpha=0.05)
    result = comparator.compare(base_report, cand_report)

    degraded_metrics = {f.metric for f in result.degraded}
    assert "timestamp_jitter_cv" in degraded_metrics


def test_jitter_degradation_direction_and_delta(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """DriftFlag for jitter has positive delta and direction='degraded'."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    result = DatasetComparator().compare(base_report, cand_report)
    jitter_flag = next(f for f in result.drift_flags if f.metric == "timestamp_jitter_cv")

    assert jitter_flag.direction == "degraded"
    assert jitter_flag.delta > 0
    assert jitter_flag.significant


def test_smoothness_improvement_detected(jerky_batch, smooth_batch, pipeline_smoothness_only):
    """Smooth candidate vs jerky baseline: ldlj flagged as improved."""
    base_report = pipeline_smoothness_only.run(jerky_batch)
    cand_report = pipeline_smoothness_only.run(smooth_batch)

    result = DatasetComparator().compare(base_report, cand_report)
    improved_metrics = {f.metric for f in result.improved}
    assert "ldlj" in improved_metrics


def test_drift_flag_p_value_present(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """DriftFlags for metrics with per-episode data have a p_value, not None."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    result = DatasetComparator().compare(base_report, cand_report)
    jitter_flag = next(f for f in result.drift_flags if f.metric == "timestamp_jitter_cv")

    assert jitter_flag.p_value is not None
    assert 0.0 <= jitter_flag.p_value <= 1.0


def test_comparison_report_degraded_and_improved_properties(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """ComparisonReport.degraded and .improved are consistent subsets of drift_flags."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    result = DatasetComparator().compare(base_report, cand_report)

    for f in result.degraded:
        assert f in result.drift_flags
        assert f.direction == "degraded"
        assert f.significant

    for f in result.improved:
        assert f in result.drift_flags
        assert f.direction == "improved"
        assert f.significant


def test_comparison_report_metadata(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """ComparisonReport carries correct dataset names and episode counts."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    result = DatasetComparator(n_permutations=99).compare(base_report, cand_report)

    assert result.baseline_name == "clean"
    assert result.candidate_name == "jittery"
    assert result.baseline_n_episodes == 10
    assert result.candidate_n_episodes == 10
    assert result.n_permutations == 99


def test_comparison_report_summary_renders(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """ComparisonReport.summary() returns a non-empty string."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    result = DatasetComparator().compare(base_report, cand_report)
    s = result.summary()

    assert "Calibra Comparison Report" in s
    assert "clean" in s
    assert "jittery" in s


def test_drift_flag_relative_change(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """relative_change = delta / |baseline.value| when baseline is non-zero."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    result = DatasetComparator().compare(base_report, cand_report)
    for flag in result.drift_flags:
        if flag.delta is not None and flag.relative_change is not None:
            b_val = flag.baseline_observed.value
            assert b_val is not None
            if abs(b_val) > 1e-12:
                expected_rel = flag.delta / abs(b_val)
                assert flag.relative_change == pytest.approx(expected_rel, rel=1e-6)


def test_drift_risk_level_degraded_tracks_candidate(
    clean_batch_diverse, jittery_batch_diverse, pipeline_temporal_only
):
    """A significant degraded DriftFlag's level matches the candidate flag's level."""
    base_report = pipeline_temporal_only.run(clean_batch_diverse)
    cand_report = pipeline_temporal_only.run(jittery_batch_diverse)

    result = DatasetComparator().compare(base_report, cand_report)
    for drift_flag in result.degraded:
        # Find the corresponding candidate flag.
        cand_flag = next(
            f for r in cand_report.analyzer_results
            for f in r.flags if f.metric == drift_flag.metric
        )
        assert drift_flag.level == cand_flag.level


def test_full_pipeline_comparison(clean_batch_diverse, jittery_batch_diverse):
    """End-to-end: full Pipeline on both batches, then compare."""
    pipeline = Pipeline()
    base_report = pipeline.run(clean_batch_diverse)
    cand_report = pipeline.run(jittery_batch_diverse)

    result = DatasetComparator().compare(base_report, cand_report)

    # Should have drift flags from multiple analyzers.
    analyzer_names = {f.analyzer_name for f in result.drift_flags}
    assert len(analyzer_names) >= 2
    assert len(result.drift_flags) >= 4
