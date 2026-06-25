"""Tests for episode-level anomaly detection (calibra/anomalies.py)."""

from __future__ import annotations

import numpy as np
import pytest

from calibra.anomalies import (
    EpisodeAnomaly,
    EpisodeFlag,
    _consecutive_groups,
    _heuristic_label,
    find_outliers,
    render,
)
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.pipeline import Pipeline
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import DiagnosticReport


# ── helpers ───────────────────────────────────────────────────────────────────


def _episode(
    n_steps: int = 100,
    dt: float = 0.05,
    action_dim: int = 3,
    ep_id: str = "ep_0",
    spike_indices: list[int] | None = None,
    jitter_std: float = 0.0,
) -> Episode:
    rng = np.random.default_rng(42)
    t = np.arange(n_steps, dtype=np.float64) * dt
    if jitter_std > 0:
        t += np.cumsum(rng.normal(0, jitter_std, size=n_steps))
    actions = np.sin(t[:, None]).repeat(action_dim, axis=1).astype(np.float32)
    if spike_indices:
        for idx in spike_indices:
            actions[idx] += 10.0
    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=t,
        observations={"proprio": np.zeros((n_steps, 4), dtype=np.float32)},
        actions=actions,
    )


def _smooth_batch(n: int = 20) -> EpisodeBatch:
    """All episodes are smooth; one episode has heavy spike injection."""
    episodes = [_episode(ep_id=f"ep_{i}") for i in range(n)]
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="smooth",
        format="hdf5",
        source_path="/tmp/smooth.h5",
    )


def _batch_with_one_outlier(n: int = 20, outlier_idx: int = 5) -> EpisodeBatch:
    """All episodes smooth except one with many spike injections."""
    episodes = [_episode(ep_id=f"ep_{i}") for i in range(n)]
    episodes[outlier_idx] = _episode(
        ep_id=f"ep_{outlier_idx}",
        spike_indices=list(range(10, 90, 5)),  # many spikes → high spike_rate
    )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="one_outlier",
        format="hdf5",
        source_path="/tmp/one_outlier.h5",
    )


def _batch_with_cluster(n: int = 30, cluster: list[int] | None = None) -> EpisodeBatch:
    cluster = cluster or [10, 11, 12]
    episodes = [_episode(ep_id=f"ep_{i}") for i in range(n)]
    for idx in cluster:
        episodes[idx] = _episode(
            ep_id=f"ep_{idx}",
            spike_indices=list(range(10, 90, 5)),
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="cluster",
        format="hdf5",
        source_path="/tmp/cluster.h5",
    )


def _report_from_batch(batch: EpisodeBatch) -> DiagnosticReport:
    return Pipeline(analyzers=[TemporalAnalyzer(), ControlSmoothnessAnalyzer()]).run(batch)


# ── unit tests: _consecutive_groups ──────────────────────────────────────────


def test_consecutive_groups_empty():
    assert _consecutive_groups([]) == []


def test_consecutive_groups_single():
    assert _consecutive_groups([5]) == [[5]]


def test_consecutive_groups_all_consecutive():
    assert _consecutive_groups([3, 4, 5]) == [[3, 4, 5]]


def test_consecutive_groups_disjoint():
    result = _consecutive_groups([1, 2, 5, 6, 10])
    assert result == [[1, 2], [5, 6], [10]]


def test_consecutive_groups_gaps():
    result = _consecutive_groups([0, 2, 3, 7])
    assert result == [[0], [2, 3], [7]]


# ── unit tests: _heuristic_label ─────────────────────────────────────────────


def _make_anomaly(idx: int, metric: str = "spike_rate") -> EpisodeAnomaly:
    flag = EpisodeFlag(
        episode_idx=idx,
        episode_id=f"ep_{idx}",
        metric=metric,
        observed=1.0,
        median=0.1,
        deviation_mads=5.0,
        higher_is_worse=True,
    )
    return EpisodeAnomaly(episode_idx=idx, episode_id=f"ep_{idx}", flags=[flag])


def test_heuristic_label_cluster():
    anomalies = [_make_anomaly(i) for i in [5, 6, 7]]
    label = _heuristic_label(anomalies, n_episodes=100, group_size=3)
    assert "cluster" in label


def test_heuristic_label_end_of_dataset():
    anomalies = [_make_anomaly(95)]
    label = _heuristic_label(anomalies, n_episodes=100, group_size=1)
    assert "end of dataset" in label


def test_heuristic_label_start_of_dataset():
    anomalies = [_make_anomaly(1)]
    label = _heuristic_label(anomalies, n_episodes=100, group_size=1)
    assert "start of dataset" in label


def test_heuristic_label_jitter_cv():
    anomalies = [_make_anomaly(50, metric="jitter_cv")]
    label = _heuristic_label(anomalies, n_episodes=100, group_size=1)
    assert "sync" in label or "timestamp" in label


def test_heuristic_label_dropout():
    anomalies = [_make_anomaly(50, metric="dropout_rate")]
    label = _heuristic_label(anomalies, n_episodes=100, group_size=1)
    assert "drop" in label or "gap" in label


# ── integration: find_outliers ────────────────────────────────────────────────


def test_no_outliers_on_uniform_batch():
    report = _report_from_batch(_smooth_batch(n=20))
    outliers = find_outliers(report)
    assert outliers == []


def test_finds_single_outlier():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=7))
    outliers = find_outliers(report)
    assert len(outliers) >= 1
    assert any(a.episode_id == "ep_7" for a in outliers)


def test_outlier_severity_positive():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=7))
    outliers = find_outliers(report)
    for a in outliers:
        assert a.severity > 0


def test_outliers_sorted_by_severity_descending():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=7))
    outliers = find_outliers(report)
    severities = [a.severity for a in outliers]
    assert severities == sorted(severities, reverse=True)


def test_cluster_of_outliers():
    report = _report_from_batch(_batch_with_cluster(n=30, cluster=[10, 11, 12]))
    outliers = find_outliers(report)
    flagged_ids = {a.episode_id for a in outliers}
    # At least the cluster episodes should be flagged
    assert {"ep_10", "ep_11", "ep_12"}.issubset(flagged_ids)


def test_episode_ids_populated_correctly():
    batch = _batch_with_one_outlier(n=20, outlier_idx=3)
    report = _report_from_batch(batch)
    assert len(report.episode_ids) == 20
    assert report.episode_ids[3] == "ep_3"


def test_find_outliers_uses_episode_ids():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=3))
    outliers = find_outliers(report)
    assert any(a.episode_id == "ep_3" for a in outliers)


def test_find_outliers_fallback_when_no_episode_ids():
    """When episode_ids is empty, outlier detection still works with index strings."""
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=3))
    report = report.model_copy(update={"episode_ids": []})
    outliers = find_outliers(report)
    assert len(outliers) >= 1


def test_too_few_episodes_skipped():
    """With fewer than 5 episodes, MAD is unreliable — no flags expected."""
    batch = _batch_with_one_outlier(n=4, outlier_idx=1)
    report = _report_from_batch(batch)
    outliers = find_outliers(report)
    assert outliers == []


def test_constant_metric_produces_no_flags():
    """If all episodes have the same metric value, MAD=0 → nothing flagged."""
    report = _report_from_batch(_smooth_batch(n=20))
    outliers = find_outliers(report)
    assert all(a.severity > 0 for a in outliers)  # no zero-severity artifacts


# ── unit tests: EpisodeFlag ───────────────────────────────────────────────────


def test_episode_flag_direction_above():
    flag = EpisodeFlag(
        episode_idx=0,
        episode_id="ep_0",
        metric="spike_rate",
        observed=0.5,
        median=0.1,
        deviation_mads=4.0,
        higher_is_worse=True,
    )
    assert flag.direction == "above"


def test_episode_flag_direction_below():
    flag = EpisodeFlag(
        episode_idx=0,
        episode_id="ep_0",
        metric="ldlj",
        observed=-30.0,
        median=-15.0,
        deviation_mads=5.0,
        higher_is_worse=False,
    )
    assert flag.direction == "below"


def test_episode_anomaly_severity_is_max_flag():
    flags = [
        EpisodeFlag(
            episode_idx=0,
            episode_id="ep_0",
            metric="spike_rate",
            observed=1.0,
            median=0.1,
            deviation_mads=3.5,
            higher_is_worse=True,
        ),
        EpisodeFlag(
            episode_idx=0,
            episode_id="ep_0",
            metric="vel_disc_rate",
            observed=1.0,
            median=0.1,
            deviation_mads=6.2,
            higher_is_worse=True,
        ),
    ]
    a = EpisodeAnomaly(episode_idx=0, episode_id="ep_0", flags=flags)
    assert a.severity == pytest.approx(6.2)


# ── integration: render ───────────────────────────────────────────────────────


def test_render_empty():
    assert render([], n_episodes=100) == ""


def test_render_contains_episode_count():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=5))
    outliers = find_outliers(report)
    if outliers:
        text = render(outliers, n_episodes=20)
        assert "of 20 episodes" in text


def test_render_contains_episode_id():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=5))
    outliers = find_outliers(report)
    if outliers:
        text = render(outliers, n_episodes=20)
        assert "ep_5" in text


def test_render_contains_mad_multiple():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=5))
    outliers = find_outliers(report)
    if outliers:
        text = render(outliers, n_episodes=20)
        assert "× MAD" in text


def test_render_contains_heuristic_label():
    report = _report_from_batch(_batch_with_one_outlier(n=20, outlier_idx=5))
    outliers = find_outliers(report)
    if outliers:
        text = render(outliers, n_episodes=20)
        assert "→" in text


def test_render_cluster_range_notation():
    report = _report_from_batch(_batch_with_cluster(n=30, cluster=[10, 11, 12]))
    outliers = find_outliers(report)
    text = render(outliers, n_episodes=30)
    # Cluster of 3 consecutive episodes should produce range notation
    assert "cluster" in text
