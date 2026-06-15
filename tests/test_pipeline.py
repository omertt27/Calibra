"""Tests for the Pipeline assembly layer."""
from __future__ import annotations

import numpy as np

from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.pipeline import Pipeline
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import DiagnosticReport, RiskLevel


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_batch(
    n_eps: int = 5,
    n_steps: int = 100,
    action_dim: int = 6,
) -> EpisodeBatch:
    rng = np.random.default_rng(0)
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        acts = rng.uniform(-1, 1, (n_steps, action_dim)).astype(np.float32)
        obs = {"proprio": rng.uniform(-1, 1, (n_steps, 8)).astype(np.float32)}
        episodes.append(Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
            timestamps=ts,
            observations=obs,
            actions=acts,
        ))
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="pipeline_test",
        format="hdf5",
        source_path="/tmp/test.h5",
    )


# ── tests ────────────────────────────────────────────────────────────────────

class TestPipelineRun:
    def test_returns_diagnostic_report(self):
        batch = _make_batch()
        report = Pipeline().run(batch)
        assert isinstance(report, DiagnosticReport)

    def test_report_metadata_matches_batch(self):
        batch = _make_batch(n_eps=7, n_steps=50)
        report = Pipeline().run(batch)
        assert report.dataset_name == batch.dataset_name
        assert report.n_episodes == 7
        assert report.n_samples == 7 * 50
        assert report.format == "hdf5"

    def test_all_three_analyzers_run(self):
        batch = _make_batch()
        report = Pipeline().run(batch)
        names = {r.analyzer_name for r in report.analyzer_results}
        assert "temporal_stability" in names
        assert "control_smoothness" in names
        assert "coverage_entropy" in names

    def test_policy_family_propagated(self):
        batch = _make_batch()
        report = Pipeline().run(batch, policy_family="diffusion")
        assert report.policy_family == "diffusion"

    def test_no_policy_family_means_no_hints(self):
        batch = _make_batch()
        report = Pipeline().run(batch, policy_family=None)
        assert report.hints == []

    def test_policy_family_produces_hints(self):
        batch = _make_batch(n_steps=200)
        report = Pipeline().run(batch, policy_family="diffusion")
        assert len(report.hints) > 0

    def test_summary_renders_without_error(self):
        batch = _make_batch()
        report = Pipeline().run(batch)
        summary = report.summary()
        assert "pipeline_test" in summary
        assert "hdf5" in summary

    def test_json_roundtrip(self):
        batch = _make_batch()
        report = Pipeline().run(batch)
        restored = DiagnosticReport.model_validate_json(report.model_dump_json())
        assert restored.n_episodes == report.n_episodes
        assert len(restored.flags) == len(report.flags)


class TestPipelineCustomAnalyzers:
    def test_single_analyzer_pipeline(self):
        batch = _make_batch()
        report = Pipeline(analyzers=[TemporalAnalyzer()]).run(batch)
        names = {r.analyzer_name for r in report.analyzer_results}
        assert names == {"temporal_stability"}

    def test_flag_count_reflects_all_analyzers(self):
        batch = _make_batch()
        full_report = Pipeline().run(batch)
        temporal_only = Pipeline(analyzers=[TemporalAnalyzer()]).run(batch)
        # Full pipeline has flags from all three analyzers
        assert len(full_report.flags) >= len(temporal_only.flags)

    def test_empty_analyzer_list(self):
        batch = _make_batch()
        report = Pipeline(analyzers=[]).run(batch)
        assert report.analyzer_results == []
        assert report.flags == []


class TestPipelineEmptyBatch:
    def test_empty_batch_no_crash(self):
        empty = EpisodeBatch(episodes=[], dataset_name="empty",
                             format="hdf5", source_path="/tmp/empty.h5")
        report = Pipeline().run(empty)
        assert isinstance(report, DiagnosticReport)
        assert report.n_episodes == 0
        assert report.flags == []


class TestPipelineFlagAggregation:
    def test_flags_at_level_cross_analyzer(self):
        batch = _make_batch()
        report = Pipeline().run(batch)
        # flags property spans all analyzer_results
        total = sum(len(r.flags) for r in report.analyzer_results)
        assert len(report.flags) == total

    def test_summary_counts_critical_and_warning(self):
        batch = _make_batch()
        report = Pipeline().run(batch)
        summary = report.summary()
        n_crit = len(report.flags_at_level(RiskLevel.CRITICAL))
        n_warn = len(report.flags_at_level(RiskLevel.WARNING))
        assert f"{n_crit} critical" in summary
        assert f"{n_warn} warning" in summary
