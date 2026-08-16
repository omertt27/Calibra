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


class TestPipelineMode:
    def test_fast_mode_runs_fewer_analyzers_than_full(self):
        batch = _make_batch()
        fast_names = {r.analyzer_name for r in Pipeline(mode="fast").run(batch).analyzer_results}
        full_names = {r.analyzer_name for r in Pipeline(mode="full").run(batch).analyzer_results}
        assert fast_names < full_names

    def test_fast_mode_excludes_influence_analyzer(self):
        batch = _make_batch()
        report = Pipeline(mode="fast").run(batch)
        names = {r.analyzer_name for r in report.analyzer_results}
        assert "influence" not in names

    def test_default_mode_is_full(self):
        batch = _make_batch()
        default_names = {r.analyzer_name for r in Pipeline().run(batch).analyzer_results}
        full_names = {r.analyzer_name for r in Pipeline(mode="full").run(batch).analyzer_results}
        assert default_names == full_names

    def test_explicit_analyzers_overrides_mode(self):
        batch = _make_batch()
        report = Pipeline(analyzers=[TemporalAnalyzer()], mode="fast").run(batch)
        names = {r.analyzer_name for r in report.analyzer_results}
        assert names == {"temporal_stability"}

    def test_invalid_mode_raises(self):
        import pytest

        with pytest.raises(ValueError):
            Pipeline(mode="bogus")


class TestPipelineEmptyBatch:
    def test_empty_batch_no_crash(self):
        empty = EpisodeBatch(
            episodes=[], dataset_name="empty", format="hdf5", source_path="/tmp/empty.h5"
        )
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


class TestReproducibilityMetadata:
    def test_calibra_version_stamped(self):
        from calibra import __version__

        report = Pipeline().run(_make_batch())
        assert report.calibra_version == __version__

    def test_analyzer_versions_cover_ran_analyzers_only(self):
        pipeline = Pipeline(mode="fast")
        report = pipeline.run(_make_batch())
        ran_names = {r.analyzer_name for r in report.analyzer_results}
        assert set(report.analyzer_versions.keys()) == ran_names
        assert all(v for v in report.analyzer_versions.values())

    def test_skipped_analyzers_excluded_from_versions(self):
        # Empty-batch runs still populate the standard analyzer set (they all
        # degrade gracefully), so use a custom single-analyzer pipeline to
        # keep this test's expectation exact.
        pipeline = Pipeline(analyzers=[TemporalAnalyzer()])
        report = pipeline.run(_make_batch())
        assert list(report.analyzer_versions.keys()) == ["temporal_stability"]

    def test_config_hash_is_deterministic_for_same_inputs(self):
        batch = _make_batch()
        r1 = Pipeline(mode="fast").run(batch)
        r2 = Pipeline(mode="fast").run(batch)
        assert r1.config_hash == r2.config_hash
        assert r1.config_hash != ""

    def test_config_hash_changes_with_policy_family(self):
        batch = _make_batch()
        r1 = Pipeline(mode="fast").run(batch, policy_family="act")
        r2 = Pipeline(mode="fast").run(batch, policy_family="diffusion")
        assert r1.config_hash != r2.config_hash

    def test_config_hash_changes_with_analyzer_set(self):
        batch = _make_batch()
        r1 = Pipeline(mode="fast").run(batch)
        r2 = Pipeline(mode="full").run(batch)
        assert r1.config_hash != r2.config_hash

    def test_generated_at_is_iso8601(self):
        report = Pipeline().run(_make_batch())
        # Should not raise — fromisoformat accepts the timespec="seconds" format we emit.
        from datetime import datetime

        datetime.fromisoformat(report.generated_at)

    def test_summary_includes_reproducibility_footer(self):
        report = Pipeline().run(_make_batch())
        summary = report.summary()
        assert f"Calibra v{report.calibra_version}" in summary
        assert report.config_hash in summary

    def test_summary_omits_footer_when_unstamped(self):
        # A hand-built report (e.g. in a test or an old cached artifact) with
        # no reproducibility metadata should render cleanly, not blank fields.
        report = DiagnosticReport(
            dataset_name="d",
            source_path="/tmp/d",
            format="hdf5",
            n_episodes=1,
            n_samples=1,
        )
        assert "Calibra v" not in report.summary()

    def test_json_roundtrip_preserves_reproducibility_fields(self):
        report = Pipeline().run(_make_batch())
        restored = DiagnosticReport.model_validate_json(report.model_dump_json())
        assert restored.calibra_version == report.calibra_version
        assert restored.config_hash == report.config_hash
        assert restored.analyzer_versions == report.analyzer_versions
