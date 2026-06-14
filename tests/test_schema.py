"""Tests for the data model layer (schema)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from calibra.schema.report import (
    AnalyzerResult,
    CompatibilityHint,
    DiagnosticReport,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)


class TestObservedValue:
    def test_renders_without_ci(self):
        v = ObservedValue(value=3.7, unit="bits")
        assert "3.7" in str(v)
        assert "bits" in str(v)
        assert "CI" not in str(v)

    def test_renders_with_ci(self):
        v = ObservedValue(value=118.0, unit="ms", ci_lower=105.0, ci_upper=131.0,
                          ci_method="bootstrap")
        s = str(v)
        assert "118" in s
        assert "105" in s
        assert "CI" in s

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValidationError):
            ObservedValue(value=1.0, ci_lower=10.0, ci_upper=5.0)


class TestRiskFlag:
    def test_render_critical(self):
        flag = RiskFlag(
            level=RiskLevel.CRITICAL,
            metric="camera_lag_std",
            observed=ObservedValue(value=118.0, unit="ms"),
            threshold=20.0,
            interpretation="High camera lag detected.",
            implication="Closed-loop policies will desync on contact transitions.",
            affected_fraction=0.041,
        )
        rendered = flag.render()
        assert "CRITICAL" in rendered
        assert "camera_lag_std" in rendered
        assert "118" in rendered
        assert "4.1%" in rendered
        assert "→" in rendered

    def test_render_ok(self):
        flag = RiskFlag(
            level=RiskLevel.OK,
            metric="jitter",
            observed=ObservedValue(value=0.05),
            interpretation="Timing is stable.",
            implication="No risk.",
        )
        assert "✅" in flag.render()


class TestDiagnosticReport:
    def _make_report(self) -> DiagnosticReport:
        return DiagnosticReport(
            dataset_name="test_ds",
            source_path="/tmp/test.h5",
            format="hdf5",
            n_episodes=50,
            n_samples=5000,
            analyzer_results=[
                AnalyzerResult(
                    analyzer_name="temporal_stability",
                    flags=[
                        RiskFlag(
                            level=RiskLevel.CRITICAL,
                            metric="camera_lag_std[camera_rgb]",
                            observed=ObservedValue(value=118.0, unit="ms"),
                            threshold=20.0,
                            interpretation="Camera lag too high.",
                            implication="Desync risk.",
                        ),
                        RiskFlag(
                            level=RiskLevel.WARNING,
                            metric="action_obs_misalignment",
                            observed=ObservedValue(value=0.041, unit="fraction"),
                            threshold=0.01,
                            interpretation="4.1% of samples misaligned.",
                            implication="Corrupted BC signal.",
                            affected_fraction=0.041,
                        ),
                    ],
                    hints=[
                        CompatibilityHint(
                            policy_family="Diffusion Policy",
                            compatible=False,
                            explanation="Camera lag exceeds threshold.",
                            caveats=["Multi-modal conditioning will be corrupted."],
                        )
                    ],
                )
            ],
        )

    def test_flag_accessors(self):
        report = self._make_report()
        assert len(report.flags) == 2
        assert len(report.flags_at_level(RiskLevel.CRITICAL)) == 1
        assert len(report.flags_at_level(RiskLevel.WARNING)) == 1
        assert len(report.flags_at_level(RiskLevel.OK)) == 0

    def test_hint_accessor(self):
        report = self._make_report()
        assert len(report.hints) == 1
        assert report.hints[0].policy_family == "Diffusion Policy"

    def test_summary_contains_key_fields(self):
        report = self._make_report()
        s = report.summary()
        assert "test_ds" in s
        assert "CRITICAL" in s
        assert "WARNING" in s
        assert "Diffusion Policy" in s
        assert "1 critical" in s

    def test_serialises_to_json(self):
        report = self._make_report()
        json_str = report.model_dump_json()
        assert "camera_lag_std" in json_str
        assert "CRITICAL" in json_str

    def test_roundtrips_json(self):
        report = self._make_report()
        restored = DiagnosticReport.model_validate_json(report.model_dump_json())
        assert len(restored.flags) == len(report.flags)
        assert restored.flags[0].level == RiskLevel.CRITICAL
