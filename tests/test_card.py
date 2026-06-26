"""Tests for calibra.card — HuggingFace dataset quality card generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from calibra.card import generate_card, generate_yaml_frontmatter, _metric_row
from calibra.schema.report import (
    AnalyzerResult,
    DiagnosticReport,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_report(
    n_episodes: int = 50,
    n_samples: int = 25000,
    spike_rate: float = 0.01,
    vel_disc: float = 0.01,
    ldlj: float = -6.0,
    dropout: float = 0.0,
    jitter_cv: float = 0.001,
    action_entropy: float = 3.5,
    contact_fraction: float = 0.15,
    n_crit_flags: int = 0,
    n_warn_flags: int = 0,
) -> DiagnosticReport:
    flags = []
    for _ in range(n_crit_flags):
        flags.append(
            RiskFlag(
                level=RiskLevel.CRITICAL,
                metric="spike_rate",
                observed=ObservedValue(value=spike_rate),
                threshold=0.05,
                interpretation="high spike rate",
                implication="train carefully",
            )
        )
    for _ in range(n_warn_flags):
        flags.append(
            RiskFlag(
                level=RiskLevel.WARNING,
                metric="ldlj",
                observed=ObservedValue(value=ldlj),
                threshold=-10.0,
                interpretation="moderate jerk",
                implication="consider pruning",
            )
        )

    ar_smoothness = AnalyzerResult(
        analyzer_name="control_smoothness",
        flags=flags,
        raw_metrics={
            "ldlj": {"mean_ldlj": ldlj},
            "jerk_spikes": {"mean_spike_fraction": spike_rate},
            "vel_discontinuities": {"mean_disc_fraction": vel_disc},
        },
    )
    ar_temporal = AnalyzerResult(
        analyzer_name="temporal_stability",
        raw_metrics={
            "dropout": {"mean_dropout_fraction": dropout},
            "jitter": {"mean_cv": jitter_cv},
        },
    )
    ar_coverage = AnalyzerResult(
        analyzer_name="coverage_entropy",
        raw_metrics={
            "action_entropy": {"entropy_bits_per_dim": action_entropy},
        },
    )
    ar_phase = AnalyzerResult(
        analyzer_name="phase_balance",
        raw_metrics={"mean_contact_fraction": contact_fraction},
    )

    return DiagnosticReport(
        dataset_name="test_dataset",
        source_path="/data/test",
        format="hdf5",
        n_episodes=n_episodes,
        n_samples=n_samples,
        analyzer_results=[ar_smoothness, ar_temporal, ar_coverage, ar_phase],
    )


# ── _metric_row ───────────────────────────────────────────────────────────────


class TestMetricRow:
    def test_pass_higher_worse(self):
        row = _metric_row("Spike rate", 0.01, "fraction", warn=0.02, crit=0.05)
        assert "✅" in row
        assert "0.01" in row

    def test_warn_higher_worse(self):
        row = _metric_row("Spike rate", 0.03, "fraction", warn=0.02, crit=0.05)
        assert "⚠️" in row

    def test_crit_higher_worse(self):
        row = _metric_row("Spike rate", 0.08, "fraction", warn=0.02, crit=0.05)
        assert "❌" in row

    def test_pass_lower_worse(self):
        row = _metric_row("LDLJ", -5.0, "score", warn=-10.0, crit=-15.0, direction="lower_worse")
        assert "✅" in row

    def test_warn_lower_worse(self):
        row = _metric_row("LDLJ", -12.0, "score", warn=-10.0, crit=-15.0, direction="lower_worse")
        assert "⚠️" in row

    def test_crit_lower_worse(self):
        row = _metric_row("LDLJ", -18.0, "score", warn=-10.0, crit=-15.0, direction="lower_worse")
        assert "❌" in row

    def test_none_value_renders_dash(self):
        row = _metric_row("Missing", None, "fraction", warn=0.02, crit=0.05)
        assert "—" in row
        assert "✅" in row  # missing → pass


# ── generate_card ─────────────────────────────────────────────────────────────


class TestGenerateCard:
    def test_returns_string(self):
        report = _make_report()
        card, _ = generate_card(report)
        assert isinstance(card, str)
        assert len(card) > 100

    def test_contains_dataset_name(self):
        report = _make_report()
        card, _ = generate_card(report)
        assert "test_dataset" in card

    def test_certified_status_clean_data(self):
        report = _make_report(n_crit_flags=0, n_warn_flags=0)
        card, _ = generate_card(report)
        assert "CERTIFIED" in card
        assert "NOT CERTIFIED" not in card

    def test_provisional_status_warns(self):
        report = _make_report(n_crit_flags=0, n_warn_flags=1)
        card, _ = generate_card(report)
        assert "PROVISIONALLY CERTIFIED" in card

    def test_not_certified_status_crits(self):
        report = _make_report(n_crit_flags=1, n_warn_flags=0)
        card, _ = generate_card(report)
        assert "NOT CERTIFIED" in card

    def test_badge_url_in_card(self):
        report = _make_report()
        card, _ = generate_card(report)
        assert "img.shields.io" in card

    def test_metric_table_present(self):
        report = _make_report()
        card, _ = generate_card(report)
        assert "Jerk spike rate" in card
        assert "Velocity discontinuity" in card
        assert "LDLJ" in card
        assert "Action entropy" in card

    def test_predicted_success_present(self):
        report = _make_report()
        card, _ = generate_card(report)
        assert "Predicted success" in card
        assert "%" in card

    def test_policy_family_shown(self):
        report = _make_report()
        card, _ = generate_card(report, policy_family="act")
        assert "act" in card

    def test_deductions_section_for_bad_data(self):
        report = _make_report(spike_rate=0.08, n_crit_flags=1)
        card, _ = generate_card(report)
        assert "Quality Issues" in card
        assert "CRITICAL" in card

    def test_no_deductions_section_for_clean_data(self):
        report = _make_report(
            spike_rate=0.005,
            vel_disc=0.005,
            ldlj=-3.0,
            action_entropy=4.0,
            n_crit_flags=0,
            n_warn_flags=0,
        )
        card, _ = generate_card(report)
        assert "Quality Issues" not in card

    def test_calibra_link_in_footer(self):
        report = _make_report()
        card, _ = generate_card(report)
        assert "github.com/omerTT/Calibra" in card

    def test_version_shown(self):
        from calibra import __version__

        report = _make_report()
        card, _ = generate_card(report)
        assert __version__ in card

    def test_episode_count_shown(self):
        report = _make_report(n_episodes=120)
        card, _ = generate_card(report)
        assert "120" in card

    def test_coreset_recommendation_present(self):
        report = _make_report()
        card, _ = generate_card(report)
        assert "prune" in card.lower() or "coreset" in card.lower()


# ── generate_yaml_frontmatter ─────────────────────────────────────────────────


class TestGenerateYamlFrontmatter:
    def test_certified_true_no_crits(self):
        report = _make_report(n_crit_flags=0)
        yaml = generate_yaml_frontmatter(report)
        assert "calibra_certified: true" in yaml

    def test_certified_false_with_crits(self):
        report = _make_report(n_crit_flags=1)
        yaml = generate_yaml_frontmatter(report)
        assert "calibra_certified: false" in yaml

    def test_version_in_yaml(self):
        from calibra import __version__

        report = _make_report()
        yaml = generate_yaml_frontmatter(report)
        assert __version__ in yaml

    def test_episode_count_in_yaml(self):
        report = _make_report(n_episodes=77)
        yaml = generate_yaml_frontmatter(report)
        assert "77" in yaml


# ── run_card CLI ──────────────────────────────────────────────────────────────


class TestRunCardCLI:
    def _make_batch(self, n=20):
        from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

        rng = np.random.default_rng(0)
        episodes = [
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=np.linspace(0, 1.0, 50),
                observations={"proprio": rng.normal(0, 1, (50, 3))},
                actions=rng.normal(0, 0.3, (50, 3)),
            )
            for i in range(n)
        ]
        return EpisodeBatch(
            episodes=episodes,
            dataset_name="cli_test",
            format="synthetic",
            source_path="synthetic",
        )

    def _clean_report(self):
        """A report with no critical or warning flags."""
        return _make_report(
            spike_rate=0.005,
            vel_disc=0.005,
            ldlj=-4.0,
            dropout=0.0,
            jitter_cv=0.001,
            action_entropy=4.0,
            contact_fraction=0.20,
            n_crit_flags=0,
            n_warn_flags=0,
        )

    def test_run_card_prints_markdown(self, capsys, monkeypatch):
        from calibra.card import run_card

        clean = self._clean_report()
        with patch("calibra.card.Pipeline") as mock_pl:
            mock_pl.return_value.analyze_path.return_value = clean
            with pytest.raises(SystemExit):
                run_card(["synthetic_path"])
        out = capsys.readouterr().out
        assert "Dataset Quality Report" in out
        assert "Calibra" in out

    def test_run_card_writes_file(self, tmp_path, monkeypatch):
        from calibra.card import run_card

        clean = self._clean_report()
        out_file = tmp_path / "card.md"
        with patch("calibra.card.Pipeline") as mock_pl:
            mock_pl.return_value.analyze_path.return_value = clean
            with pytest.raises(SystemExit):
                run_card(["synthetic_path", "--out", str(out_file)])
        assert out_file.exists()
        assert "Dataset Quality Report" in out_file.read_text()

    def test_run_card_exit_0_clean(self, monkeypatch):
        from calibra.card import run_card

        clean = self._clean_report()
        with patch("calibra.card.Pipeline") as mock_pl:
            mock_pl.return_value.analyze_path.return_value = clean
            with pytest.raises(SystemExit) as exc:
                run_card(["synthetic_path"])
        assert exc.value.code == 0

    def test_run_card_exit_1_critical(self, monkeypatch):
        from calibra.card import run_card
        import calibra.ingestion.registry as registry
        from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

        rng = np.random.default_rng(1)
        # Extremely spiky data → critical flag
        episodes = [
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=np.linspace(0, 1.0, 50),
                observations={"proprio": rng.normal(0, 1, (50, 3))},
                actions=rng.normal(0, 10.0, (50, 3)),
            )
            for i in range(5)
        ]
        bad_batch = EpisodeBatch(
            episodes=episodes, dataset_name="bad", format="synthetic", source_path="s"
        )
        monkeypatch.setattr(registry, "load", lambda path, reader=None: bad_batch)

        # Patch the report to have a critical flag directly
        mock_report = _make_report(n_crit_flags=1)
        mock_pipeline = MagicMock()
        mock_pipeline.analyze_path.return_value = mock_report
        with patch("calibra.card.Pipeline", return_value=mock_pipeline):
            with pytest.raises(SystemExit) as exc:
                run_card(["bad_path"])
        assert exc.value.code == 1
