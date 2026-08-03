"""Tests for the `calibra integrity` CLI (calibra/integrity.py)."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from calibra.integrity import _integrity_flags, _integrity_score, run_integrity
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import (
    AnalyzerResult,
    DiagnosticReport,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)

# ── metric-whitelist filtering (unit-level, hand-built fixtures) ────────────


def _flag(metric: str, level: RiskLevel) -> RiskFlag:
    return RiskFlag(
        level=level,
        metric=metric,
        observed=ObservedValue(value=0.1),
        interpretation="x",
        implication="y",
    )


class TestIntegrityFiltering:
    def test_only_whitelisted_metrics_pass(self):
        report = DiagnosticReport(
            dataset_name="d",
            source_path="/p",
            format="hdf5",
            n_episodes=1,
            n_samples=1,
            analyzer_results=[
                AnalyzerResult(
                    analyzer_name="task_structure",
                    flags=[
                        _flag("short_episode_fraction", RiskLevel.OK),
                        _flag("trajectory_diversity", RiskLevel.WARNING),
                        _flag("contact_density", RiskLevel.INFO),
                    ],
                ),
                AnalyzerResult(
                    analyzer_name="temporal_stability",
                    flags=[
                        _flag("timestamp_jitter_cv", RiskLevel.OK),
                        _flag("camera_lag_std[camera_rgb]", RiskLevel.WARNING),
                    ],
                ),
                AnalyzerResult(
                    analyzer_name="control_smoothness",
                    flags=[
                        _flag("ldlj", RiskLevel.CRITICAL),
                        _flag("jerk_spike_rate", RiskLevel.OK),
                        _flag("velocity_discontinuity_rate", RiskLevel.WARNING),
                        _flag("action_state_divergence", RiskLevel.WARNING),
                        _flag("motion_collection_signature", RiskLevel.INFO),
                    ],
                ),
            ],
        )
        flags = _integrity_flags(report)
        metrics = {f.metric for f in flags}
        assert metrics == {
            "short_episode_fraction",
            "timestamp_jitter_cv",
            "camera_lag_std[camera_rgb]",
            "ldlj",
            "jerk_spike_rate",
            "velocity_discontinuity_rate",
        }
        assert "trajectory_diversity" not in metrics
        assert "contact_density" not in metrics
        # action_state_divergence (tracking error) and motion_collection_signature
        # (scripted-vs-teleop signature) are Quality-layer questions, not "is the
        # recorded motion itself jittery" — deliberately excluded from Integrity.
        assert "action_state_divergence" not in metrics
        assert "motion_collection_signature" not in metrics

    def test_score_all_ok_is_100(self):
        flags = [_flag("timestamp_jitter_cv", RiskLevel.OK)]
        score, status = _integrity_score(flags)
        assert score == 100
        assert status == "Healthy"

    def test_score_critical_is_zero(self):
        flags = [_flag("timestamp_jitter_cv", RiskLevel.CRITICAL)]
        score, status = _integrity_score(flags)
        assert score == 0
        assert status == "Critical"

    def test_score_no_flags_defaults_healthy(self):
        score, status = _integrity_score([])
        assert score == 100
        assert status == "Healthy"


# ── CLI ──────────────────────────────────────────────────────────────────────


def _smooth_actions(rng: np.random.Generator, steps: int, dt: float, dims: int = 6) -> np.ndarray:
    """Low-frequency sinusoid trajectory — smooth by construction (bounded
    jerk), unlike i.i.d. per-step noise which trips ControlSmoothnessAnalyzer's
    jerk/velocity-discontinuity checks (now part of the integrity whitelist)
    regardless of episode length."""
    t = np.arange(steps) * dt
    actions = np.zeros((steps, dims), dtype=np.float32)
    for d in range(dims):
        for h in range(1, 3):
            freq = h * rng.uniform(0.05, 0.12)
            amp = rng.uniform(0.05, 0.15) / h
            phase = rng.uniform(0, 2 * np.pi)
            actions[:, d] += amp * np.sin(2 * np.pi * freq * t + phase)
        actions[:, d] += rng.uniform(-0.3, 0.3)
    return actions.astype(np.float32)


def _make_batch(n_eps: int = 20, n_short: int = 4, n_steps: int = 80) -> EpisodeBatch:
    """20 episodes, 4 of them (20%) far shorter — pushes short_episode_fraction
    past the 15% CRITICAL threshold in TaskStructureAnalyzer."""
    rng = np.random.default_rng(0)
    episodes = []
    for i in range(n_eps):
        steps = 5 if i < n_short else n_steps
        ts = np.arange(steps, dtype=np.float64) * 0.05
        obs = {"proprio": rng.uniform(-1, 1, (steps, 8)).astype(np.float32)}
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=ts,
                observations=obs,
                actions=_smooth_actions(rng, steps, dt=0.05),
            )
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="integrity_test",
        format="hdf5",
        source_path="/dummy/path.h5",
    )


class TestRunIntegrity:
    def test_text_output_reports_critical_and_exits_1(self, capsys):
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5"])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Dataset Integrity" in out
        assert "short_episode_fraction" in out
        assert "Integrity Score" in out

    def test_json_output_structure(self, capsys):
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json"])
        assert exc_info.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_episodes"] == 20
        assert any(f["metric"] == "short_episode_fraction" for f in payload["critical"])
        assert "integrity_score" in payload
        assert "status" in payload

    def test_clean_dataset_exits_0(self, capsys):
        batch = _make_batch(n_short=0)
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5"])
        assert exc_info.value.code == 0

    def _jittery_batch(self) -> EpisodeBatch:
        rng = np.random.default_rng(0)
        n_eps, n_steps = 8, 80
        episodes = []
        for i in range(n_eps):
            ts = np.arange(n_steps, dtype=np.float64) * 0.05
            episodes.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                    timestamps=ts,
                    observations={"proprio": rng.uniform(-1, 1, (n_steps, 8)).astype(np.float32)},
                    actions=rng.uniform(-1, 1, (n_steps, 6)).astype(np.float32),
                )
            )
        return EpisodeBatch(
            episodes=episodes, dataset_name="jittery", format="hdf5", source_path="/dummy/path.h5"
        )

    def test_jittery_actions_flagged_critical(self, capsys):
        """i.i.d.-noise actions (velocity reverses every step) must trip the
        smoothness checks — the regression this fixture rewrite guards against."""
        batch = self._jittery_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json"])
        payload = json.loads(capsys.readouterr().out)
        flagged = {f["metric"] for f in payload["critical"] + payload["warnings"]}
        assert flagged & {"ldlj", "jerk_spike_rate", "velocity_discontinuity_rate"}
        # Motion-review CRITICALs are a review signal, not an objective acquisition
        # failure — they must not fail CI by default (this is the exact "6/6 exit 1
        # but 5/6 Warning" confusion reported by an HF user probing v0.7.1).
        assert payload["ci_result"] == "Passed"
        assert exc_info.value.code == 0
        for f in payload["critical"]:
            assert f["suggested_action"] == "inspect"

    def test_jittery_actions_fail_ci_with_strict(self, capsys):
        batch = self._jittery_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json", "--strict"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["ci_result"] == "Failed"
        assert "--strict" in payload["ci_reason"]
        assert exc_info.value.code == 1

    def test_block_level_critical_fails_ci_without_strict(self, capsys):
        """short_episode_fraction is an objective completeness failure — should
        still fail CI by default, unlike the motion-review case above."""
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["ci_result"] == "Failed"
        assert "short_episode_fraction" in payload["ci_reason"]
        assert exc_info.value.code == 1

    def test_not_evaluated_lists_skipped_image_checks(self, capsys):
        """No image data on this batch -> duplicate_frame/camera_freeze/blur
        analyzers are skipped by capability gating; they should be surfaced as
        'Not Evaluated', not silently absent."""
        batch = _make_batch(n_short=0)
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json"])
        assert exc_info.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        checks = {item["check"] for item in payload["not_evaluated"]}
        assert {"duplicate_frame", "camera_freeze", "blur"} <= checks
        for item in payload["not_evaluated"]:
            assert item["reason"] == "requires: images"

    # ── --policy ─────────────────────────────────────────────────────────

    def _write_policy(self, tmp_path, mapping: dict) -> str:
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(mapping))
        return str(path)

    def test_policy_downgrades_block_metric_to_inspect(self, tmp_path, capsys):
        """short_episode_fraction is block-by-default; a policy explicitly
        marking it 'inspect' should flip ci_result to Passed and exit 0."""
        policy_path = self._write_policy(tmp_path, {"short_episode_fraction": "inspect"})
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json", "--policy", policy_path])
        payload = json.loads(capsys.readouterr().out)
        assert payload["ci_result"] == "Passed"
        assert payload["policy_path"] == policy_path
        assert exc_info.value.code == 0
        crit = next(f for f in payload["critical"] if f["metric"] == "short_episode_fraction")
        assert crit["suggested_action"] == "inspect"

    def test_policy_upgrades_motion_review_metric_to_block(self, tmp_path, capsys):
        """ldlj is inspect-by-default (motion-review); a policy explicitly
        marking it 'block' should flip ci_result to Failed and exit 1."""
        policy_path = self._write_policy(tmp_path, {"ldlj": "block"})
        batch = self._jittery_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json", "--policy", policy_path])
        payload = json.loads(capsys.readouterr().out)
        assert payload["ci_result"] == "Failed"
        assert "ldlj" in payload["ci_reason"]
        assert exc_info.value.code == 1

    def test_policy_unknown_metric_warns_not_crashes(self, tmp_path, capsys):
        policy_path = self._write_policy(tmp_path, {"totally_made_up_metric": "block"})
        batch = _make_batch(n_short=0)
        with patch("calibra.ingestion.registry.load", return_value=batch):
            with pytest.raises(SystemExit) as exc_info:
                run_integrity(["/dummy/path.h5", "--json", "--policy", policy_path])
        assert exc_info.value.code == 0
        err = capsys.readouterr().err
        assert "unknown metric" in err
        assert "totally_made_up_metric" in err

    def test_strict_and_policy_are_mutually_exclusive(self, tmp_path, capsys):
        policy_path = self._write_policy(tmp_path, {"ldlj": "block"})
        with pytest.raises(SystemExit) as exc_info:
            run_integrity(["/dummy/path.h5", "--strict", "--policy", policy_path])
        assert exc_info.value.code == 2
        assert "mutually exclusive" in capsys.readouterr().err

    def test_invalid_policy_file_exits_2(self, tmp_path, capsys):
        policy_path = self._write_policy(tmp_path, {"ldlj": "delete"})
        with pytest.raises(SystemExit) as exc_info:
            run_integrity(["/dummy/path.h5", "--policy", policy_path])
        assert exc_info.value.code == 2
        assert "error loading policy file" in capsys.readouterr().err
