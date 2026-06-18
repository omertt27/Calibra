"""Tests for calibra score, predict, sim2real, and transfer commands."""
from __future__ import annotations

import numpy as np
import pytest

from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import DiagnosticReport, AnalyzerResult, RiskFlag, RiskLevel, ObservedValue
from calibra.pipeline import Pipeline


def _make_batch(
    n_eps: int = 15,
    n_steps: int = 100,
    action_dim: int = 7,
    dt: float = 0.05,
    has_camera: bool = True,
    has_lang: bool = True,
    action_std: float = 0.3,
) -> EpisodeBatch:
    rng = np.random.default_rng(1)
    episodes = []
    for i in range(n_eps):
        obs: dict = {}
        if has_camera:
            obs["camera_rgb"] = rng.random((n_steps, 32, 32, 3)).astype(np.float32)
        obs["proprio"] = rng.random((n_steps, action_dim)).astype(np.float32)
        actions = rng.normal(0, action_std, size=(n_steps, action_dim)).astype(np.float32)
        episodes.append(Episode(
            metadata=EpisodeMetadata(
                episode_id=f"ep_{i}",
                task_description="pick cube" if has_lang else None,
            ),
            timestamps=np.arange(n_steps, dtype=np.float64) * dt,
            observations=obs,
            actions=actions,
        ))
    return EpisodeBatch(
        episodes=episodes, dataset_name="test_ds",
        format="hdf5", source_path="/tmp/test.h5",
    )


# ── score tests ───────────────────────────────────────────────────────────────

class TestScore:
    def test_compute_score_returns_valid_range(self):
        from calibra.score import compute_score
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = compute_score(report)
        assert 0.0 <= result["total_score"] <= 100.0

    def test_score_category_valid(self):
        from calibra.score import compute_score, _category
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = compute_score(report)
        assert result["category"] in ("Excellent", "Good", "Fair", "Poor", "Critical")

    def test_score_dimensions_present(self):
        from calibra.score import compute_score
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = compute_score(report)
        dims = result["dimensions"]
        assert set(dims.keys()) == {
            "temporal_stability", "control_smoothness",
            "coverage_diversity", "task_structure",
        }
        for dim in dims.values():
            assert 0.0 <= dim["score"] <= dim["max"]

    def test_render_score_returns_string(self):
        from calibra.score import compute_score, render_score
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = compute_score(report)
        rendered = render_score(result)
        assert "CALIBRA SCORE" in rendered
        assert str(int(result["total_score"])) in rendered

    def test_render_badge_format(self):
        from calibra.score import compute_score, render_badge
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = compute_score(report)
        badge = render_badge(result)
        assert badge.startswith("![Calibra Score]")
        assert "img.shields.io" in badge

    def test_exit_code_logic(self):
        from calibra.score import _exit_code
        assert _exit_code(95.0) == 0
        assert _exit_code(75.0) == 0
        assert _exit_code(60.0) == 1
        assert _exit_code(39.0) == 2


# ── predict tests ─────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_outcome_range(self):
        from calibra.predict import predict_outcome
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = predict_outcome(report, policy_family="diffusion")
        assert 0.0 <= result["predicted_score"] <= 100.0
        lo, hi = result["predicted_range"]
        assert lo <= result["predicted_score"] <= hi

    def test_predict_tier_valid(self):
        from calibra.predict import predict_outcome
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = predict_outcome(report)
        assert result["tier"] in ("STRONG", "GOOD", "MARGINAL", "RISKY", "UNLIKELY")

    def test_predict_deductions_explained(self):
        from calibra.predict import predict_outcome
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = predict_outcome(report)
        for d in result["deductions"]:
            assert "metric" in d
            assert "penalty" in d
            assert "reason" in d
            assert d["penalty"] > 0

    def test_render_prediction_string(self):
        from calibra.predict import predict_outcome, render_prediction
        batch = _make_batch()
        report = Pipeline().run(batch)
        result = predict_outcome(report)
        rendered = render_prediction(result)
        assert "TRAINING OUTCOME PREDICTION" in rendered


# ── sim2real tests ────────────────────────────────────────────────────────────

class TestSim2Real:
    def test_gap_analysis_structure(self):
        from calibra.sim2real import analyze_gap
        sim_batch  = _make_batch(n_eps=10, action_std=0.1)
        real_batch = _make_batch(n_eps=10, action_std=0.5)
        sim_report  = Pipeline().run(sim_batch)
        real_report = Pipeline().run(real_batch)
        result = analyze_gap(sim_report, real_report, sim_batch, real_batch)
        assert "overall_risk" in result
        assert result["overall_risk"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert "gaps" in result

    def test_identical_datasets_low_risk(self):
        from calibra.sim2real import analyze_gap
        batch   = _make_batch()
        report  = Pipeline().run(batch)
        result  = analyze_gap(report, report, batch, batch)
        assert result["overall_risk"] in ("LOW", "MEDIUM")

    def test_kl_divergence_helper(self):
        from calibra.sim2real import _kl_divergence_marginal
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, (500, 3)).astype(np.float32)
        b = rng.normal(0, 1, (500, 3)).astype(np.float32)
        kl_same = _kl_divergence_marginal(a, b)
        b_far = rng.normal(5, 1, (500, 3)).astype(np.float32)
        kl_far = _kl_divergence_marginal(a, b_far)
        assert kl_far > kl_same

    def test_render_sim2real(self):
        from calibra.sim2real import analyze_gap, render_sim2real
        batch  = _make_batch()
        report = Pipeline().run(batch)
        result = analyze_gap(report, report)
        rendered = render_sim2real(result)
        assert "SIM-TO-REAL" in rendered


# ── transfer tests ────────────────────────────────────────────────────────────

class TestTransfer:
    def test_identical_datasets_direct(self):
        from calibra.transfer import analyze_transfer
        batch  = _make_batch()
        report = Pipeline().run(batch)
        result = analyze_transfer(report, report, batch, batch)
        assert result["overall_level"] in ("DIRECT", "ADAPT")

    def test_dim_mismatch_detected(self):
        from calibra.transfer import _action_dim_compatibility
        level, _ = _action_dim_compatibility(7, 14)
        assert level == "DIFFICULT"
        level2, _ = _action_dim_compatibility(14, 7)
        assert level2 == "ADAPT"
        level3, _ = _action_dim_compatibility(7, 7)
        assert level3 == "DIRECT"

    def test_frequency_mismatch(self):
        from calibra.transfer import _frequency_compatibility
        level, _ = _frequency_compatibility(50.0, 10.0)
        # ratio = 5x → should be DIFFICULT
        assert level in ("DIFFICULT", "ADAPT")
        level2, _ = _frequency_compatibility(50.0, 50.0)
        assert level2 == "DIRECT"

    def test_result_has_required_keys(self):
        from calibra.transfer import analyze_transfer
        batch  = _make_batch()
        report = Pipeline().run(batch)
        result = analyze_transfer(report, report, batch, batch)
        assert "overall_level" in result
        assert "dimensions" in result
        assert result["overall_level"] in ("DIRECT", "ADAPT", "DIFFICULT", "INCOMPATIBLE")

    def test_render_transfer(self):
        from calibra.transfer import analyze_transfer, render_transfer
        batch  = _make_batch()
        report = Pipeline().run(batch)
        result = analyze_transfer(report, report, batch, batch)
        rendered = render_transfer(result)
        assert "CROSS-EMBODIMENT" in rendered
