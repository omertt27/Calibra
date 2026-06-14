"""Tests for the temporal stability analyzer."""
from __future__ import annotations

import numpy as np
import pytest

from calibra.analyzers.temporal import (
    TemporalAnalyzer,
    _bootstrap_ci,
    _episode_camera_lag_std,
    _episode_dropout_fraction,
    _episode_jitter_cv,
    _episode_misalignment_fraction,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel


# ── unit tests: per-episode helpers ─────────────────────────────────────────

class TestEpisodeJitterCV:
    def test_uniform_timing_returns_zero(self):
        ep = _uniform_episode(100, 0.1)
        cv = _episode_jitter_cv(ep)
        assert cv is not None
        assert cv == pytest.approx(0.0, abs=1e-9)

    def test_high_jitter_returns_high_cv(self):
        rng = np.random.default_rng(1)
        ts = np.cumsum(np.abs(rng.normal(0.1, 0.05, 100)))
        ep = _ep_with_timestamps(ts)
        cv = _episode_jitter_cv(ep)
        assert cv is not None
        assert cv > 0.1

    def test_too_short_returns_none(self):
        ep = _uniform_episode(2, 0.1)
        assert _episode_jitter_cv(ep) is None


class TestEpisodeDropoutFraction:
    def test_no_dropout(self):
        ep = _uniform_episode(100, 0.1)
        assert _episode_dropout_fraction(ep, k=3.0) == pytest.approx(0.0)

    def test_known_dropout_fraction(self):
        # Insert a 5× gap at every 10th step → 10 out of 99 deltas
        ts = np.arange(100, dtype=float) * 0.1
        ts[10] += 0.4   # create one large gap
        ts[20] += 0.4
        ep = _ep_with_timestamps(ts)
        frac = _episode_dropout_fraction(ep, k=3.0)
        assert frac == pytest.approx(2 / 99, rel=0.05)

    def test_too_short_returns_zero(self):
        ep = _uniform_episode(2, 0.1)
        assert _episode_dropout_fraction(ep) == 0.0


class TestEpisodeCameraLagStd:
    def test_no_per_modality_ts_returns_none(self):
        ep = _uniform_episode(50, 0.1)
        assert _episode_camera_lag_std(ep, "camera_rgb") is None

    def test_zero_lag(self):
        ep = _uniform_episode(50, 0.1)
        ep.obs_timestamps["camera_rgb"] = ep.timestamps.copy()
        assert _episode_camera_lag_std(ep, "camera_rgb") == pytest.approx(0.0)

    def test_known_lag_std(self):
        rng = np.random.default_rng(0)
        ep = _uniform_episode(200, 0.1)
        ep.obs_timestamps["camera_rgb"] = ep.timestamps + rng.normal(0, 0.025, 200)
        lag_std = _episode_camera_lag_std(ep, "camera_rgb")
        assert lag_std is not None
        assert 0.018 < lag_std < 0.032   # ≈25ms ± tolerance


class TestEpisodeMisalignment:
    def test_no_action_timestamps_returns_zero(self):
        ep = _uniform_episode(50, 0.1)
        assert _episode_misalignment_fraction(ep, tol_s=0.005) == 0.0

    def test_constant_offset_above_tol(self):
        ep = _uniform_episode(100, 0.1)
        ep.action_timestamps = ep.timestamps + 0.010  # 10ms > 5ms tol
        frac = _episode_misalignment_fraction(ep, tol_s=0.005)
        assert frac == pytest.approx(1.0)

    def test_constant_offset_below_tol(self):
        ep = _uniform_episode(100, 0.1)
        ep.action_timestamps = ep.timestamps + 0.002  # 2ms < 5ms tol
        frac = _episode_misalignment_fraction(ep, tol_s=0.005)
        assert frac == pytest.approx(0.0)


class TestBootstrapCI:
    def test_single_value_returns_same_bounds(self):
        arr = np.array([5.0])
        stat, lo, hi = _bootstrap_ci(arr, np.mean, n_boot=100)
        assert stat == pytest.approx(5.0)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_ci_bounds_bracket_point_estimate(self):
        # Bootstrap CIs always bracket the point estimate by construction.
        rng = np.random.default_rng(7)
        arr = rng.normal(3.0, 0.5, 30)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, n_boot=2000, ci_level=0.95)
        assert lo <= stat <= hi

    def test_ci_narrows_with_more_samples(self):
        rng = np.random.default_rng(7)
        small = rng.normal(3.0, 0.5, 10)
        large = rng.normal(3.0, 0.5, 200)
        _, lo_s, hi_s = _bootstrap_ci(small, np.mean, n_boot=1000)
        _, lo_l, hi_l = _bootstrap_ci(large, np.mean, n_boot=1000)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_ci_ordering(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stat, lo, hi = _bootstrap_ci(arr, np.mean)
        assert lo <= stat <= hi


# ── integration tests: TemporalAnalyzer end-to-end ──────────────────────────

class TestTemporalAnalyzerClean:
    def test_clean_batch_all_ok(self, clean_batch):
        result = TemporalAnalyzer().analyze(clean_batch)
        non_ok = [f for f in result.flags if f.level != RiskLevel.OK]
        assert non_ok == [], f"Unexpected flags on clean data: {non_ok}"

    def test_returns_correct_analyzer_name(self, clean_batch):
        result = TemporalAnalyzer().analyze(clean_batch)
        assert result.analyzer_name == "temporal_stability"

    def test_raw_metrics_populated(self, clean_batch):
        result = TemporalAnalyzer().analyze(clean_batch)
        assert "jitter" in result.raw_metrics
        assert "dropout" in result.raw_metrics


class TestTemporalAnalyzerJitter:
    def test_jittery_batch_raises_warning_or_critical(self, jittery_batch):
        result = TemporalAnalyzer().analyze(jittery_batch)
        jitter_flags = [f for f in result.flags if "jitter" in f.metric]
        assert jitter_flags, "Expected at least one jitter flag"
        assert jitter_flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)

    def test_ci_lower_bound_is_positive(self, jittery_batch):
        result = TemporalAnalyzer().analyze(jittery_batch)
        jitter_flag = next(f for f in result.flags if "jitter" in f.metric)
        assert jitter_flag.observed.ci_lower is not None
        assert jitter_flag.observed.ci_lower >= 0.0


class TestTemporalAnalyzerDropout:
    def test_dropout_batch_flags_dropout(self, dropout_batch):
        result = TemporalAnalyzer().analyze(dropout_batch)
        dropout_flags = [f for f in result.flags if "dropout" in f.metric]
        assert dropout_flags
        assert dropout_flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)

    def test_affected_fraction_nonzero(self, dropout_batch):
        result = TemporalAnalyzer().analyze(dropout_batch)
        dropout_flag = next(f for f in result.flags if "dropout" in f.metric)
        assert dropout_flag.affected_fraction is not None
        assert dropout_flag.affected_fraction > 0.0


class TestTemporalAnalyzerCameraLag:
    def test_cam_lag_batch_flags_lag(self, cam_lag_batch):
        result = TemporalAnalyzer().analyze(cam_lag_batch)
        cam_flags = [f for f in result.flags if "camera_lag" in f.metric]
        assert cam_flags, "Expected at least one camera_lag flag"
        # Fixture uses 50ms std, threshold is 20ms — should be CRITICAL
        assert cam_flags[0].level == RiskLevel.CRITICAL, (
            f"Expected CRITICAL, got {cam_flags[0].level}; "
            f"observed={cam_flags[0].observed}"
        )

    def test_observed_unit_is_ms(self, cam_lag_batch):
        result = TemporalAnalyzer().analyze(cam_lag_batch)
        cam_flag = next(f for f in result.flags if "camera_lag" in f.metric)
        assert cam_flag.observed.unit == "ms"
        assert cam_flag.observed.value > 10   # should be ~25ms


class TestTemporalAnalyzerMisalignment:
    def test_misaligned_batch_flags_alignment(self, misaligned_batch):
        result = TemporalAnalyzer().analyze(misaligned_batch)
        align_flags = [f for f in result.flags if "misalignment" in f.metric]
        assert align_flags
        assert align_flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)


class TestTemporalAnalyzerPolicyHints:
    def test_no_hints_without_policy_family(self, cam_lag_batch):
        result = TemporalAnalyzer().analyze(cam_lag_batch, policy_family=None)
        assert result.hints == []

    def test_diffusion_hint_incompatible_with_cam_lag(self, cam_lag_batch):
        result = TemporalAnalyzer().analyze(cam_lag_batch, policy_family="diffusion")
        dp_hints = [h for h in result.hints if "Diffusion" in h.policy_family]
        assert dp_hints
        assert dp_hints[0].compatible is False

    def test_diffusion_hint_compatible_on_clean_data(self, clean_batch):
        result = TemporalAnalyzer().analyze(clean_batch, policy_family="diffusion")
        dp_hints = [h for h in result.hints if "Diffusion" in h.policy_family]
        assert dp_hints
        assert dp_hints[0].compatible is True


class TestTemporalAnalyzerEdgeCases:
    def test_empty_batch(self):
        empty = EpisodeBatch(episodes=[], dataset_name="empty",
                             format="hdf5", source_path="/tmp/x.h5")
        result = TemporalAnalyzer().analyze(empty)
        assert result.flags == []
        assert result.hints == []

    def test_single_episode_no_crash(self):
        ep = _uniform_episode(50, 0.1)
        batch = EpisodeBatch(episodes=[ep], dataset_name="single",
                             format="hdf5", source_path="/tmp/x.h5")
        result = TemporalAnalyzer().analyze(batch)
        assert result.analyzer_name == "temporal_stability"

    def test_very_short_episodes_handled(self):
        eps = [_uniform_episode(2, 0.1, ep_id=str(i)) for i in range(5)]
        batch = EpisodeBatch(episodes=eps, dataset_name="short",
                             format="hdf5", source_path="/tmp/x.h5")
        result = TemporalAnalyzer().analyze(batch)
        assert result is not None


# ── helpers ──────────────────────────────────────────────────────────────────

def _uniform_episode(n: int, dt: float, ep_id: str = "ep_0") -> Episode:
    ts = np.arange(n, dtype=np.float64) * dt
    rng = np.random.default_rng(0)
    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=ts,
        observations={"camera_rgb": rng.random((n, 4, 4, 3)).astype(np.float32)},
        actions=rng.random((n, 7)).astype(np.float32),
    )


def _ep_with_timestamps(ts: np.ndarray) -> Episode:
    n = len(ts)
    rng = np.random.default_rng(0)
    return Episode(
        metadata=EpisodeMetadata(episode_id="ep_0"),
        timestamps=ts,
        observations={"proprio": rng.random((n, 7)).astype(np.float32)},
        actions=rng.random((n, 7)).astype(np.float32),
    )
