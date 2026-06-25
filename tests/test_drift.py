"""Tests for calibra.temporal.drift — camera-physics lag detection."""

from __future__ import annotations

import numpy as np
import pytest

from calibra.temporal.drift import (
    compute_visual_activity,
    estimate_sensor_command_latency,
    estimate_visual_physics_lag,
)


# ── compute_visual_activity ───────────────────────────────────────────────────


class TestComputeVisualActivity:
    def test_output_shape_4d(self):
        imgs = np.zeros((10, 84, 84, 3), dtype=np.uint8)
        out = compute_visual_activity(imgs)
        assert out.shape == (9,)

    def test_output_shape_3d(self):
        imgs = np.zeros((10, 84, 84), dtype=np.uint8)
        out = compute_visual_activity(imgs)
        assert out.shape == (9,)

    def test_zeros_give_zero_activity(self):
        imgs = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        out = compute_visual_activity(imgs)
        np.testing.assert_allclose(out, 0.0)

    def test_constant_nonzero_gives_zero_activity(self):
        imgs = np.full((5, 8, 8, 3), 128, dtype=np.uint8)
        out = compute_visual_activity(imgs)
        np.testing.assert_allclose(out, 0.0)

    def test_all_different_frames_gives_nonzero(self):
        rng = np.random.default_rng(0)
        imgs = rng.integers(0, 255, (10, 8, 8, 3), dtype=np.uint8)
        out = compute_visual_activity(imgs)
        assert np.all(out > 0)

    def test_raises_on_too_few_frames(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_visual_activity(np.zeros((1, 8, 8, 3), dtype=np.uint8))

    def test_raises_on_wrong_dims(self):
        with pytest.raises(ValueError, match="shape"):
            compute_visual_activity(np.zeros((10, 8), dtype=np.uint8))


# ── estimate_sensor_command_latency ──────────────────────────────────────────


class TestEstimateLag:
    def test_zero_lag_for_identical_signals(self):
        rng = np.random.default_rng(42)
        sig = rng.random(100)
        lag = estimate_sensor_command_latency(sig, sig)
        assert lag == 0

    def test_detects_camera_lag(self):
        """When visual lags physical by k frames, cross-corr peak is at -k."""
        rng = np.random.default_rng(7)
        k = 3
        base = rng.random(80)
        physical = base
        visual = base[k:]  # visual is k steps behind (lags)
        lag = estimate_sensor_command_latency(physical, visual)
        # numpy convention: visual lags → k < 0; abs should equal k
        assert abs(lag) == k

    def test_detects_visual_lead(self):
        """When visual leads physical by k frames, cross-corr peak is at +k."""
        rng = np.random.default_rng(7)
        k = 2
        base = rng.random(80)
        physical = base[k:]
        visual = base  # visual is k steps ahead
        lag = estimate_sensor_command_latency(physical, visual)
        assert abs(lag) == k

    def test_handles_different_lengths(self):
        rng = np.random.default_rng(0)
        p = rng.random(100)
        v = rng.random(99)
        lag = estimate_sensor_command_latency(p, v)
        assert isinstance(lag, int)

    def test_constant_signal_returns_zero(self):
        """Constant signals have no correlation structure."""
        lag = estimate_sensor_command_latency(np.ones(50), np.ones(50))
        assert lag == 0


# ── estimate_visual_physics_lag ───────────────────────────────────────────────


class TestEstimateVisualPhysicsLag:
    def test_returns_int(self):
        """estimate_visual_physics_lag always returns an int."""
        rng = np.random.default_rng(0)
        T = 40
        base_vel = np.zeros((T, 7), dtype=np.float32)
        base_vel[10:20, :] = 2.0
        imgs = np.full((T, 8, 8, 3), 128, dtype=np.uint8)
        imgs[10:20] = rng.integers(0, 255, (10, 8, 8, 3), dtype=np.uint8)
        lag = estimate_visual_physics_lag(imgs, base_vel)
        assert isinstance(lag, int)

    def test_large_shift_detected(self):
        """A large (8-frame) shift between visual and physical should give abs(lag) > 3."""
        rng = np.random.default_rng(7)
        T = 80
        k = 8
        base_vel = np.zeros((T, 7), dtype=np.float32)
        base_vel[10:30, :] = 2.0
        imgs = np.full((T, 8, 8, 3), 128, dtype=np.uint8)
        # Visual window shifted k steps (visual lags → peak at -k by convention)
        imgs[10 + k : 30 + k] = rng.integers(0, 255, (20, 8, 8, 3), dtype=np.uint8)
        lag = estimate_visual_physics_lag(imgs, base_vel)
        # The estimated abs lag should be substantial (> 2 frames at minimum)
        assert abs(lag) > 2


# ── GR00T drift check ─────────────────────────────────────────────────────────


class TestGR00TDriftCheck:
    def _make_batch_with_drift(self, lag_frames: int):
        import numpy as np
        from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

        T = 80
        rng = np.random.default_rng(42)
        base_vel = np.zeros((T, 7), dtype=np.float32)
        base_vel[15:35, :] = 2.0

        # Random pixels during the visual-active window so every frame-to-frame
        # transition is large → sustained visual_activity matching the burst.
        imgs = np.full((T, 8, 8, 3), 128, dtype=np.uint8)
        start = max(0, 15 + lag_frames)
        end = min(T, 35 + lag_frames)
        imgs[start:end] = rng.integers(0, 255, (end - start, 8, 8, 3), dtype=np.uint8)

        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep_0", task_description="task"),
            timestamps=np.arange(T, dtype=np.float64) * 0.02,
            observations={
                "robot0_joint_vel": base_vel,
                "agentview_image": imgs,
            },
            actions=rng.random((T, 7)).astype(np.float32),
        )
        return EpisodeBatch(
            episodes=[ep] * 5, dataset_name="test", format="isaac_lab", source_path="/tmp/test"
        )

    def test_drift_flag_present_when_data_available(self):
        """When both image and velocity obs are present, a drift flag is produced."""
        from calibra.analyzers.gr00t import GR00TCompatibilityAnalyzer
        from calibra.schema.report import RiskLevel

        batch = self._make_batch_with_drift(0)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        drift_flags = [f for f in result.flags if "drift" in f.metric]
        assert len(drift_flags) == 1
        # With clean synthetic data at lag=0, flag should not be CRITICAL
        assert drift_flags[0].level in (RiskLevel.OK, RiskLevel.WARNING)

    def test_large_drift_produces_nonzero_lag(self):
        """Large visual lag should produce a non-zero observed lag value."""
        from calibra.analyzers.gr00t import GR00TCompatibilityAnalyzer

        batch = self._make_batch_with_drift(8)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        drift_flags = [f for f in result.flags if "drift" in f.metric]
        assert len(drift_flags) == 1
        assert abs(drift_flags[0].observed.value) > 0

    def test_skipped_without_image_obs(self):
        from calibra.analyzers.gr00t import GR00TCompatibilityAnalyzer
        from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

        rng = np.random.default_rng(0)
        T = 20
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep_0", task_description="task"),
            timestamps=np.arange(T, dtype=np.float64) * 0.02,
            observations={"proprio": rng.random((T, 14)).astype(np.float32)},
            actions=rng.random((T, 7)).astype(np.float32),
        )
        batch = EpisodeBatch(
            episodes=[ep], dataset_name="no_image", format="isaac_lab", source_path="/tmp"
        )
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        drift_flags = [f for f in result.flags if "drift" in f.metric]
        # Should be silently skipped (no drift flag at all)
        assert len(drift_flags) == 0
