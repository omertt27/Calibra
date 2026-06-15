"""Tests for the control smoothness analyzer."""
from __future__ import annotations

import numpy as np

from calibra.analyzers.smoothness import (
    ControlSmoothnessAnalyzer,
    _episode_jerk_spike_fraction,
    _episode_ldlj,
    _episode_vel_disc_fraction,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel


# ── helpers ──────────────────────────────────────────────────────────────────

def _sine_episode(
    n_steps: int = 100,
    dt: float = 0.05,
    freq: float = 0.5,
    amplitude: float = 1.0,
    action_dim: int = 3,
    ep_id: str = "ep_0",
) -> Episode:
    """Smooth sinusoidal action trajectory (position)."""
    t = np.arange(n_steps) * dt
    ts = t.copy()
    actions = np.column_stack([
        amplitude * np.sin(2 * np.pi * freq * t + d * 0.3)
        for d in range(action_dim)
    ]).astype(np.float32)
    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=ts,
        observations={"proprio": np.zeros((n_steps, 4), dtype=np.float32)},
        actions=actions,
    )


def _jerky_episode(
    n_steps: int = 100,
    dt: float = 0.05,
    n_spikes: int = 5,
    action_dim: int = 3,
    ep_id: str = "ep_0",
) -> Episode:
    """Smooth trajectory with random jerk spikes injected."""
    rng = np.random.default_rng(0)
    t = np.arange(n_steps) * dt
    actions = np.sin(t[:, None] * 2 * np.pi * 0.5).repeat(action_dim, axis=1).astype(np.float32)
    for _ in range(n_spikes):
        idx = rng.integers(5, n_steps - 5)
        actions[idx] += rng.uniform(3, 8, size=action_dim).astype(np.float32)
    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=t,
        observations={"proprio": np.zeros((n_steps, 4), dtype=np.float32)},
        actions=actions,
    )


def _batch_of(episodes) -> EpisodeBatch:
    return EpisodeBatch(episodes=list(episodes), dataset_name="test",
                        format="hdf5", source_path="/tmp/x.h5")


# ── unit tests: LDLJ ─────────────────────────────────────────────────────────

class TestLDLJ:
    def test_smooth_sine_is_less_negative_than_jerky(self):
        smooth = _sine_episode(200, dt=0.05)
        jerky  = _jerky_episode(200, dt=0.05, n_spikes=10)
        ldlj_smooth = _episode_ldlj(smooth, "position", list(range(3)))
        ldlj_jerky  = _episode_ldlj(jerky,  "position", list(range(3)))
        assert ldlj_smooth is not None and ldlj_jerky is not None
        assert ldlj_smooth > ldlj_jerky  # less negative = smoother

    def test_too_short_returns_none(self):
        ep = _sine_episode(n_steps=4)
        assert _episode_ldlj(ep, "position", [0, 1, 2]) is None

    def test_constant_action_returns_none(self):
        t = np.arange(50) * 0.05
        acts = np.ones((50, 3), dtype=np.float32)
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep"),
            timestamps=t, observations={},
            actions=acts,
        )
        assert _episode_ldlj(ep, "position", [0, 1, 2]) is None

    def test_velocity_action_type(self):
        ep = _sine_episode(100, dt=0.05)
        # treat actions as velocities → fewer derivatives
        ldlj = _episode_ldlj(ep, "velocity", [0, 1, 2])
        assert ldlj is not None
        assert ldlj < 0  # always negative for real movements

    def test_ldlj_is_always_negative(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            t = np.arange(100) * 0.05
            acts = rng.random((100, 4)).astype(np.float32)
            ep = Episode(
                metadata=EpisodeMetadata(episode_id="ep"),
                timestamps=t, observations={}, actions=acts,
            )
            v = _episode_ldlj(ep, "position", [0, 1, 2, 3])
            if v is not None:
                assert v < 0


# ── unit tests: jerk spike fraction ──────────────────────────────────────────

class TestJerkSpikeFraction:
    def test_smooth_has_low_spike_fraction(self):
        ep = _sine_episode(200, dt=0.05)
        frac = _episode_jerk_spike_fraction(ep, "position", [0, 1, 2], k=5.0)
        assert frac is not None
        assert frac < 0.05

    def test_injected_spikes_detected(self):
        ep = _jerky_episode(200, dt=0.05, n_spikes=10)
        frac = _episode_jerk_spike_fraction(ep, "position", [0, 1, 2], k=5.0)
        assert frac is not None
        assert frac > 0.01  # at least some spikes detected

    def test_too_short_returns_none(self):
        ep = _sine_episode(n_steps=4)
        assert _episode_jerk_spike_fraction(ep, "position", [0, 1, 2]) is None


# ── unit tests: velocity discontinuity fraction ───────────────────────────────

class TestVelDiscFraction:
    def test_smooth_no_discontinuities(self):
        ep = _sine_episode(200, dt=0.05, freq=0.1)  # slow oscillation
        frac = _episode_vel_disc_fraction(ep, "position", [0, 1, 2], threshold=0.5)
        assert frac is not None
        assert frac < 0.05

    def test_step_function_all_discontinuities(self):
        t = np.arange(50) * 0.05
        # Alternating +1/-1 → velocity is discontinuous at every step
        acts = np.column_stack([np.sign(np.sin(np.pi * np.arange(50)))] * 3).astype(np.float32)
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep"),
            timestamps=t, observations={}, actions=acts,
        )
        frac = _episode_vel_disc_fraction(ep, "position", [0, 1, 2], threshold=0.01)
        assert frac is not None
        assert frac > 0.5

    def test_too_short_returns_none(self):
        ep = _sine_episode(n_steps=2)
        assert _episode_vel_disc_fraction(ep, "position", [0]) is None


# ── unit tests: active_dims (gripper exclusion) ───────────────────────────────

class TestGetVelocity:
    def test_excludes_correct_dims(self):
        ep = _sine_episode(50, action_dim=7)
        analyzer = ControlSmoothnessAnalyzer(gripper_dims=[-1])
        active = analyzer._active_dims(ep)
        assert 6 not in active   # last dim (gripper) excluded
        assert 0 in active

    def test_no_exclusion(self):
        ep = _sine_episode(50, action_dim=7)
        analyzer = ControlSmoothnessAnalyzer(gripper_dims=[])
        active = analyzer._active_dims(ep)
        assert len(active) == 7


# ── integration: ControlSmoothnessAnalyzer ───────────────────────────────────

class TestControlSmoothnessAnalyzerSmooth:
    def test_smooth_batch_no_critical_flags(self):
        # A 0.5 Hz sine at LDLJ ≈ -13 falls in the WARNING band with default
        # thresholds (warning=-10). Use a low-frequency sine (0.05 Hz) to
        # get LDLJ ≈ -3 (well above WARNING), confirming the metric works.
        eps = [_sine_episode(200, dt=0.05, freq=0.05, ep_id=str(i)) for i in range(8)]
        batch = _batch_of(eps)
        result = ControlSmoothnessAnalyzer().analyze(batch)
        critical = [f for f in result.flags if f.level == RiskLevel.CRITICAL]
        assert critical == [], f"Unexpected CRITICAL on smooth data: {[f.metric for f in critical]}"

    def test_analyzer_name(self):
        eps = [_sine_episode(50, ep_id=str(i)) for i in range(3)]
        result = ControlSmoothnessAnalyzer().analyze(_batch_of(eps))
        assert result.analyzer_name == "control_smoothness"

    def test_raw_metrics_populated(self):
        eps = [_sine_episode(100, ep_id=str(i)) for i in range(5)]
        result = ControlSmoothnessAnalyzer().analyze(_batch_of(eps))
        assert "ldlj" in result.raw_metrics
        assert "jerk_spikes" in result.raw_metrics
        assert "vel_discontinuities" in result.raw_metrics


class TestControlSmoothnessAnalyzerJerky:
    def test_jerky_batch_flags_ldlj(self):
        eps = [_jerky_episode(200, dt=0.05, n_spikes=20, ep_id=str(i))
               for i in range(8)]
        result = ControlSmoothnessAnalyzer(
            ldlj_warning=-1.0, ldlj_critical=-3.0
        ).analyze(_batch_of(eps))
        ldlj_flags = [f for f in result.flags if f.metric == "ldlj"]
        assert ldlj_flags
        assert ldlj_flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)

    def test_jerky_batch_flags_spikes(self):
        eps = [_jerky_episode(200, dt=0.05, n_spikes=20, ep_id=str(i))
               for i in range(8)]
        result = ControlSmoothnessAnalyzer(
            jerk_spike_warning=0.001
        ).analyze(_batch_of(eps))
        spike_flags = [f for f in result.flags if "spike" in f.metric]
        assert spike_flags
        assert spike_flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)


class TestControlSmoothnessAnalyzerEdgeCases:
    def test_empty_batch(self):
        batch = _batch_of([])
        result = ControlSmoothnessAnalyzer().analyze(batch)
        assert result.flags == []

    def test_single_episode_no_crash(self):
        result = ControlSmoothnessAnalyzer().analyze(_batch_of([_sine_episode(50)]))
        assert result.analyzer_name == "control_smoothness"

    def test_policy_hints_not_emitted_without_family(self):
        eps = [_jerky_episode(100, ep_id=str(i)) for i in range(3)]
        result = ControlSmoothnessAnalyzer().analyze(_batch_of(eps), policy_family=None)
        assert result.hints == []

    def test_diffusion_hint_emitted_with_policy(self):
        eps = [_sine_episode(100, ep_id=str(i)) for i in range(3)]
        result = ControlSmoothnessAnalyzer().analyze(
            _batch_of(eps), policy_family="diffusion"
        )
        dp_hints = [h for h in result.hints if "Diffusion" in h.policy_family]
        assert dp_hints


# ── scripted motion signature ─────────────────────────────────────────────────

def _scripted_episode(n_steps: int = 200, dt: float = 0.02, ep_id: str = "ep_0") -> Episode:
    """
    Synthetic scripted-planner episode: sharp waypoint transitions (high spike)
    with no direction reversals (near-zero vel_disc).

    Structure: piecewise linear trajectory between 5 waypoints, with near-
    instantaneous transitions. The transition steps have extreme jerk. Between
    waypoints the motion is perfectly smooth, so velocity never reverses.
    """
    rng = np.random.default_rng(0)
    n_dim = 6
    waypoints = rng.random((6, n_dim)).astype(np.float32)
    seg_len = n_steps // 5

    actions = np.zeros((n_steps, n_dim), dtype=np.float32)
    for i in range(5):
        start = i * seg_len
        end = min((i + 1) * seg_len, n_steps)
        # Smooth linear interpolation within segment
        t_seg = np.linspace(0, 1, end - start)
        actions[start:end] = (
            waypoints[i] * (1 - t_seg[:, None]) + waypoints[i + 1] * t_seg[:, None]
        )
    # Inject abrupt spikes at waypoint boundaries
    for i in range(1, 5):
        idx = i * seg_len
        if idx < n_steps:
            actions[idx] = waypoints[i] + rng.random(n_dim).astype(np.float32) * 5.0

    ts = np.arange(n_steps, dtype=float) * dt
    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=ts,
        observations={"state": np.zeros((n_steps, n_dim), dtype=np.float32)},
        actions=actions,
    )


class TestScriptedMotionSignature:
    def test_smooth_human_data_does_not_trigger(self):
        """Clean sinusoidal (human-like) data must NOT fire the scripted flag."""
        eps = [_sine_episode(200, ep_id=str(i)) for i in range(5)]
        result = ControlSmoothnessAnalyzer().analyze(_batch_of(eps))
        sig_flags = [f for f in result.flags if f.metric == "motion_collection_signature"]
        assert sig_flags == [], "Human-like data should not trigger scripted signature"

    def test_velocity_action_type_does_not_trigger(self):
        """Scripted detection only applies to position-control data."""
        eps = [_scripted_episode(200, ep_id=str(i)) for i in range(5)]
        # Force velocity action type — should suppress the check
        result = ControlSmoothnessAnalyzer(action_type="velocity").analyze(_batch_of(eps))
        sig_flags = [f for f in result.flags if f.metric == "motion_collection_signature"]
        assert sig_flags == [], "Scripted check should not fire for velocity action_type"

    def test_scripted_episode_triggers_info_flag(self):
        """Scripted planner data must fire an INFO flag with correct metric name."""
        eps = [_scripted_episode(200, ep_id=str(i)) for i in range(5)]
        # Lower thresholds to ensure detection regardless of synthetic data specifics.
        analyzer = ControlSmoothnessAnalyzer(
            action_type="position",
            scripted_spike_min=0.01,       # very sensitive for synthetic data
            scripted_vel_disc_max=0.20,    # relaxed — synthetic vel_disc varies
        )
        result = analyzer.analyze(_batch_of(eps))
        sig_flags = [f for f in result.flags if f.metric == "motion_collection_signature"]
        assert sig_flags, "Scripted planner data should trigger motion_collection_signature"
        assert sig_flags[0].level == RiskLevel.INFO

    def test_scripted_flag_level_is_info(self):
        """The scripted signature flag must always be INFO, never WARNING/CRITICAL."""
        # Patch the analyzer to force the flag to fire.
        from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
        spike_raw = {"mean_spike_fraction": 0.25}
        disc_raw  = {"mean_disc_fraction": 0.005}
        analyzer = ControlSmoothnessAnalyzer(action_type="position")
        flag = analyzer._check_scripted_motion_signature(spike_raw, disc_raw)
        assert flag is not None
        assert flag.level == RiskLevel.INFO

    def test_below_spike_threshold_no_flag(self):
        """Spike below threshold: no scripted flag, even with low vel_disc."""
        analyzer = ControlSmoothnessAnalyzer(action_type="position")
        flag = analyzer._check_scripted_motion_signature(
            {"mean_spike_fraction": 0.05},  # below default 0.10
            {"mean_disc_fraction": 0.005},
        )
        assert flag is None

    def test_above_vel_disc_threshold_no_flag(self):
        """Vel_disc above threshold: no scripted flag, even with high spike."""
        analyzer = ControlSmoothnessAnalyzer(action_type="position")
        flag = analyzer._check_scripted_motion_signature(
            {"mean_spike_fraction": 0.25},
            {"mean_disc_fraction": 0.05},   # above default 0.015
        )
        assert flag is None

    def test_scripted_flag_includes_prune_guidance(self):
        """The implication text must mention the quality filter and max-spike-rate."""
        analyzer = ControlSmoothnessAnalyzer(action_type="position")
        flag = analyzer._check_scripted_motion_signature(
            {"mean_spike_fraction": 0.22},
            {"mean_disc_fraction": 0.007},
        )
        assert flag is not None
        assert "--max-spike-rate" in flag.implication

    def test_scripted_diffusion_hint_has_caveat(self):
        """When scripted signature is present, Diffusion Policy hint gets a caveat."""
        analyzer = ControlSmoothnessAnalyzer(action_type="position")
        flag = analyzer._check_scripted_motion_signature(
            {"mean_spike_fraction": 0.22},
            {"mean_disc_fraction": 0.007},
        )
        assert flag is not None
        hints = analyzer._policy_hints([flag], "diffusion", {
            "ldlj": {"mean_ldlj": -5.0},
            "jerk_spikes": {"mean_spike_fraction": 0.22},
        })
        dp_hints = [h for h in hints if "Diffusion" in h.policy_family]
        assert dp_hints
        scripted_caveats = [c for c in dp_hints[0].caveats if "cripted" in c]
        assert scripted_caveats, "Expected a caveat mentioning scripted data"
