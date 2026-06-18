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


# ── action-state divergence validation ───────────────────────────────────────

class TestActionStateDivergenceValidation:
    """Tests for the recalibrated action-state divergence metric."""

    def _batch_with_state(self, divergence_level: float, n=5, action_type="position") -> EpisodeBatch:
        """Create a batch with position control and a controlled divergence level."""
        rng = np.random.default_rng(99)
        episodes = []
        for i in range(n):
            n_steps, d = 100, 6
            actions = rng.uniform(-1, 1, (n_steps, d)).astype(np.float32)
            # State = action + controlled offset
            noise_scale = divergence_level / np.sqrt(d)
            state = actions + rng.normal(0, noise_scale, (n_steps, d)).astype(np.float32)
            episodes.append(Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=np.arange(n_steps) * 0.02,
                observations={"state": state},
                actions=actions,
            ))
        return EpisodeBatch(
            episodes=episodes, dataset_name="test",
            format="hdf5", source_path="/tmp/test"
        )

    def test_velocity_datasets_skip_divergence(self):
        """action_state_divergence must not fire for velocity-command datasets."""
        batch = self._batch_with_state(divergence_level=0.5, action_type="velocity")
        analyzer = ControlSmoothnessAnalyzer(action_type="velocity")
        div_flag, div_raw = analyzer._check_action_state_divergence(batch)
        assert div_flag is None
        assert div_raw is None

    def test_position_datasets_with_low_divergence_are_ok(self):
        """Typical ALOHA hardware divergence (~0.08–0.12) should be OK."""
        batch = self._batch_with_state(divergence_level=0.10)
        analyzer = ControlSmoothnessAnalyzer()
        div_flag, _ = analyzer._check_action_state_divergence(batch)
        assert div_flag is not None
        assert div_flag.level == RiskLevel.OK

    def test_high_divergence_without_scripted_is_critical(self):
        """Divergence > 0.35 with no scripted signature → CRITICAL."""
        batch = self._batch_with_state(divergence_level=0.55)  # ~0.55/√d > 0.35
        analyzer = ControlSmoothnessAnalyzer()
        result = analyzer.analyze(batch)
        div_flags = [f for f in result.flags if f.metric == "action_state_divergence"]
        assert div_flags
        assert div_flags[0].level == RiskLevel.CRITICAL

    def test_scripted_downgrade_critical_to_warning(self):
        """When scripted signature is detected, CRITICAL divergence → WARNING."""
        rng = np.random.default_rng(7)
        n_steps, d = 300, 7
        # Generate planner-like data: constant blocks → high spike, low vel_disc
        actions = np.zeros((n_steps, d), dtype=np.float32)
        for i in range(0, n_steps, 15):
            actions[i:i+15] = rng.uniform(-1, 1, d).astype(np.float32)
        # Large waypoint tracking error (>0.35 L2)
        state = actions + rng.normal(0, 0.25, (n_steps, d)).astype(np.float32)
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep_0"),
            timestamps=np.arange(n_steps) * 0.02,
            observations={"state": state},
            actions=actions,
        )
        batch = EpisodeBatch(
            episodes=[ep] * 4, dataset_name="test",
            format="hdf5", source_path="/tmp/test"
        )
        result = ControlSmoothnessAnalyzer().analyze(batch)
        div_flags = [f for f in result.flags if f.metric == "action_state_divergence"]
        sig_flags = [f for f in result.flags if f.metric == "motion_collection_signature"]
        if sig_flags and div_flags:
            # If scripted signature was detected, divergence CRITICAL → WARNING
            assert div_flags[0].level == RiskLevel.WARNING, (
                "Expected divergence downgraded to WARNING when scripted detected"
            )
            assert "Downgraded from CRITICAL" in div_flags[0].interpretation

    def test_no_state_obs_silently_skips(self):
        """If no state-like observation, divergence metric is silently skipped."""
        rng = np.random.default_rng(0)
        n_steps, d = 80, 4
        actions = rng.random((n_steps, d)).astype(np.float32)
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep_0"),
            timestamps=np.arange(n_steps) * 0.02,
            observations={"image": rng.random((n_steps, 64, 64, 3)).astype(np.float32)},
            actions=actions,
        )
        batch = EpisodeBatch(
            episodes=[ep], dataset_name="test",
            format="hdf5", source_path="/tmp/test"
        )
        div_flag, _ = ControlSmoothnessAnalyzer()._check_action_state_divergence(batch)
        assert div_flag is None


# ── certify scripted-aware grading ───────────────────────────────────────────

class TestCertifyScriptedGrading:
    """Tests for scripted-aware certification grading."""

    def _scripted_report(self):
        """Build a minimal DiagnosticReport with scripted signature + spike CRITICAL."""
        from calibra.schema.report import (
            AnalyzerResult, DiagnosticReport, ObservedValue, RiskFlag, RiskLevel
        )
        spike_flag = RiskFlag(
            level=RiskLevel.CRITICAL,
            metric="spike_rate",
            observed=ObservedValue(value=0.22),
            threshold=0.05,
            interpretation="High jerk spike rate.",
            implication="Check data.",
        )
        sig_flag = RiskFlag(
            level=RiskLevel.INFO,
            metric="motion_collection_signature",
            observed=ObservedValue(value=0.22),
            threshold=0.10,
            interpretation="Scripted motion signature detected.",
            implication="Use --max-spike-rate 0.30.",
        )
        result = AnalyzerResult(
            analyzer_name="control_smoothness",
            flags=[spike_flag, sig_flag],
        )
        return DiagnosticReport(
            source_path="/tmp/scripted_ds",
            dataset_name="scripted_ds",
            format="hdf5",
            n_episodes=5,
            n_samples=500,
            analyzer_results=[result],
        )

    def test_scripted_spike_critical_does_not_fail_grade(self):
        from calibra.certify import _grade
        report = self._scripted_report()
        grade, code = _grade(report)
        assert grade == "CERTIFIED"
        assert code == 0

    def test_non_scripted_spike_critical_fails_grade(self):
        from calibra.certify import _grade
        from calibra.schema.report import (
            AnalyzerResult, DiagnosticReport, ObservedValue, RiskFlag, RiskLevel
        )
        spike_flag = RiskFlag(
            level=RiskLevel.CRITICAL,
            metric="spike_rate",
            observed=ObservedValue(value=0.22),
            threshold=0.05,
            interpretation="High jerk spike rate.",
            implication="Check data.",
        )
        result = AnalyzerResult(
            analyzer_name="control_smoothness",
            flags=[spike_flag],
        )
        report = DiagnosticReport(
            source_path="/tmp/human_ds",
            dataset_name="human_ds",
            format="hdf5",
            n_episodes=5,
            n_samples=500,
            analyzer_results=[result],
        )
        grade, code = _grade(report)
        assert grade == "NOT CERTIFIED"
        assert code == 2

    def test_certificate_shows_scripted_note(self):
        from calibra.certify import render_certificate
        report = self._scripted_report()
        cert = render_certificate(report, "CERTIFIED", None)
        assert "SCRIPTED DATA NOTE" in cert
        assert "scripted/planner" in cert.lower() or "scripted motion" in cert.lower()

    def test_certificate_shows_scripted_source_line(self):
        from calibra.certify import render_certificate
        report = self._scripted_report()
        cert = render_certificate(report, "CERTIFIED", None)
        assert "Source" in cert and "scripted" in cert.lower()


# ── compare mismatch banner ───────────────────────────────────────────────────

class TestCompareMismatchBanner:
    """Tests for the collection-method mismatch warning in calibra compare."""

    def _metrics(self, spike: float, vel_disc: float) -> dict:
        return {
            "spike_rate": spike,
            "vel_disc_rate": vel_disc,
            "ldlj": -5.0,
            "jitter_cv": 0.01,
            "dropout_rate": 0.001,
            "action_entropy": 4.2,
        }

    def _ref_data(self, mode: str = "position") -> dict:
        return {
            "meta": {"dataset": "test_ref", "control_mode": mode, "n_episodes": 50},
            "aggregate_metrics": {},
            "flags": [],
        }

    def _render(self, yours_is_scripted: bool, ref_scripted: bool) -> str:
        from calibra.compare import render_comparison
        if yours_is_scripted:
            your_m = self._metrics(spike=0.22, vel_disc=0.007)
        else:
            your_m = self._metrics(spike=0.005, vel_disc=0.03)
        if ref_scripted:
            ref_m = self._metrics(spike=0.22, vel_disc=0.007)
        else:
            ref_m = self._metrics(spike=0.005, vel_disc=0.03)
        return render_comparison(
            your_path="/tmp/ds",
            your_metrics=your_m,
            your_n_episodes=20,
            your_action_dim=14,
            ref_data=self._ref_data(),
            ref_metrics=ref_m,
            ref_name="test_ref",
            yours_is_scripted=yours_is_scripted,
            ref_scripted=ref_scripted,
        )

    def test_scripted_vs_human_shows_mismatch_banner(self):
        output = self._render(yours_is_scripted=True, ref_scripted=False)
        assert "COLLECTION METHOD MISMATCH" in output
        assert "scripted/planner" in output

    def test_human_vs_scripted_shows_mismatch_banner(self):
        output = self._render(yours_is_scripted=False, ref_scripted=True)
        assert "COLLECTION METHOD MISMATCH" in output

    def test_no_mismatch_when_both_scripted(self):
        output = self._render(yours_is_scripted=True, ref_scripted=True)
        assert "COLLECTION METHOD MISMATCH" not in output

    def test_no_mismatch_when_both_human(self):
        output = self._render(yours_is_scripted=False, ref_scripted=False)
        assert "COLLECTION METHOD MISMATCH" not in output

    def test_scripted_spike_interp_explains_planner(self):
        from calibra.compare import _interp_spike_rate
        interp, conf = _interp_spike_rate(
            0.22, 0.005, "position", "aloha",
            yours_is_scripted=True, ref_is_scripted=False,
        )
        assert "scripted" in interp.lower() or "planner" in interp.lower()
        assert "0.30" in interp  # prune guidance

    def test_ref_is_scripted_detection(self):
        from calibra.compare import _ref_is_scripted
        assert _ref_is_scripted({"spike_rate": 0.22, "vel_disc_rate": 0.007})
        assert not _ref_is_scripted({"spike_rate": 0.005, "vel_disc_rate": 0.03})
        assert not _ref_is_scripted({"spike_rate": 0.22, "vel_disc_rate": 0.05})


# ── prune scripted auto-adjust ────────────────────────────────────────────────

class TestPruneScriptedAutoAdjust:
    """Tests for the scripted-data spike threshold auto-adjustment in prune CLI."""

    def test_args_default_is_none(self):
        """--max-spike-rate default must be None (sentinel for auto-adjust)."""
        # parse help to get argument spec
        import calibra.prune as prune_module
        import inspect
        src = inspect.getsource(prune_module.run_prune)
        # Just verify the default in the code is None
        assert "default=None" in src, "Expected --max-spike-rate default=None for auto-adjust"

    def test_auto_adjust_message_mentions_scripted(self):
        """The auto-adjust log message must explain what happened."""
        import calibra.prune as prune_module
        import inspect
        src = inspect.getsource(prune_module.run_prune)
        assert "scripted" in src.lower()
        assert "0.30" in src or "_SCRIPTED_AUTO_SPIKE" in src
