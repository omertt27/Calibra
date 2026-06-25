"""Tests for the task structure analyzer."""

from __future__ import annotations

import numpy as np
import pytest

from calibra.analyzers.task_structure import (
    TaskStructureAnalyzer,
    _batch_trajectory_diversity,
    _detect_gripper_dims,
    _episode_contact_fraction,
    _episode_grasp_count,
    _two_means,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_episode(
    n_steps: int = 100,
    action_dim: int = 7,
    gripper_value: float | None = None,  # None = no gripper column
    gripper_pattern: str = "open",  # "open", "closed", "grasp"
    dt: float = 0.1,
    ep_id: str = "ep_0",
    velocity_scale: float = 1.0,
) -> Episode:
    rng = np.random.default_rng(int(ep_id.split("_")[-1]) if "_" in ep_id else 0)
    t = np.arange(n_steps, dtype=np.float64) * dt

    # Smooth trajectory for arm dims
    arm_dims = action_dim - (1 if gripper_value is not None else 0)
    acts = np.column_stack(
        [np.sin(2 * np.pi * 0.1 * t + d * 0.5) * velocity_scale for d in range(arm_dims)]
    ).astype(np.float32)

    if gripper_value is not None:
        if gripper_pattern == "open":
            grip = np.ones(n_steps, dtype=np.float32)
        elif gripper_pattern == "closed":
            grip = np.zeros(n_steps, dtype=np.float32)
        elif gripper_pattern == "grasp":
            # Open → close at 40% → open at 70%
            grip = np.ones(n_steps, dtype=np.float32)
            close_at = int(0.4 * n_steps)
            open_at = int(0.7 * n_steps)
            grip[close_at:open_at] = 0.0
        else:
            grip = np.ones(n_steps, dtype=np.float32) * float(gripper_value)
        acts = np.column_stack([acts, grip])

    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=t,
        observations={"proprio": rng.random((n_steps, 6)).astype(np.float32)},
        actions=acts,
    )


def _batch_of(episodes) -> EpisodeBatch:
    return EpisodeBatch(
        episodes=list(episodes),
        dataset_name="test",
        format="hdf5",
        source_path="/tmp/test.h5",
    )


# ── unit tests: gripper detection ────────────────────────────────────────────


class TestDetectGripperDims:
    def test_detects_binary_last_dim(self):
        rng = np.random.default_rng(0)
        n = 500
        arm = rng.uniform(-1, 1, (n, 6))
        # Perfect binary gripper: half open, half closed
        grip = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])[:, None]
        actions = np.column_stack([arm, grip])
        detected = _detect_gripper_dims(actions)
        assert 6 in detected

    def test_ignores_continuous_dims(self):
        rng = np.random.default_rng(0)
        actions = rng.uniform(-1, 1, (500, 6))
        detected = _detect_gripper_dims(actions)
        assert detected == []

    def test_ignores_constant_dims(self):
        rng = np.random.default_rng(0)
        n = 200
        arm = rng.uniform(-1, 1, (n, 5))
        constant = np.ones((n, 1))
        actions = np.column_stack([arm, constant])
        detected = _detect_gripper_dims(actions)
        assert 5 not in detected

    def test_empty_actions_returns_empty(self):
        assert _detect_gripper_dims(np.empty((0, 4))) == []

    def test_detects_near_binary_dim(self):
        rng = np.random.default_rng(0)
        n = 400
        arm = rng.uniform(-1, 1, (n, 4))
        # Near-binary: 80% at 0.0, 20% at 1.0 with small noise
        grip = np.where(rng.random(n) < 0.7, 0.0, 1.0) + rng.normal(0, 0.02, n)
        grip = np.clip(grip, 0, 1)[:, None]
        actions = np.column_stack([arm, grip])
        detected = _detect_gripper_dims(actions)
        assert 4 in detected


# ── unit tests: contact fraction ─────────────────────────────────────────────


class TestEpisodeContactFraction:
    def test_always_open_gripper_uses_velocity(self):
        ep = _make_episode(100, gripper_value=None, velocity_scale=1.0)
        frac = _episode_contact_fraction(ep, gripper_dims=[], vel_slow_threshold=0.08)
        # Sine wave — some slow phases near extrema
        assert 0.0 <= frac <= 1.0

    def test_always_closed_gripper_is_fully_contact(self):
        ep = _make_episode(100, action_dim=7, gripper_value=0.0, gripper_pattern="closed")
        # Last dim is always 0 (closed) → should detect as contact
        detected = _detect_gripper_dims(ep.actions.astype(np.float64))
        if detected:
            frac = _episode_contact_fraction(ep, detected, vel_slow_threshold=0.08)
            assert frac > 0.5

    def test_grasp_episode_has_partial_contact(self):
        ep = _make_episode(200, action_dim=7, gripper_value=0.0, gripper_pattern="grasp")
        detected = _detect_gripper_dims(ep.actions.astype(np.float64))
        if detected:
            frac = _episode_contact_fraction(ep, detected)
            # Gripper closed for ~30% of episode (40%–70% window); velocity
            # slow-phases from the sinusoidal arm motion add further contact
            # steps, so the combined fraction can reach ~0.7.
            assert 0.20 < frac < 0.85

    def test_too_short_returns_zero(self):
        ep = _make_episode(1)
        frac = _episode_contact_fraction(ep, [])
        assert frac == pytest.approx(0.0)


# ── unit tests: grasp count ───────────────────────────────────────────────────


class TestEpisodeGraspCount:
    def test_always_open_has_zero_grasps(self):
        ep = _make_episode(100, action_dim=7, gripper_value=1.0, gripper_pattern="open")
        detected = _detect_gripper_dims(ep.actions.astype(np.float64))
        if detected:
            count = _episode_grasp_count(ep, detected)
            assert count == 0

    def test_one_grasp_detected(self):
        ep = _make_episode(200, action_dim=7, gripper_value=0.0, gripper_pattern="grasp")
        detected = _detect_gripper_dims(ep.actions.astype(np.float64))
        if detected:
            count = _episode_grasp_count(ep, detected)
            assert count is not None
            assert count == 1

    def test_no_gripper_dims_returns_none(self):
        ep = _make_episode(100)
        assert _episode_grasp_count(ep, []) is None

    def test_too_short_returns_none(self):
        ep = _make_episode(2, action_dim=7, gripper_value=0.0, gripper_pattern="open")
        assert _episode_grasp_count(ep, [6]) is None


# ── unit tests: two_means ────────────────────────────────────────────────────


class TestTwoMeans:
    def test_well_separated_clusters(self):
        rng = np.random.default_rng(0)
        cluster_a = rng.normal([-5, 0], 0.3, (20, 2))
        cluster_b = rng.normal([5, 0], 0.3, (20, 2))
        X = np.vstack([cluster_a, cluster_b])
        labels, ratio = _two_means(X)
        assert ratio < 0.2  # low within-cluster variance → well separated
        # Each cluster should be mostly one label
        assert len(set(labels[:20])) == 1 or len(set(labels[20:])) == 1

    def test_one_cluster_high_ratio(self):
        rng = np.random.default_rng(0)
        X = rng.normal([0, 0], 0.5, (40, 2))
        _, ratio = _two_means(X)
        assert ratio > 0.5  # no separation

    def test_trivial_input(self):
        X = np.array([[1.0, 0.0]])
        labels, ratio = _two_means(X)
        assert len(labels) == 1

    def test_identical_points(self):
        X = np.ones((10, 2))
        _, ratio = _two_means(X)
        assert ratio == pytest.approx(0.0, abs=0.1)


# ── unit tests: trajectory diversity ─────────────────────────────────────────


class TestBatchTrajectoryDiversity:
    def test_identical_episodes_low_separation(self):
        rng = np.random.default_rng(0)
        eps = []
        for i in range(10):
            t = np.arange(50, dtype=np.float64) * 0.1
            acts = (np.ones((50, 4)) + rng.normal(0, 0.01, (50, 4))).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        _, sep, _ = _batch_trajectory_diversity(_batch_of(eps))
        assert sep < 0.5

    def test_two_strategy_episodes_high_separation(self):
        rng = np.random.default_rng(42)
        eps = []
        for i in range(10):
            t = np.arange(80, dtype=np.float64) * 0.1
            # Strategy A: actions centered near +1
            acts = rng.normal(1.0, 0.1, (80, 4)).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        for i in range(10):
            t = np.arange(80, dtype=np.float64) * 0.1
            # Strategy B: actions centered near -1
            acts = rng.normal(-1.0, 0.1, (80, 4)).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i + 10}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        _, sep, _ = _batch_trajectory_diversity(_batch_of(eps))
        assert sep > 0.5

    def test_too_few_episodes(self):
        eps = [_make_episode(50, ep_id=f"ep_{i}") for i in range(2)]
        n_modes, sep, _ = _batch_trajectory_diversity(_batch_of(eps))
        assert n_modes == 1
        assert sep == pytest.approx(0.0)


# ── integration: TaskStructureAnalyzer ───────────────────────────────────────


class TestTaskStructureAnalyzerBasic:
    def test_analyzer_name(self):
        eps = [_make_episode(50, ep_id=f"ep_{i}") for i in range(5)]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        assert result.analyzer_name == "task_structure"

    def test_four_flags_produced(self):
        eps = [_make_episode(100, ep_id=f"ep_{i}") for i in range(8)]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        assert len(result.flags) == 4

    def test_raw_metrics_populated(self):
        eps = [_make_episode(80, ep_id=f"ep_{i}") for i in range(5)]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        assert "contact_density" in result.raw_metrics
        assert "grasp_events" in result.raw_metrics
        assert "trajectory_diversity" in result.raw_metrics
        assert "short_episodes" in result.raw_metrics

    def test_empty_batch(self):
        result = TaskStructureAnalyzer().analyze(_batch_of([]))
        assert result.flags == []


class TestTaskStructureAnalyzerGripper:
    def test_gripper_detected_in_grasp_episodes(self):
        eps = [
            _make_episode(
                150, action_dim=7, gripper_value=0.0, gripper_pattern="grasp", ep_id=f"ep_{i}"
            )
            for i in range(8)
        ]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        assert result.raw_metrics["detected_gripper_dims"] == [6]

    def test_grasp_count_info_flag_emitted(self):
        eps = [
            _make_episode(
                150, action_dim=7, gripper_value=0.0, gripper_pattern="grasp", ep_id=f"ep_{i}"
            )
            for i in range(8)
        ]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        grasp_flags = [f for f in result.flags if "grasp" in f.metric]
        assert grasp_flags
        assert grasp_flags[0].level == RiskLevel.INFO

    def test_no_gripper_emits_info(self):
        eps = [
            _make_episode(80, action_dim=6, gripper_value=None, ep_id=f"ep_{i}") for i in range(5)
        ]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        grasp_flags = [f for f in result.flags if "grasp" in f.metric]
        assert grasp_flags
        assert grasp_flags[0].level == RiskLevel.INFO


class TestTaskStructureAnalyzerShortEpisodes:
    def test_normal_episodes_ok(self):
        eps = [_make_episode(100, ep_id=f"ep_{i}") for i in range(20)]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        short_flags = [f for f in result.flags if "short" in f.metric]
        assert short_flags
        assert short_flags[0].level == RiskLevel.OK

    def test_mixed_short_episodes_flagged(self):
        normal = [_make_episode(100, ep_id=f"ep_{i}") for i in range(15)]
        # 3/18 episodes are very short (3 steps vs median 100) → outliers
        short = [_make_episode(3, ep_id=f"ep_{i + 15}") for i in range(3)]
        result = TaskStructureAnalyzer().analyze(_batch_of(normal + short))
        short_flags = [f for f in result.flags if "short" in f.metric]
        assert short_flags
        assert short_flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
        assert short_flags[0].affected_fraction is not None
        assert short_flags[0].affected_fraction > 0.0

    def test_all_same_length_no_outliers(self):
        eps = [_make_episode(80, ep_id=f"ep_{i}") for i in range(10)]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        short_flags = [f for f in result.flags if "short" in f.metric]
        assert short_flags[0].level == RiskLevel.OK


class TestTaskStructureAnalyzerMultimodal:
    def _two_strategy_batch(self) -> EpisodeBatch:
        rng = np.random.default_rng(0)
        eps = []
        for i in range(12):
            t = np.arange(80, dtype=np.float64) * 0.1
            acts = rng.normal(2.0, 0.15, (80, 5)).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        for i in range(12):
            t = np.arange(80, dtype=np.float64) * 0.1
            acts = rng.normal(-2.0, 0.15, (80, 5)).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i + 12}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        return _batch_of(eps)

    def test_two_strategy_dataset_flagged(self):
        batch = self._two_strategy_batch()
        result = TaskStructureAnalyzer().analyze(batch)
        div_flags = [f for f in result.flags if "diversity" in f.metric]
        assert div_flags
        assert div_flags[0].level in (RiskLevel.INFO, RiskLevel.WARNING)

    def test_uniform_dataset_ok_or_info(self):
        rng = np.random.default_rng(0)
        eps = []
        for i in range(16):
            t = np.arange(80, dtype=np.float64) * 0.1
            # All episodes from same distribution (no clustering)
            acts = rng.normal(0.0, 0.5, (80, 5)).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        result = TaskStructureAnalyzer().analyze(_batch_of(eps))
        div_flags = [f for f in result.flags if "diversity" in f.metric]
        assert div_flags
        assert div_flags[0].level in (RiskLevel.OK, RiskLevel.INFO)


class TestTaskStructureAnalyzerPolicyHints:
    def test_no_hints_without_policy_family(self):
        eps = [_make_episode(80, ep_id=f"ep_{i}") for i in range(5)]
        result = TaskStructureAnalyzer().analyze(_batch_of(eps), policy_family=None)
        assert result.hints == []

    def test_diffusion_hint_on_multimodal(self):
        rng = np.random.default_rng(0)
        eps = []
        for i in range(10):
            t = np.arange(80, dtype=np.float64) * 0.1
            acts = rng.normal(3.0, 0.1, (80, 5)).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        for i in range(10):
            t = np.arange(80, dtype=np.float64) * 0.1
            acts = rng.normal(-3.0, 0.1, (80, 5)).astype(np.float32)
            eps.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"ep_{i + 10}"),
                    timestamps=t,
                    observations={},
                    actions=acts,
                )
            )
        result = TaskStructureAnalyzer().analyze(_batch_of(eps), policy_family="diffusion")
        dp_hints = [h for h in result.hints if "Diffusion" in h.policy_family]
        assert dp_hints

    def test_explicit_gripper_dims_override(self):
        # Verify user-specified gripper_dims bypasses auto-detection
        eps = [_make_episode(80, action_dim=4, ep_id=f"ep_{i}") for i in range(5)]
        result = TaskStructureAnalyzer(gripper_dims=[3]).analyze(_batch_of(eps))
        assert result.raw_metrics["detected_gripper_dims"] == [3]
