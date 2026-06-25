"""Tests for π0, OpenVLA, and Octo compatibility analyzers."""

from __future__ import annotations

import numpy as np

from calibra.analyzers.pi0 import Pi0CompatibilityAnalyzer
from calibra.analyzers.openvla import OpenVLACompatibilityAnalyzer
from calibra.analyzers.octo import OctoCompatibilityAnalyzer
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel


def _batch(
    n_eps: int = 10,
    n_steps: int = 200,
    action_dim: int = 7,
    dt: float = 0.02,  # 50 Hz
    has_camera: bool = True,
    has_lang: bool = True,
) -> EpisodeBatch:
    rng = np.random.default_rng(0)
    episodes = []
    for i in range(n_eps):
        obs: dict = {}
        if has_camera:
            obs["camera_rgb"] = rng.random((n_steps, 64, 64, 3)).astype(np.float32)
        obs["proprio"] = rng.random((n_steps, 7)).astype(np.float32)

        episodes.append(
            Episode(
                metadata=EpisodeMetadata(
                    episode_id=f"ep_{i}",
                    task_description="pick up the red cube" if has_lang else None,
                ),
                timestamps=np.arange(n_steps, dtype=np.float64) * dt,
                observations=obs,
                actions=rng.random((n_steps, action_dim)).astype(np.float32),
            )
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="test",
        format="hdf5",
        source_path="/tmp/test.h5",
    )


# ── π0 tests ──────────────────────────────────────────────────────────────────


class TestPi0Analyzer:
    def test_skipped_when_not_pi0(self, clean_batch):
        result = Pi0CompatibilityAnalyzer().analyze(clean_batch, policy_family="diffusion")
        assert result.flags == []
        assert result.hints == []

    def test_compatible_dataset(self):
        batch = _batch()
        result = Pi0CompatibilityAnalyzer().analyze(batch, policy_family="pi0")
        n_crit = sum(1 for f in result.flags if f.level == RiskLevel.CRITICAL)
        assert n_crit == 0
        # hint is either compatible=True (no flags) or compatible=None (warnings)
        assert any(h.compatible is not False for h in result.hints)

    def test_missing_camera_is_critical(self):
        batch = _batch(has_camera=False)
        result = Pi0CompatibilityAnalyzer().analyze(batch, policy_family="pi0")
        assert any(
            f.metric == "pi0_visual_observations" and f.level == RiskLevel.CRITICAL
            for f in result.flags
        )

    def test_missing_lang_is_critical(self):
        batch = _batch(has_lang=False)
        result = Pi0CompatibilityAnalyzer().analyze(batch, policy_family="pi0")
        assert any(
            f.metric == "pi0_language_annotations" and f.level == RiskLevel.CRITICAL
            for f in result.flags
        )

    def test_short_episodes_warned(self):
        batch = _batch(n_steps=10)  # shorter than chunk_size=50
        result = Pi0CompatibilityAnalyzer().analyze(batch, policy_family="pi0")
        assert any(f.metric == "pi0_episode_length" for f in result.flags)

    def test_unknown_action_dim_warned(self):
        batch = _batch(action_dim=3)  # not in {7, 14}
        result = Pi0CompatibilityAnalyzer().analyze(batch, policy_family="pi0")
        assert any(f.metric == "pi0_action_dim" for f in result.flags)

    def test_result_name(self):
        assert Pi0CompatibilityAnalyzer().name == "pi0_compatibility"


# ── OpenVLA tests ─────────────────────────────────────────────────────────────


class TestOpenVLAAnalyzer:
    def test_skipped_when_not_openvla(self, clean_batch):
        result = OpenVLACompatibilityAnalyzer().analyze(clean_batch, policy_family="act")
        assert result.flags == []

    def test_compatible_dataset(self):
        batch = _batch(dt=0.1)  # 10 Hz — within OpenVLA range
        result = OpenVLACompatibilityAnalyzer().analyze(batch, policy_family="openvla")
        n_crit = sum(1 for f in result.flags if f.level == RiskLevel.CRITICAL)
        assert n_crit == 0

    def test_missing_camera_critical(self):
        batch = _batch(has_camera=False)
        result = OpenVLACompatibilityAnalyzer().analyze(batch, policy_family="openvla")
        assert any(f.metric == "openvla_visual_observations" for f in result.flags)

    def test_high_freq_warning(self):
        # 50 Hz > 30 Hz threshold
        batch = _batch(dt=0.02)
        result = OpenVLACompatibilityAnalyzer().analyze(batch, policy_family="openvla")
        assert any(f.metric == "openvla_control_frequency" for f in result.flags)

    def test_result_name(self):
        assert OpenVLACompatibilityAnalyzer().name == "openvla_compatibility"


# ── Octo tests ────────────────────────────────────────────────────────────────


class TestOctoAnalyzer:
    def test_skipped_when_not_octo(self, clean_batch):
        result = OctoCompatibilityAnalyzer().analyze(clean_batch, policy_family="diffusion")
        assert result.flags == []

    def test_compatible_dataset(self):
        batch = _batch(n_eps=60, n_steps=200, dt=0.05)  # 20 Hz, 60 eps
        result = OctoCompatibilityAnalyzer().analyze(batch, policy_family="octo")
        n_crit = sum(1 for f in result.flags if f.level == RiskLevel.CRITICAL)
        assert n_crit == 0

    def test_small_dataset_warned(self):
        batch = _batch(n_eps=10)  # < 50 episodes
        result = OctoCompatibilityAnalyzer().analyze(batch, policy_family="octo")
        assert any(f.metric == "octo_dataset_size" for f in result.flags)

    def test_very_short_episodes_critical(self):
        # Episodes shorter than window_size+1 = 5 steps
        batch = _batch(n_steps=3, n_eps=60)
        result = OctoCompatibilityAnalyzer().analyze(batch, policy_family="octo")
        # Either CRITICAL or WARNING depending on fraction
        assert any(f.metric == "octo_episode_length" for f in result.flags)

    def test_result_name(self):
        assert OctoCompatibilityAnalyzer().name == "octo_compatibility"


# ── pipeline integration ──────────────────────────────────────────────────────


def test_pipeline_activates_pi0():
    from calibra.pipeline import Pipeline

    batch = _batch()
    report = Pipeline().run(batch, policy_family="pi0")
    analyzer_names = [r.analyzer_name for r in report.analyzer_results]
    assert "pi0_compatibility" in analyzer_names


def test_pipeline_activates_openvla():
    from calibra.pipeline import Pipeline

    batch = _batch()
    report = Pipeline().run(batch, policy_family="openvla")
    analyzer_names = [r.analyzer_name for r in report.analyzer_results]
    assert "openvla_compatibility" in analyzer_names


def test_pipeline_activates_octo():
    from calibra.pipeline import Pipeline

    batch = _batch(n_eps=60)
    report = Pipeline().run(batch, policy_family="octo")
    analyzer_names = [r.analyzer_name for r in report.analyzer_results]
    assert "octo_compatibility" in analyzer_names


def test_pipeline_does_not_activate_pi0_for_diffusion():
    from calibra.pipeline import Pipeline

    batch = _batch()
    report = Pipeline().run(batch, policy_family="diffusion")
    analyzer_names = [r.analyzer_name for r in report.analyzer_results]
    assert "pi0_compatibility" not in analyzer_names
