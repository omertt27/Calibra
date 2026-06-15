"""Tests for the GR00T Compatibility Analyzer."""
from __future__ import annotations

import numpy as np
import pytest

from calibra.analyzers.gr00t import GR00TCompatibilityAnalyzer
from calibra.pipeline import Pipeline
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import AnalyzerResult, RiskLevel


# ── fixtures ─────────────────────────────────────────────────────────────────

def _make_ep(
    n_steps: int = 50,
    action_dim: int = 7,
    dt: float = 0.02,
    with_camera: bool = True,
    task: str | None = "pick up the red block",
) -> Episode:
    rng = np.random.default_rng(0)
    ts = np.arange(n_steps, dtype=np.float64) * dt
    obs: dict = {"proprio": rng.random((n_steps, 14)).astype(np.float32)}
    if with_camera:
        obs["agentview_image"] = rng.integers(0, 255, (n_steps, 84, 84, 3), dtype=np.uint8)
    return Episode(
        metadata=EpisodeMetadata(episode_id="ep_0", task_description=task),
        timestamps=ts,
        observations=obs,
        actions=rng.random((n_steps, action_dim)).astype(np.float32),
    )


def _make_batch(episodes: list[Episode], name: str = "test") -> EpisodeBatch:
    return EpisodeBatch(
        episodes=episodes, dataset_name=name, format="isaac_lab", source_path="/tmp/test"
    )


# ── policy_family gate ────────────────────────────────────────────────────────

class TestGR00TPolicyGate:
    def test_empty_result_when_no_policy(self):
        batch = _make_batch([_make_ep()])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family=None)
        assert isinstance(result, AnalyzerResult)
        assert result.flags == []
        assert result.hints == []

    def test_empty_result_for_other_policy(self):
        batch = _make_batch([_make_ep()])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="diffusion")
        assert result.flags == []

    def test_fires_for_gr00t(self):
        batch = _make_batch([_make_ep()])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        assert len(result.flags) > 0

    def test_fires_for_gr00t_case_insensitive(self):
        batch = _make_batch([_make_ep()])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="GR00T")
        assert len(result.flags) > 0


# ── visual modality check ─────────────────────────────────────────────────────

class TestGR00TVisualModality:
    def test_ok_when_camera_present(self):
        batch = _make_batch([_make_ep(with_camera=True)])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        visual_flags = [f for f in result.flags if "visual_modality" in f.metric]
        assert visual_flags[0].level == RiskLevel.OK

    def test_critical_when_no_camera(self):
        batch = _make_batch([_make_ep(with_camera=False)])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        visual_flags = [f for f in result.flags if "visual_modality" in f.metric]
        assert visual_flags[0].level == RiskLevel.CRITICAL

    def test_camera_key_variants_detected(self):
        for key in ("camera_rgb", "rgb_image", "visual_obs", "depth_frame"):
            rng = np.random.default_rng(0)
            ep = _make_ep(with_camera=False)
            ep.observations[key] = rng.random((50, 10)).astype(np.float32)
            batch = _make_batch([ep])
            result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
            visual_flags = [f for f in result.flags if "visual_modality" in f.metric]
            assert visual_flags[0].level == RiskLevel.OK, f"failed for key={key!r}"


# ── language annotation check ─────────────────────────────────────────────────

class TestGR00TLanguage:
    def test_ok_when_all_annotated(self):
        eps = [_make_ep(task="pick up the cube") for _ in range(5)]
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        lang_flags = [f for f in result.flags if "language" in f.metric]
        assert lang_flags[0].level == RiskLevel.OK

    def test_warning_when_half_annotated(self):
        eps = [_make_ep(task="task") for _ in range(5)]
        eps += [_make_ep(task=None) for _ in range(5)]
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        lang_flags = [f for f in result.flags if "language" in f.metric]
        assert lang_flags[0].level == RiskLevel.WARNING

    def test_critical_when_none_annotated(self):
        eps = [_make_ep(task=None) for _ in range(5)]
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        lang_flags = [f for f in result.flags if "language" in f.metric]
        assert lang_flags[0].level == RiskLevel.CRITICAL

    def test_empty_string_treated_as_unannotated(self):
        eps = [_make_ep(task="") for _ in range(5)]
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        lang_flags = [f for f in result.flags if "language" in f.metric]
        assert lang_flags[0].level == RiskLevel.CRITICAL


# ── episode length check ──────────────────────────────────────────────────────

class TestGR00TEpisodeLength:
    def test_ok_when_episodes_long_enough(self):
        eps = [_make_ep(n_steps=50) for _ in range(5)]
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        len_flags = [f for f in result.flags if "episode_length" in f.metric]
        assert len_flags[0].level == RiskLevel.OK

    def test_warning_when_some_too_short(self):
        eps = [_make_ep(n_steps=50) for _ in range(9)]
        eps.append(_make_ep(n_steps=8))  # 1/10 < 20% → WARNING
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        len_flags = [f for f in result.flags if "episode_length" in f.metric]
        assert len_flags[0].level == RiskLevel.WARNING

    def test_critical_when_many_too_short(self):
        eps = [_make_ep(n_steps=5) for _ in range(10)]  # all < chunk_size=16
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        len_flags = [f for f in result.flags if "episode_length" in f.metric]
        assert len_flags[0].level == RiskLevel.CRITICAL

    def test_custom_chunk_size(self):
        eps = [_make_ep(n_steps=25) for _ in range(5)]
        batch = _make_batch(eps)
        analyzer = GR00TCompatibilityAnalyzer(chunk_size=32)
        result = analyzer.analyze(batch, policy_family="gr00t")
        len_flags = [f for f in result.flags if "episode_length" in f.metric]
        assert len_flags[0].level != RiskLevel.OK


# ── control frequency check ───────────────────────────────────────────────────

class TestGR00TControlFrequency:
    def test_ok_at_50hz(self):
        eps = [_make_ep(dt=0.02) for _ in range(5)]  # 50 Hz
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        freq_flags = [f for f in result.flags if "control_frequency" in f.metric]
        assert freq_flags[0].level == RiskLevel.OK

    def test_warning_below_15hz(self):
        eps = [_make_ep(dt=0.1) for _ in range(5)]  # 10 Hz
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        freq_flags = [f for f in result.flags if "control_frequency" in f.metric]
        assert freq_flags[0].level == RiskLevel.WARNING

    def test_warning_above_120hz(self):
        eps = [_make_ep(dt=0.005) for _ in range(5)]  # 200 Hz
        batch = _make_batch(eps)
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        freq_flags = [f for f in result.flags if "control_frequency" in f.metric]
        assert freq_flags[0].level == RiskLevel.WARNING


# ── action dim check ──────────────────────────────────────────────────────────

class TestGR00TActionDim:
    @pytest.mark.parametrize("dim", [7, 8, 14, 16])
    def test_ok_for_known_dims(self, dim):
        batch = _make_batch([_make_ep(action_dim=dim)])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        dim_flags = [f for f in result.flags if "action_dim" in f.metric]
        assert dim_flags[0].level == RiskLevel.OK

    def test_warning_for_unusual_dim(self):
        batch = _make_batch([_make_ep(action_dim=5)])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        dim_flags = [f for f in result.flags if "action_dim" in f.metric]
        assert dim_flags[0].level == RiskLevel.WARNING


# ── overall hint ──────────────────────────────────────────────────────────────

class TestGR00TOverallHint:
    def test_compatible_true_when_all_ok(self):
        ep = _make_ep(n_steps=50, action_dim=7, dt=0.02,
                      with_camera=True, task="pick the cube")
        batch = _make_batch([ep])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        gr00t_hints = [h for h in result.hints if h.policy_family == "GR00T N1"]
        assert gr00t_hints[0].compatible is True

    def test_compatible_false_when_critical(self):
        ep = _make_ep(with_camera=False, task=None)
        batch = _make_batch([ep])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        gr00t_hints = [h for h in result.hints if h.policy_family == "GR00T N1"]
        assert gr00t_hints[0].compatible is False

    def test_compatible_none_when_warnings_only(self):
        # Low freq → WARNING, but camera + language are ok
        ep = _make_ep(dt=0.1, with_camera=True, task="task", action_dim=7, n_steps=50)
        batch = _make_batch([ep])
        result = GR00TCompatibilityAnalyzer().analyze(batch, policy_family="gr00t")
        gr00t_hints = [h for h in result.hints if h.policy_family == "GR00T N1"]
        assert gr00t_hints[0].compatible is None


# ── pipeline integration ──────────────────────────────────────────────────────

class TestGR00TPipelineIntegration:
    def test_gr00t_analyzer_injected_in_pipeline(self):
        ep = _make_ep()
        batch = _make_batch([ep])
        report = Pipeline().run(batch, policy_family="gr00t")
        names = {r.analyzer_name for r in report.analyzer_results}
        assert "gr00t_compatibility" in names

    def test_gr00t_analyzer_absent_without_policy(self):
        ep = _make_ep()
        batch = _make_batch([ep])
        report = Pipeline().run(batch, policy_family=None)
        names = {r.analyzer_name for r in report.analyzer_results}
        assert "gr00t_compatibility" not in names

    def test_gr00t_analyzer_absent_for_diffusion(self):
        ep = _make_ep()
        batch = _make_batch([ep])
        report = Pipeline().run(batch, policy_family="diffusion")
        names = {r.analyzer_name for r in report.analyzer_results}
        assert "gr00t_compatibility" not in names

    def test_gr00t_hints_appear_in_report_summary(self):
        ep = _make_ep()
        batch = _make_batch([ep])
        report = Pipeline().run(batch, policy_family="gr00t")
        summary = report.summary()
        assert "GR00T" in summary
