"""Tests for calibra.analyze — the unified `calibra analyze` demo command."""

from __future__ import annotations

import numpy as np

from calibra.analyze import _to_json, render_analysis, run_analysis
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_batch(
    n_eps: int = 10,
    n_steps: int = 60,
    action_dim: int = 6,
    with_proprio: bool = True,
    tasks: "list[str] | None" = None,
) -> EpisodeBatch:
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        t = np.linspace(0, 2 * np.pi, n_steps)
        acts = np.stack([np.sin(t + 0.1 * i) for _ in range(action_dim)], axis=1).astype(np.float32)
        obs = {}
        if with_proprio:
            obs["proprio"] = acts.copy()
        task = tasks[i % len(tasks)] if tasks else None
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}", task_description=task),
                timestamps=ts,
                observations=obs,
                actions=acts,
            )
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="analyze_test",
        format="hdf5",
        source_path="/tmp/analyze_test.h5",
    )


# ── run_analysis ─────────────────────────────────────────────────────────────


class TestRunAnalysis:
    def test_returns_populated_result(self):
        batch = _make_batch()
        result = run_analysis(batch)
        assert result.report.n_episodes == 10
        assert 0.0 <= result.score_result["total_score"] <= 100.0
        assert set(result.integrity_by_category.keys()) == {
            "Timestamps & sync",
            "Episode structure",
            "Camera feed",
            "Motion & control",
        }

    def test_camera_feed_not_evaluated_without_camera_data(self):
        batch = _make_batch()
        result = run_analysis(batch)
        assert result.integrity_by_category["Camera feed"] == []

    def test_below_min_episodes_skips_recommendation(self):
        batch = _make_batch(n_eps=3)
        result = run_analysis(batch)
        assert result.prune_result is None
        assert result.regime_diagnosis is None
        assert result.redundancy is None

    def test_at_least_5_episodes_produces_recommendation(self):
        batch = _make_batch(n_eps=10)
        result = run_analysis(batch)
        assert result.prune_result is not None
        assert result.regime_diagnosis is not None
        assert result.prune_result.n_original == 10

    def test_redundancy_none_without_proprio_capability(self):
        batch = _make_batch(n_eps=10, with_proprio=False)
        result = run_analysis(batch)
        # regime diagnosis still runs (uses smoothness/temporal metrics),
        # but state_redundancy specifically requires state/proprio capability.
        assert result.regime_diagnosis is not None
        assert result.redundancy is None

    def test_explicit_keep_fraction_overrides_heuristic(self):
        batch = _make_batch(n_eps=10)
        result = run_analysis(batch, keep_fraction=0.2)
        assert result.keep_fraction == 0.2

    def test_n_tasks_counts_distinct_non_none(self):
        batch = _make_batch(n_eps=9, tasks=["pick", "place", "pick"])
        result = run_analysis(batch)
        assert result.n_tasks == 2

    def test_n_tasks_zero_when_no_task_descriptions(self):
        batch = _make_batch(n_eps=5, tasks=None)
        result = run_analysis(batch)
        assert result.n_tasks == 0

    def test_action_dim_reflects_batch(self):
        batch = _make_batch(action_dim=7)
        result = run_analysis(batch)
        assert result.action_dim == 7

    def test_config_hash_stamped_on_report(self):
        batch = _make_batch()
        result = run_analysis(batch)
        assert result.report.config_hash != ""
        assert result.report.calibra_version != ""


# ── rendering ────────────────────────────────────────────────────────────────


class TestRenderAnalysis:
    def test_render_includes_all_sections(self):
        batch = _make_batch()
        result = run_analysis(batch)
        text = render_analysis(result)
        assert "CALIBRA ANALYSIS" in text
        assert "Integrity" in text
        assert "Quality (Calibra Score)" in text
        assert "Coverage / diversity" in text
        assert "Redundancy" in text
        assert "RECOMMENDATION" in text
        assert "Training set" in text

    def test_render_handles_below_min_episodes_gracefully(self):
        batch = _make_batch(n_eps=3)
        result = run_analysis(batch)
        text = render_analysis(result)
        assert "Not enough episodes" in text

    def test_render_includes_reproducibility_footer(self):
        batch = _make_batch()
        result = run_analysis(batch)
        text = render_analysis(result)
        assert f"Calibra v{result.report.calibra_version}" in text
        assert result.report.config_hash in text

    def test_render_mentions_heuristic_disclaimer(self):
        batch = _make_batch()
        result = run_analysis(batch)
        text = render_analysis(result)
        assert "heuristic starting point" in text
        assert "design-partner protocol" in text


# ── JSON ─────────────────────────────────────────────────────────────────────


class TestToJson:
    def test_json_serializable(self):
        import json

        batch = _make_batch()
        result = run_analysis(batch)
        payload = _to_json(result)
        json.dumps(payload)  # must not raise

    def test_json_recommendation_includes_prune_result(self):
        batch = _make_batch()
        result = run_analysis(batch)
        payload = _to_json(result)
        assert payload["recommendation"]["prune_result"]["n_original"] == 10

    def test_json_recommendation_none_when_too_few_episodes(self):
        batch = _make_batch(n_eps=3)
        result = run_analysis(batch)
        payload = _to_json(result)
        assert payload["recommendation"]["prune_result"] is None

    def test_json_integrity_categories_present(self):
        batch = _make_batch()
        result = run_analysis(batch)
        payload = _to_json(result)
        assert set(payload["integrity"]["categories"].keys()) == {
            "Timestamps & sync",
            "Episode structure",
            "Camera feed",
            "Motion & control",
        }
