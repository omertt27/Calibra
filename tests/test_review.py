"""Tests for the `calibra review` CLI (calibra/review.py)."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from calibra.review import run_review
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata


def _make_batch(n_eps: int = 8, n_steps: int = 80, outlier_idx: int | None = 3) -> EpisodeBatch:
    rng = np.random.default_rng(0)
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        acts = np.cumsum(rng.normal(0, 0.02, (n_steps, 6)), axis=0).astype(np.float32)
        if outlier_idx is not None and i == outlier_idx:
            spike_steps = rng.choice(np.arange(10, n_steps - 10), size=8, replace=False)
            acts[spike_steps] += rng.normal(0, 3.0, (8, 6)).astype(np.float32)
        obs = {"proprio": rng.uniform(-1, 1, (n_steps, 8)).astype(np.float32)}
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=ts,
                observations=obs,
                actions=acts,
            )
        )
    return EpisodeBatch(
        episodes=episodes, dataset_name="review_test", format="hdf5", source_path="/dummy/path.h5"
    )


class TestRunReview:
    def test_prints_ranked_queue_with_outlier_first(self, capsys):
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            run_review(["/dummy/path.h5", "--top", "3"])
        out = capsys.readouterr().out
        assert "Episode Review Queue" in out
        assert "1. ep_ep_3" in out
        assert "Suggested action: Inspect" in out

    def test_json_output_structure(self, capsys):
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            run_review(["/dummy/path.h5", "--top", "3", "--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["n_episodes"] == 8
        assert payload["top_n"] == 3
        assert payload["episode_ids"][0] == "ep_3"
        assert payload["assessments"][0]["suggested_action"] == "Inspect"

    def test_output_file_written(self, tmp_path, capsys):
        batch = _make_batch()
        out_path = tmp_path / "episode_ids.json"
        with patch("calibra.ingestion.registry.load", return_value=batch):
            run_review(["/dummy/path.h5", "--top", "2", "--output", str(out_path)])
        payload = json.loads(out_path.read_text())
        assert len(payload["episode_ids"]) == 2
        assert payload["episode_ids"][0] == "ep_3"

    def test_top_limits_count(self, capsys):
        batch = _make_batch(n_eps=8)
        with patch("calibra.ingestion.registry.load", return_value=batch):
            run_review(["/dummy/path.h5", "--top", "2", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["assessments"]) == 2

    def test_fast_mode_has_no_coverage_value(self, capsys):
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            run_review(["/dummy/path.h5", "--mode", "fast", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert all(a["coverage_value"] is None for a in payload["assessments"])

    def test_full_mode_has_coverage_value(self, capsys):
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            run_review(["/dummy/path.h5", "--mode", "full", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert all(a["coverage_value"] is not None for a in payload["assessments"])

    def test_group_by_is_threaded_through(self, capsys):
        batch = _make_batch()
        with patch("calibra.ingestion.registry.load", return_value=batch):
            run_review(["/dummy/path.h5", "--group-by", "task", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["group_by"] == ["task"]

    def test_no_episodes_message(self, capsys):
        empty = EpisodeBatch(
            episodes=[], dataset_name="empty", format="hdf5", source_path="/dummy/empty.h5"
        )
        with patch("calibra.ingestion.registry.load", return_value=empty):
            run_review(["/dummy/empty.h5"])
        out = capsys.readouterr().out
        assert "No episodes ranked" in out
