"""Tests for curriculum partitioning, energy strategy and latent-space options in `calibra prune`."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.prune import run_prune


def _make_batch(n_episodes: int = 6, n_steps: int = 20):
    rng = np.random.default_rng(42)
    episodes = []
    for i in range(n_episodes):
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=np.arange(n_steps, dtype=float) * 0.02,
                observations={
                    "proprio": rng.random((n_steps, 4)).astype(np.float32),
                    "camera_rgb": rng.random((n_steps, 8, 8, 3)).astype(np.float32),
                },
                actions=rng.random((n_steps, 4)).astype(np.float32),
            )
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="test_curriculum",
        format="hdf5",
        source_path="/tmp/test_curriculum.h5",
    )


class TestPruneExtensions:
    def test_curriculum_partitioning(self):
        batch = _make_batch(n_episodes=6, n_steps=30)
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "coreset.json"
            curr_json = Path(tmp) / "curriculum_index.json"

            with patch("calibra.ingestion.registry.load", return_value=batch):
                run_prune(
                    [
                        "/dummy/path",
                        "--keep",
                        "1.0",
                        "--out",
                        str(out_json),
                        "--curriculum",
                        "--max-spike-rate",
                        "1.0",
                        "--max-vel-disc-rate",
                        "1.0",
                        "--max-dropout",
                        "1.0",
                        "--min-ldlj",
                        "-1000.0",
                    ]
                )

            assert out_json.exists()
            assert curr_json.exists()

            with open(curr_json, "r") as f:
                data = json.load(f)

            assert "stage_1_intuitive_physics" in data
            assert "stage_2_spatial_planning" in data
            assert "stage_3_task_completions" in data

            s1 = data["stage_1_intuitive_physics"]
            s2 = data["stage_2_spatial_planning"]
            s3 = data["stage_3_task_completions"]

            assert len(s1) == 2
            assert len(s2) == 2
            assert len(s3) == 2

            all_ids = set(s1) | set(s2) | set(s3)
            assert len(all_ids) == 6
            assert all_ids == {f"ep_{i}" for i in range(6)}

    def test_latent_space_proprio_and_visual(self):
        batch = _make_batch(n_episodes=4, n_steps=20)
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "coreset.json"
            with patch("calibra.ingestion.registry.load", return_value=batch):
                # Test visual latent space
                run_prune(
                    [
                        "/dummy/path",
                        "--keep",
                        "0.5",
                        "--out",
                        str(out_json),
                        "--latent-space",
                        "visual",
                        "--max-spike-rate",
                        "1.0",
                        "--max-vel-disc-rate",
                        "1.0",
                        "--max-dropout",
                        "1.0",
                        "--min-ldlj",
                        "-1000.0",
                    ]
                )
            assert out_json.exists()

    def test_strategy_energy(self):
        batch = _make_batch(n_episodes=12, n_steps=25)
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "coreset.json"
            with patch("calibra.ingestion.registry.load", return_value=batch):
                run_prune(
                    [
                        "/dummy/path",
                        "--keep",
                        "0.5",
                        "--out",
                        str(out_json),
                        "--strategy",
                        "energy",
                        "--max-spike-rate",
                        "1.0",
                        "--max-vel-disc-rate",
                        "1.0",
                        "--max-dropout",
                        "1.0",
                        "--min-ldlj",
                        "-1000.0",
                    ]
                )
            assert out_json.exists()
            with open(out_json, "r") as f:
                data = json.load(f)
            assert data["n_kept"] == 6
