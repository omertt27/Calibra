"""Tests for the GRAIL adapter."""
from __future__ import annotations

import pickle
import gzip
import numpy as np

from calibra.ingestion.adapters.grail import GRAILReader
from calibra.schema.episode import EpisodeBatch
from calibra.core.normalizer import SchemaNormalizer


def _write_mock_grail_file(
    path: str,
    n_steps: int = 50,
    num_dof: int = 29,
    compressed: bool = False,
    with_vel: bool = True,
    with_root: bool = True,
    empty: bool = False,
) -> None:
    data = {}
    if not empty:
        rng = np.random.default_rng(0)
        data["dof_pos"] = rng.random((n_steps, num_dof)).astype(np.float32)
        if with_vel:
            data["dof_vel"] = rng.random((n_steps, num_dof)).astype(np.float32)
        if with_root:
            data["root_state"] = rng.random((n_steps, 13)).astype(np.float32)

    if compressed:
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f)
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f)


# ── detection ────────────────────────────────────────────────────────────────

class TestGRAILDetection:
    def test_can_read_true_for_uncompressed_pkl(self, tmp_path):
        p = tmp_path / "trajectory_0.pkl"
        _write_mock_grail_file(str(p))
        assert GRAILReader.can_read(str(p))

    def test_can_read_true_for_compressed_pkl(self, tmp_path):
        p = tmp_path / "trajectory_0.pkl.gz"
        _write_mock_grail_file(str(p), compressed=True)
        assert GRAILReader.can_read(str(p))

    def test_can_read_false_for_empty_pkl(self, tmp_path):
        p = tmp_path / "empty.pkl"
        _write_mock_grail_file(str(p), empty=True)
        assert not GRAILReader.can_read(str(p))

    def test_can_read_false_for_non_pkl(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n")
        assert not GRAILReader.can_read(str(p))

    def test_can_read_directory(self, tmp_path):
        # Empty directory should return False
        assert not GRAILReader.can_read(str(tmp_path))
        
        # Directory with valid pkl should return True
        _write_mock_grail_file(str(tmp_path / "traj.pkl"))
        assert GRAILReader.can_read(str(tmp_path))


# ── reading ──────────────────────────────────────────────────────────────────

class TestGRAILRead:
    def test_returns_episode_batch(self, tmp_path):
        p = tmp_path / "trajectory.pkl"
        _write_mock_grail_file(str(p), n_steps=30, num_dof=19)
        batch = GRAILReader(fps=50.0).read(str(p))
        assert isinstance(batch, EpisodeBatch)
        assert batch.format == "grail"
        assert batch.n_episodes == 1
        assert batch.n_samples == 30
        
        ep = batch.episodes[0]
        assert ep.actions.shape == (30, 19)
        assert "dof_pos" in ep.observations
        assert "dof_vel" in ep.observations
        assert "root_state" in ep.observations
        assert ep.timestamps.shape == (30,)
        np.testing.assert_allclose(ep.timestamps[1] - ep.timestamps[0], 0.02, atol=1e-6)

    def test_read_directory(self, tmp_path):
        _write_mock_grail_file(str(tmp_path / "traj_1.pkl"), n_steps=20)
        _write_mock_grail_file(str(tmp_path / "traj_2.pkl"), n_steps=40)
        batch = GRAILReader().read(str(tmp_path))
        assert batch.n_episodes == 2
        assert batch.n_samples == 60
        assert batch.episodes[0].metadata.episode_id == "traj_1"
        assert batch.episodes[1].metadata.episode_id == "traj_2"

    def test_normalization_maps_correctly(self):
        normalizer = SchemaNormalizer()
        raw_obs = {"dof_pos": np.zeros((10, 5)), "dof_vel": np.ones((10, 5))}
        normalized = normalizer.normalize(raw_obs)
        assert "joint_position" in normalized
        assert "joint_velocity" in normalized
        assert np.array_equal(normalized["joint_position"], raw_obs["dof_pos"])
        assert np.array_equal(normalized["joint_velocity"], raw_obs["dof_vel"])
