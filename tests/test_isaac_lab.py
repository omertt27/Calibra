"""Tests for the Isaac Lab HDF5 adapter."""

from __future__ import annotations


import numpy as np
import pytest

from calibra.ingestion.adapters.isaac_lab import IsaacLabReader
from calibra.schema.episode import EpisodeBatch


def _write_isaac_lab_file(
    path: str,
    n_demos: int = 3,
    n_steps: int = 50,
    action_dim: int = 7,
    with_images: bool = False,
    env_name: str = "Lift",
    with_mask: bool = True,
) -> None:
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        f.attrs["total"] = n_demos
        f.attrs["env"] = env_name

        data_grp = f.create_group("data")
        for i in range(n_demos):
            demo = data_grp.create_group(f"demo_{i}")
            demo.attrs["num_samples"] = n_steps

            obs_grp = demo.create_group("obs")
            obs_grp.create_dataset(
                "robot0_joint_pos", data=rng.random((n_steps, 7)).astype(np.float32)
            )
            obs_grp.create_dataset(
                "robot0_joint_vel", data=rng.random((n_steps, 7)).astype(np.float32)
            )
            obs_grp.create_dataset(
                "robot0_eef_pos", data=rng.random((n_steps, 3)).astype(np.float32)
            )
            obs_grp.create_dataset(
                "robot0_eef_quat", data=rng.random((n_steps, 4)).astype(np.float32)
            )
            obs_grp.create_dataset(
                "robot0_gripper_qpos", data=rng.random((n_steps, 2)).astype(np.float32)
            )
            if with_images:
                obs_grp.create_dataset(
                    "agentview_image",
                    data=rng.integers(0, 255, (n_steps, 84, 84, 3), dtype=np.uint8),
                )

            demo.create_dataset(
                "actions", data=rng.random((n_steps, action_dim)).astype(np.float32)
            )
            demo.create_dataset("dones", data=np.zeros(n_steps, dtype=bool))
            demo.create_dataset("rewards", data=rng.random(n_steps).astype(np.float32))

        if with_mask:
            mask_grp = f.create_group("mask")
            keys = np.array([f"demo_{i}" for i in range(n_demos)], dtype="S20")
            mask_grp.create_dataset("train", data=keys[: max(1, n_demos - 1)])
            if n_demos > 1:
                mask_grp.create_dataset("valid", data=keys[-1:])


# ── detection ────────────────────────────────────────────────────────────────


class TestIsaacLabDetection:
    def test_can_read_true_for_isaac_lab_file(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p))
        assert IsaacLabReader.can_read(str(p))

    def test_can_read_false_for_plain_hdf5(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        p = tmp_path / "plain.hdf5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("actions", data=np.zeros((10, 7)))
        assert not IsaacLabReader.can_read(str(p))

    def test_can_read_false_for_non_hdf5(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n")
        assert not IsaacLabReader.can_read(str(p))

    def test_can_read_directory(self, tmp_path):
        _write_isaac_lab_file(str(tmp_path / "demo.hdf5"))
        assert IsaacLabReader.can_read(str(tmp_path))


# ── reading ──────────────────────────────────────────────────────────────────


class TestIsaacLabRead:
    def test_returns_episode_batch(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=3, n_steps=50)
        batch = IsaacLabReader().read(str(p))
        assert isinstance(batch, EpisodeBatch)

    def test_episode_count(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=5)
        batch = IsaacLabReader().read(str(p))
        assert batch.n_episodes == 5

    def test_step_count(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=3, n_steps=50)
        batch = IsaacLabReader().read(str(p))
        assert batch.n_samples == 3 * 50

    def test_format_name(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p))
        batch = IsaacLabReader().read(str(p))
        assert batch.format == "isaac_lab"

    def test_dataset_name_from_env_attr(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), env_name="PickAndPlace")
        batch = IsaacLabReader().read(str(p))
        assert batch.dataset_name == "PickAndPlace"

    def test_action_shape(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=2, n_steps=40, action_dim=8)
        batch = IsaacLabReader().read(str(p))
        for ep in batch.episodes:
            assert ep.actions.shape == (40, 8)

    def test_episode_ids_are_demo_keys(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=3)
        batch = IsaacLabReader().read(str(p))
        ids = [ep.metadata.episode_id for ep in batch.episodes]
        assert ids == ["demo_0", "demo_1", "demo_2"]

    def test_timestamps_synthesized_at_50hz(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=1, n_steps=10)
        batch = IsaacLabReader().read(str(p))
        ts = batch.episodes[0].timestamps
        assert ts.shape == (10,)
        np.testing.assert_allclose(ts[1] - ts[0], 0.02, atol=1e-6)

    def test_proprio_built_from_kinematic_obs(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=1, n_steps=20)
        batch = IsaacLabReader().read(str(p))
        ep = batch.episodes[0]
        assert "proprio" in ep.observations
        # joint_pos(7) + joint_vel(7) + eef_pos(3) + eef_quat(4) + gripper(2) = 23
        assert ep.observations["proprio"].shape == (20, 23)

    def test_image_observations_present(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=1, n_steps=10, with_images=True)
        batch = IsaacLabReader().read(str(p))
        ep = batch.episodes[0]
        assert "agentview_image" in ep.observations
        assert "camera_agentview_image" in ep.observations

    def test_success_from_mask(self, tmp_path):
        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p), n_demos=3, with_mask=True)
        batch = IsaacLabReader().read(str(p))
        # demo_0 and demo_1 are in train mask → success=True
        assert batch.episodes[0].metadata.success is True
        assert batch.episodes[1].metadata.success is True

    def test_read_directory(self, tmp_path):
        _write_isaac_lab_file(str(tmp_path / "run1.hdf5"), n_demos=2)
        _write_isaac_lab_file(str(tmp_path / "run2.hdf5"), n_demos=3)
        batch = IsaacLabReader().read(str(tmp_path))
        assert batch.n_episodes == 5

    def test_registry_auto_detects(self, tmp_path):
        """Auto-detection via load() picks IsaacLabReader over HDF5Reader."""
        from calibra.ingestion import registry

        p = tmp_path / "demo.hdf5"
        _write_isaac_lab_file(str(p))
        reader_cls = registry.detect_reader(str(p))
        assert reader_cls is IsaacLabReader
