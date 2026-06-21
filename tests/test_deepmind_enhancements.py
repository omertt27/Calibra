import tempfile
import numpy as np
import pytest
from pathlib import Path

from calibra.schema.episode import Episode, EpisodeMetadata, EpisodeBatch, LazyEpisodeList
from calibra.kinematics.checker import KinematicURDFChecker
from calibra.sim2real import analyze_gap
from calibra.pipeline import Pipeline
from calibra.benchmark import run_benchmark


def test_lazy_episode_list():
    called = []
    
    def loader_fn(idx):
        called.append(idx)
        return Episode(
            metadata=EpisodeMetadata(episode_id=str(idx)),
            timestamps=np.array([0.0, 0.1]),
            observations={"proprio": np.random.randn(2, 3)},
            actions=np.random.randn(2, 2),
        )

    lazy_list = LazyEpisodeList(loader_fn=loader_fn, length=10, cache_size=2)
    
    assert len(lazy_list) == 10
    
    # Access first episode
    ep0 = lazy_list[0]
    assert ep0.metadata.episode_id == "0"
    assert called == [0]
    
    # Access again, should hit cache
    ep0_cached = lazy_list[0]
    assert ep0_cached is ep0
    assert called == [0]  # No new call
    
    # Access ep1, ep2 to evict ep0
    _ = lazy_list[1]
    _ = lazy_list[2]
    
    # Access ep0 again, cache should have evicted it
    _ = lazy_list[0]
    assert called == [0, 1, 2, 0]


def test_kinematic_urdf_checker():
    # Create a temporary URDF file
    urdf_content = """<?xml version="1.0"?>
    <robot name="test_robot">
      <joint name="joint_1" type="revolute">
        <limit lower="-1.0" upper="1.0" velocity="2.0" effort="10.0"/>
      </joint>
      <joint name="joint_2" type="revolute">
        <limit lower="-2.0" upper="2.0" velocity="1.0" effort="5.0"/>
      </joint>
    </robot>
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_content)
        temp_urdf_path = f.name

    try:
        checker = KinematicURDFChecker(temp_urdf_path)
        assert "joint_1" in checker.joint_limits
        assert checker.joint_limits["joint_1"]["lower"] == -1.0
        assert checker.joint_limits["joint_2"]["velocity"] == 1.0

        # Normal episode
        ep_ok = Episode(
            metadata=EpisodeMetadata(episode_id="ok"),
            timestamps=np.array([0.0, 1.0, 2.0]),
            observations={"proprio": np.array([[0.0, 0.0], [0.5, 0.5], [0.8, 0.9]])},
            actions=np.zeros((3, 2)),
        )
        violations = checker.check_episode(ep_ok)
        assert len(violations) == 0

        # Episode violating position limit (joint_1 upper is 1.0, joint_2 upper is 2.0)
        ep_pos_fail = Episode(
            metadata=EpisodeMetadata(episode_id="pos_fail"),
            timestamps=np.array([0.0, 1.0, 2.0]),
            observations={"proprio": np.array([[0.0, 0.0], [1.5, 0.5], [0.8, 0.9]])},
            actions=np.zeros((3, 2)),
        )
        violations = checker.check_episode(ep_pos_fail)
        assert "joint_1" in violations
        assert violations["joint_1"][0][2].startswith("position_overflow")

        # Episode violating velocity limit (joint_2 limit is 1.0 rad/s)
        ep_vel_fail = Episode(
            metadata=EpisodeMetadata(episode_id="vel_fail"),
            timestamps=np.array([0.0, 0.5]),
            observations={"proprio": np.array([[0.0, 0.0], [0.0, 1.5]])},  # diff = 1.5, dt = 0.5 -> vel = 3.0
            actions=np.zeros((2, 2)),
        )
        violations = checker.check_episode(ep_vel_fail)
        assert "joint_2" in violations
        assert violations["joint_2"][0][2].startswith("velocity_exceeded")
    finally:
        Path(temp_urdf_path).unlink()


def test_sim2real_transition_and_visual_gaps():
    # Build a simulated episode batch and a real episode batch
    sim_cam = np.zeros((3, 4, 4, 3))
    sim_cam[:, :, :, 0] = 10.0  # R channel active

    sim_episodes = [
        Episode(
            metadata=EpisodeMetadata(episode_id="sim_0"),
            timestamps=np.array([0.0, 0.1, 0.2]),
            observations={
                "proprio": np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
                "camera_rgb": sim_cam,
            },
            actions=np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]]),
        )
    ]
    sim_batch = EpisodeBatch(
        episodes=sim_episodes,
        dataset_name="sim_ds",
        format="hdf5",
        source_path="sim.h5",
    )

    real_cam = np.zeros((3, 4, 4, 3))
    real_cam[:, :, :, 1] = 200.0  # G channel active (different color space/distribution)

    real_episodes = [
        Episode(
            metadata=EpisodeMetadata(episode_id="real_0"),
            timestamps=np.array([0.0, 0.1, 0.2]),
            observations={
                # Real has different physics: transition state moves slower
                "proprio": np.array([[0.0, 0.0], [0.02, 0.02], [0.04, 0.04]]),
                # Real has different camera appearance (visual gap)
                "camera_rgb": real_cam,
            },
            actions=np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]]),
        )
    ]
    real_batch = EpisodeBatch(
        episodes=real_episodes,
        dataset_name="real_ds",
        format="hdf5",
        source_path="real.h5",
    )

    pipeline = Pipeline()
    sim_report = pipeline.run(sim_batch)
    real_report = pipeline.run(real_batch)

    # Perform gap analysis
    res = analyze_gap(sim_report, real_report, sim_batch=sim_batch, real_batch=real_batch)
    
    assert "transition_dynamics_gap" in res["gaps"]
    assert "visual_domain_gap" in res["gaps"]
    assert res["gaps"]["transition_dynamics_gap"]["value"] > 0.0
    assert res["gaps"]["visual_domain_gap"]["value"] > 0.0


def test_benchmark_cli(capsys, monkeypatch):
    # Mock EpisodeBatch
    episodes = [
        Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
            timestamps=np.linspace(0, 2.0, 20),
            observations={"proprio": np.random.randn(20, 3)},
            actions=np.random.randn(20, 2),
        )
        for i in range(10)
    ]
    mock_batch = EpisodeBatch(
        episodes=episodes,
        dataset_name="mock_dataset",
        format="lerobot",
        source_path="mock_path",
    )

    # Mock load function to return mock_batch
    import calibra.ingestion.registry as registry
    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)

    # Run benchmark command
    run_benchmark(["mock_path", "--keep", "0.3", "--policy", "diffusion"])
    captured = capsys.readouterr()
    
    assert "RAW DATASET" in captured.out
    assert "RANDOM PRUNED" in captured.out
    assert "CALIBRA CORESET" in captured.out
    assert "Compute Cost Savings:" in captured.out
    assert "RECOMMENDED TRAINING COMMAND BRIDGES" in captured.out
