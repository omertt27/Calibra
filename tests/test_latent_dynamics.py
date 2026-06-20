"""Tests for the LatentDynamicsAnalyzer and State Encoders."""
from __future__ import annotations

import numpy as np
import pytest

from calibra.analyzers.latent_dynamics import (
    JointStateEncoder,
    LatentDynamicsAnalyzer,
    _compute_entropy_2d,
    _compute_knn_density,
    _pca_project_kd,
    _compute_normalized_hsic,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_batch(
    n_eps: int = 3,
    n_steps: int = 50,
    state_dim: int = 8,
    action_dim: int = 6,
    collapsed: bool = False,
) -> EpisodeBatch:
    rng = np.random.default_rng(42)
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        if collapsed:
            # All steps are identical (static robot state)
            obs = {"proprio": np.zeros((n_steps, state_dim), dtype=np.float32)}
            acts = np.zeros((n_steps, action_dim), dtype=np.float32)
        else:
            # Normal varying trajectory
            obs = {"proprio": rng.normal(0, 1.0, (n_steps, state_dim)).astype(np.float32)}
            acts = rng.uniform(-1, 1, (n_steps, action_dim)).astype(np.float32)
            
        episodes.append(Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
            timestamps=ts,
            observations=obs,
            actions=acts,
        ))
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="latent_test",
        format="hdf5",
        source_path="/tmp/test.h5",
    )


# ── tests ────────────────────────────────────────────────────────────────────

class TestJointStateEncoder:
    def test_concatenates_multiple_keys(self):
        obs = {
            "joint_pos": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "gripper_state": np.array([[0.0], [1.0]]),
            "camera": np.array([[10, 20], [30, 40]]),
        }
        encoder = JointStateEncoder(["joint_pos", "gripper_state"])
        encoded = encoder.encode(obs)
        assert encoded.shape == (2, 3)
        assert np.allclose(encoded, [[1, 2, 0], [3, 4, 1]])

    def test_flattens_multi_dim_tensors(self):
        obs = {
            "state_3d": np.arange(12).reshape(2, 2, 3), # (T, 2, 3)
        }
        encoder = JointStateEncoder(["state_3d"])
        encoded = encoder.encode(obs)
        assert encoded.shape == (2, 6)
        assert np.allclose(encoded, np.arange(12).reshape(2, 6))

    def test_raises_on_missing_keys(self):
        obs = {"other": np.array([[1.0]])}
        encoder = JointStateEncoder(["missing"])
        with pytest.raises(ValueError, match="None of the select keys"):
            encoder.encode(obs)


class TestNumericalHelpers:
    def test_pca_project_kd(self):
        rng = np.random.default_rng(0)
        # Create 2D data embedded in 5D
        data_2d = rng.normal(0, 1.0, (10, 2))
        data_5d = np.zeros((10, 5))
        data_5d[:, :2] = data_2d
        
        proj, var_ratio = _pca_project_kd(data_5d, k=2)
        assert proj.shape == (10, 2)
        assert var_ratio > 0.99  # Top-2 variance explains everything

    def test_compute_entropy_2d(self):
        # Empty array
        assert _compute_entropy_2d(np.empty((0, 2))) == 0.0
        # Highly concentrated data
        data_flat = np.zeros((50, 2))
        assert _compute_entropy_2d(data_flat) == 0.0
        # Uniform spread data
        rng = np.random.default_rng(0)
        data_spread = rng.uniform(-1, 1, (100, 2))
        entropy = _compute_entropy_2d(data_spread, bins=5)
        assert entropy > 1.0

    def test_compute_knn_density(self):
        # Too few samples
        assert _compute_knn_density(np.zeros((3, 2)), k=5) == 0.0
        
        # Grid of coordinates
        points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        # Distance to 1st nearest neighbor is 1.0
        dens = _compute_knn_density(points, k=1)
        assert np.isclose(dens, 1.0)

    def test_compute_normalized_hsic(self):
        rng = np.random.default_rng(42)
        # 1. Independent data
        X_ind = rng.normal(0, 1, (100, 2))
        Y_ind = rng.normal(0, 1, (100, 2))
        hsic_ind = _compute_normalized_hsic(X_ind, Y_ind)
        assert hsic_ind < 0.15

        # 2. Highly dependent data
        X_dep = rng.normal(0, 1, (100, 2))
        Y_dep = X_dep ** 2 + rng.normal(0, 0.05, (100, 2))
        hsic_dep = _compute_normalized_hsic(X_dep, Y_dep)
        assert hsic_dep > 0.3
        assert hsic_dep > hsic_ind


class TestLatentDynamicsAnalyzer:
    def test_runs_successfully_on_standard_batch(self):
        batch = _make_batch()
        analyzer = LatentDynamicsAnalyzer()
        result = analyzer.analyze(batch)
        
        assert result.analyzer_name == "latent_dynamics"
        # 5 flags: latent_state_entropy, transition_redundancy, dynamics_predictability_r2, causal_action_effect_mi, outlier_transition_fraction
        assert len(result.flags) == 5
        
        # Verify metrics keys
        metrics = result.raw_metrics
        assert "state_space_entropy_2d" in metrics
        assert "dynamics_r2_predictability" in metrics
        assert "action_controllability_r2" in metrics
        assert "action_effect_mi" in metrics
        assert "state_knn_density" in metrics
        assert "state_redundancy" in metrics
        assert "transition_redundancy" in metrics
        assert "per_episode_exclusive_novelty" in metrics
        
        # Exclusive novelty maps to the episode keys in the batch
        novelty_dict = metrics["per_episode_exclusive_novelty"]
        assert len(novelty_dict) == 3
        assert "ep_0" in novelty_dict
        assert "ep_1" in novelty_dict
        assert "ep_2" in novelty_dict
        for v in novelty_dict.values():
            assert 0.0 <= v <= 1.0

    def test_handles_collapsed_data_gracefully(self):
        batch = _make_batch(collapsed=True)
        analyzer = LatentDynamicsAnalyzer()
        result = analyzer.analyze(batch)
        
        # State space entropy should trigger WARNING
        state_flags = [f for f in result.flags if f.metric == "latent_state_entropy"]
        assert len(state_flags) == 1
        assert state_flags[0].level == RiskLevel.WARNING
        
        # State redundancy should be extremely high (near 1.0)
        assert result.raw_metrics["state_redundancy"] > 0.95

    def test_skips_when_no_proprioception_found(self):
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep_0"),
            timestamps=np.array([0.0, 0.1]),
            observations={"images": np.zeros((2, 64, 64, 3))},
            actions=np.zeros((2, 3)),
        )
        batch = EpisodeBatch(
            episodes=[ep],
            dataset_name="no_proprio",
            format="hdf5",
            source_path="/tmp/test.h5",
        )
        analyzer = LatentDynamicsAnalyzer()
        result = analyzer.analyze(batch)
        assert "skipped" in result.raw_metrics
        assert len(result.flags) == 0
