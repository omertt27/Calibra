import sys
import numpy as np
import pytest
from pathlib import Path

from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.pipeline import Pipeline
from calibra.cure import smooth_actions, interpolate_episode, trim_dead_time, run_cure
from calibra.sim2real import analyze_gap
from calibra.curation.latent_embed import extract_latent_embeddings


def _make_mock_episode(n_steps=50, has_jitter=False, has_dead_time=False):
    rng = np.random.default_rng(42)
    
    # Timestamps
    if has_jitter:
        timestamps = np.cumsum(rng.uniform(0.02, 0.08, size=n_steps))
    else:
        timestamps = np.arange(n_steps, dtype=np.float64) * 0.05
        
    # Actions
    actions = rng.normal(0, 0.1, size=(n_steps, 3)).astype(np.float32)
    
    # Inject a large jerk spike
    if n_steps > 10:
        actions[10] = actions[10] + 5.0
        
    if has_dead_time and n_steps > 20:
        # Start and end have zero action difference
        actions[:5] = 0.0
        actions[-5:] = 0.0
        
    obs = {
        "camera_rgb": (rng.random((n_steps, 16, 16, 3)) * 255).astype(np.uint8),
        "proprio": rng.random((n_steps, 3)).astype(np.float32),
    }
    
    return Episode(
        metadata=EpisodeMetadata(episode_id="test_ep"),
        timestamps=timestamps,
        observations=obs,
        actions=actions,
    )


def test_action_smoothing():
    ep = _make_mock_episode()
    smoothed = smooth_actions(ep.actions, window_len=5, polyorder=2)
    assert smoothed.shape == ep.actions.shape
    # Jerk should be lower on average
    orig_jerk = np.mean(np.abs(np.diff(ep.actions, n=2, axis=0)))
    smooth_jerk = np.mean(np.abs(np.diff(smoothed, n=2, axis=0)))
    assert smooth_jerk < orig_jerk


def test_interpolation():
    ep = _make_mock_episode(has_jitter=True)
    t, act, obs = interpolate_episode(ep.timestamps, ep.actions, ep.observations, target_hz=20.0)
    
    # Check that timestamps are perfectly uniform
    diffs = np.diff(t)
    assert np.allclose(diffs, diffs[0], atol=1e-5)
    assert act.shape[0] == len(t)
    assert obs["camera_rgb"].shape[0] == len(t)
    assert obs["proprio"].shape[0] == len(t)


def test_dead_time_trimming():
    ep = _make_mock_episode(has_dead_time=True)
    t, act, obs = trim_dead_time(ep.timestamps, ep.actions, ep.observations, threshold=0.1)
    
    assert len(t) < len(ep.timestamps)
    assert act.shape[0] == len(t)
    assert obs["camera_rgb"].shape[0] == len(t)


def test_pretraining_alignment_index():
    ep1 = _make_mock_episode()
    ep2 = _make_mock_episode()
    
    batch1 = EpisodeBatch(episodes=[ep1], dataset_name="sim", format="hdf5", source_path="/tmp/sim.h5")
    batch2 = EpisodeBatch(episodes=[ep2], dataset_name="real", format="hdf5", source_path="/tmp/real.h5")
    
    pipeline = Pipeline()
    report1 = pipeline.run(batch1)
    report2 = pipeline.run(batch2)
    
    result = analyze_gap(report1, report2, batch1, batch2)
    assert "pretraining_alignment_index" in result
    # For identical mock datasets, PAI should be very high
    assert result["pretraining_alignment_index"] > 80.0


def test_latent_embeddings_clip_fallback():
    ep = _make_mock_episode()
    batch = EpisodeBatch(episodes=[ep], dataset_name="test", format="hdf5", source_path="/tmp/test.h5")
    
    # Should run clip/vlm type and gracefully fall back to resnet/visual if transformers is not present
    embs = extract_latent_embeddings(batch, model_type="clip")
    assert ep.metadata.episode_id in embs
    assert embs[ep.metadata.episode_id].ndim == 1


def test_cure_cli_manifest(tmp_path):
    h5py = pytest.importorskip("h5py")
    ep = _make_mock_episode()
    # Save a mock HDF5 file
    h5_file = tmp_path / "mock.h5"
    with h5py.File(h5_file, "w") as f:
        g = f.create_group("episode_0")
        g.create_dataset("timestamps", data=ep.timestamps)
        g.create_dataset("actions", data=ep.actions)
        obs_g = g.create_group("observations")
        obs_g.create_dataset("camera_rgb", data=ep.observations["camera_rgb"])
        obs_g.create_dataset("proprio", data=ep.observations["proprio"])
        
    out_dir = tmp_path / "cured"
    run_cure([str(h5_file), "--out", str(out_dir), "--remedy", "smooth,trim"])
    
    manifest_path = out_dir / "cure_manifest.json"
    assert manifest_path.exists()
    
    # Verify we can find npz files
    npz_files = list(out_dir.glob("*.npz"))
    assert len(npz_files) > 0
