"""
Tests for calibra.corrupt — corruption transforms and CLI rendering.
"""

from __future__ import annotations

import numpy as np

from calibra.corrupt import (
    CorruptionConfig,
    apply_corruptions,
    render_corruption_report,
    _drop_frames,
    _add_jitter,
    _inject_spikes,
    _delay_episode,
    _truncate_episode,
    _copy_episode,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata


# ── fixtures ──────────────────────────────────────────────────────────────────


def _episode(n_steps: int = 60, action_dim: int = 4, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, float(n_steps - 1) / 10.0, n_steps)
    return Episode(
        metadata=EpisodeMetadata(episode_id="0"),
        timestamps=ts,
        observations={"state": rng.standard_normal((n_steps, 7)).astype(np.float32)},
        actions=rng.standard_normal((n_steps, action_dim)).astype(np.float32),
    )


def _batch(n_episodes: int = 8, n_steps: int = 60) -> EpisodeBatch:
    return EpisodeBatch(
        episodes=[_episode(n_steps, seed=i) for i in range(n_episodes)],
        dataset_name="test",
        format="hdf5",
        source_path="/tmp/test",
    )


# ── CorruptionConfig ──────────────────────────────────────────────────────────


def test_config_is_empty_when_no_flags():
    assert CorruptionConfig().is_empty()


def test_config_not_empty_with_any_flag():
    assert not CorruptionConfig(drop_frames=0.1).is_empty()
    assert not CorruptionConfig(add_jitter_ms=10.0).is_empty()
    assert not CorruptionConfig(inject_spikes=0.05).is_empty()
    assert not CorruptionConfig(delay_episode=0.2).is_empty()
    assert not CorruptionConfig(truncate_episodes=0.3).is_empty()


def test_config_describe_lists_active():
    cfg = CorruptionConfig(drop_frames=0.10, add_jitter_ms=20.0)
    desc = cfg.describe()
    assert any("drop_frames" in d for d in desc)
    assert any("add_jitter_ms" in d for d in desc)
    assert not any("inject_spikes" in d for d in desc)


# ── individual transforms ─────────────────────────────────────────────────────


def test_drop_frames_reduces_steps():
    rng = np.random.default_rng(42)
    ep = _episode(100)
    result = _drop_frames(_copy_episode(ep), 0.20, rng)
    assert result.n_steps < ep.n_steps
    assert result.n_steps >= 2


def test_drop_frames_keeps_arrays_aligned():
    rng = np.random.default_rng(42)
    ep = _episode(100)
    result = _drop_frames(_copy_episode(ep), 0.15, rng)
    n = result.n_steps
    assert result.actions.shape[0] == n
    for v in result.observations.values():
        assert v.shape[0] == n


def test_drop_frames_timestamps_monotone():
    rng = np.random.default_rng(42)
    ep = _episode(100)
    result = _drop_frames(_copy_episode(ep), 0.20, rng)
    assert np.all(np.diff(result.timestamps) >= 0)


def test_add_jitter_preserves_length():
    rng = np.random.default_rng(42)
    ep = _episode(50)
    result = _add_jitter(_copy_episode(ep), 30.0, rng)
    assert result.n_steps == ep.n_steps


def test_add_jitter_timestamps_still_sorted():
    rng = np.random.default_rng(42)
    ep = _episode(50)
    result = _add_jitter(_copy_episode(ep), 50.0, rng)
    assert np.all(np.diff(result.timestamps) >= 0)


def test_inject_spikes_preserves_length():
    rng = np.random.default_rng(42)
    ep = _episode(80)
    result = _inject_spikes(_copy_episode(ep), 0.10, rng)
    assert result.n_steps == ep.n_steps


def test_inject_spikes_modifies_actions():
    rng = np.random.default_rng(42)
    ep = _episode(80)
    original_actions = ep.actions.copy()
    result = _inject_spikes(_copy_episode(ep), 0.10, rng)
    # At least some actions should differ
    assert not np.allclose(result.actions, original_actions)


def test_delay_episode_shifts_timestamps():
    rng = np.random.default_rng(42)
    ep = _episode(40)
    original_start = ep.timestamps[0]
    result = _delay_episode(_copy_episode(ep), rng)
    assert result.timestamps[0] > original_start


def test_truncate_episode_shortens():
    ep = _episode(100)
    result = _truncate_episode(_copy_episode(ep))
    assert result.n_steps < ep.n_steps
    assert result.n_steps >= 2


def test_truncate_episode_keeps_arrays_aligned():
    ep = _episode(100)
    result = _truncate_episode(_copy_episode(ep))
    n = result.n_steps
    assert result.actions.shape[0] == n
    for v in result.observations.values():
        assert v.shape[0] == n


# ── apply_corruptions ─────────────────────────────────────────────────────────


def test_apply_corruptions_does_not_modify_original():
    batch = _batch()
    original_n = batch.n_samples
    cfg = CorruptionConfig(drop_frames=0.10, inject_spikes=0.05)
    _ = apply_corruptions(batch, cfg)
    assert batch.n_samples == original_n


def test_apply_corruptions_drop_frames_reduces_total_steps():
    batch = _batch(n_episodes=10, n_steps=80)
    cfg = CorruptionConfig(drop_frames=0.20)
    corrupted = apply_corruptions(batch, cfg)
    assert corrupted.n_samples < batch.n_samples


def test_apply_corruptions_dataset_name_tagged():
    batch = _batch()
    cfg = CorruptionConfig(add_jitter_ms=10.0)
    corrupted = apply_corruptions(batch, cfg)
    assert "corrupted" in corrupted.dataset_name.lower()


def test_apply_corruptions_all_flags_compose():
    batch = _batch(n_episodes=12, n_steps=60)
    cfg = CorruptionConfig(
        drop_frames=0.05,
        add_jitter_ms=10.0,
        inject_spikes=0.03,
        delay_episode=0.2,
        truncate_episodes=0.2,
        seed=7,
    )
    corrupted = apply_corruptions(batch, cfg)
    assert corrupted.n_episodes == batch.n_episodes
    assert corrupted.n_samples <= batch.n_samples


def test_apply_corruptions_reproducible_with_seed():
    batch = _batch()
    cfg = CorruptionConfig(drop_frames=0.10, inject_spikes=0.05, seed=99)
    c1 = apply_corruptions(batch, cfg)
    c2 = apply_corruptions(batch, cfg)
    assert c1.n_samples == c2.n_samples
    for e1, e2 in zip(c1.episodes, c2.episodes):
        assert np.allclose(e1.timestamps, e2.timestamps)


# ── render_corruption_report ──────────────────────────────────────────────────


def _dummy_metrics(base: float = 0.0) -> dict:
    return {
        "jitter_cv": base + 0.01,
        "dropout_rate": base + 0.02,
        "spike_rate": base + 0.03,
        "vel_disc_rate": base + 0.04,
        "ldlj": -(5.0 + base),
        "action_entropy": 4.5 - base,
    }


def test_render_contains_header():
    cfg = CorruptionConfig(drop_frames=0.10)
    output = render_corruption_report(
        "my_dataset",
        cfg,
        _dummy_metrics(0),
        _dummy_metrics(0.05),
        10,
        10,
    )
    assert "calibra corrupt" in output
    assert "drop_frames" in output


def test_render_contains_all_metric_labels():
    cfg = CorruptionConfig(inject_spikes=0.05)
    output = render_corruption_report(
        "ds",
        cfg,
        _dummy_metrics(),
        _dummy_metrics(0.1),
        5,
        5,
    )
    assert "Jerk spike rate" in output
    assert "Timestamp dropout" in output
    assert "Action entropy" in output


def test_render_with_none_metrics():
    cfg = CorruptionConfig(drop_frames=0.05)
    metrics = {
        k: None
        for k in [
            "jitter_cv",
            "dropout_rate",
            "spike_rate",
            "vel_disc_rate",
            "ldlj",
            "action_entropy",
        ]
    }
    output = render_corruption_report("ds", cfg, metrics, metrics, 5, 5)
    assert "n/a" in output
