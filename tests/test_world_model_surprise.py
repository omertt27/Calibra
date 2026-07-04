"""Tests for calibra.world_model.surprise — lightweight (non-torch) world-model curation."""

from __future__ import annotations

import numpy as np

from calibra.pipeline import Pipeline
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.world_model.surprise import (
    LatentEncoder,
    LinearLatentPredictor,
    compute_surprise_scores,
    curate_for_world_model,
)

_LENIENT_QUALITY = dict(
    max_spike_rate=1.0,
    max_vel_disc_rate=1.0,
    max_dropout_fraction=1.0,
    min_ldlj=-1000.0,
)


def _make_batch(n=20, novel_ids=(2, 9, 15), n_steps=40):
    rng = np.random.default_rng(0)
    episodes = []
    for i in range(n):
        scale = 3.0 if i in novel_ids else 1.0
        actions = rng.normal(0, scale, (n_steps, 4)).astype(np.float32)
        state = (rng.random((n_steps, 6)).astype(np.float32)) * scale
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=np.arange(n_steps) * 0.02,
                observations={"proprio": state},
                actions=actions,
            )
        )
    return EpisodeBatch(episodes=episodes, dataset_name="wm_test", format="hdf5", source_path="/tmp/wm")


# ── encoder / predictor ────────────────────────────────────────────────────────


def test_latent_encoder_pca_shape():
    X = np.random.default_rng(0).normal(size=(50, 8))
    enc = LatentEncoder(latent_dim=4, method="pca").fit(X)
    Z = enc.transform(X)
    assert Z.shape == (50, 4)


def test_latent_encoder_random_fallback_for_tiny_input():
    X = np.random.default_rng(0).normal(size=(1, 3))
    enc = LatentEncoder(latent_dim=4, method="pca").fit(X)
    Z = enc.transform(X)
    assert Z.shape == (1, 4)


def test_linear_latent_predictor_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    W = rng.normal(size=(3, 2))
    Y = X @ W
    pred = LinearLatentPredictor(ridge=1e-6).fit(X, Y)
    Y_hat = pred.predict(X)
    assert np.mean((Y_hat - Y) ** 2) < 1e-6


# ── compute_surprise_scores ────────────────────────────────────────────────────


def test_compute_surprise_scores_empty_for_small_batch():
    batch = _make_batch(n=2)
    assert compute_surprise_scores(batch) == {}


def test_compute_surprise_scores_range_and_coverage():
    batch = _make_batch(n=20)
    scores = compute_surprise_scores(batch)
    assert set(scores.keys()) == {ep.metadata.episode_id for ep in batch.episodes}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_compute_surprise_scores_flags_injected_novelty():
    novel_ids = {2, 9, 15}
    batch = _make_batch(n=20, novel_ids=tuple(novel_ids))
    scores = compute_surprise_scores(batch)
    ranked = sorted(scores, key=lambda eid: scores[eid], reverse=True)
    top_3 = set(ranked[:3])
    assert top_3 == {f"ep_{i}" for i in novel_ids}


# ── curate_for_world_model (decision rule) ─────────────────────────────────────


def test_curate_counts_sum_to_original():
    batch = _make_batch(n=20)
    report = Pipeline().run(batch)
    result = curate_for_world_model(batch, report, quality_kwargs=_LENIENT_QUALITY)
    assert result.n_quality_fail + result.n_kept + result.n_pruned_redundant == result.n_original


def test_curate_keeps_high_surprise_prunes_rest():
    novel_ids = {2, 9, 15}
    batch = _make_batch(n=20, novel_ids=tuple(novel_ids))
    report = Pipeline().run(batch)
    result = curate_for_world_model(batch, report, quality_kwargs=_LENIENT_QUALITY)

    assert result.n_quality_fail == 0
    assert set(result.keep_episode_ids) == {f"ep_{i}" for i in novel_ids}
    assert set(result.redundant_ids) == {
        f"ep_{i}" for i in range(20) if i not in novel_ids
    }


def test_curate_quality_fail_takes_priority_over_surprise():
    # An episode with a hard jerk spike should be pruned as a quality failure
    # even if its dynamics also happen to be surprising.
    batch = _make_batch(n=20, novel_ids=(2, 9, 15))
    batch.episodes[2].actions[10] += 500.0  # inject a hard spike into a "novel" episode
    report = Pipeline().run(batch)

    result = curate_for_world_model(batch, report)  # default (strict) quality thresholds
    assert "ep_2" in result.quality_fail_ids
    assert "ep_2" not in result.keep_episode_ids


def test_curate_top_novel_sorted_descending_with_reasons():
    batch = _make_batch(n=20, novel_ids=(2, 9, 15))
    report = Pipeline().run(batch)
    result = curate_for_world_model(batch, report, quality_kwargs=_LENIENT_QUALITY, top=2)

    assert len(result.top_novel) == 2
    surprises = [e.surprise for e in result.top_novel]
    assert surprises == sorted(surprises, reverse=True)
    assert all(e.reason for e in result.top_novel)


def test_summary_format_matches_expected_banner():
    batch = _make_batch(n=20, novel_ids=(2, 9, 15))
    report = Pipeline().run(batch)
    result = curate_for_world_model(batch, report, quality_kwargs=_LENIENT_QUALITY)
    text = result.summary()

    assert text.startswith("WORLD-MODEL CURATION SUMMARY")
    assert f"Original episodes: {result.n_original}" in text
    assert f"Quality failures: {result.n_quality_fail}" in text
    assert f"High-surprise kept: {result.n_kept}" in text
    assert f"Low-surprise pruned: {result.n_pruned_redundant}" in text
    assert "Top novel episodes:" in text
