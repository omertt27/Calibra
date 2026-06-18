"""
calibra.curation.entropy — Trajectory-level Shannon entropy for coreset selection.

Procedural Isaac Lab datasets suffer from high trajectory redundancy: a robot
arm repeating the same grasp path 10,000 times with sub-millimeter variation
yields effectively one unique trajectory. Training a behaviour-cloning policy on
this data saturates performance early, wastes GPU time proportional to the
redundant volume, and produces policies that fail on out-of-distribution starts.

This module scores each episode by the information-theoretic richness of its
action sequence. Episodes with low entropy are near-duplicates of the dataset
mode; episodes with high entropy cover rare, diverse behaviour.

`rank_by_entropy` returns episodes sorted by richness so that `calibra prune`
can preferentially retain the diverse tail when down-sampling to a coreset.

Integration with CoresetSelector
---------------------------------
Set `entropy_weight > 0` in `CoresetSelector` to blend per-trajectory entropy
into the feature matrix used by the greedy max-coverage algorithm. At weight 1.0
the selector becomes a pure entropy-based sampler; at 0.0 it falls back to the
default action-mean/std + quality features.

Recommended for GR00T fine-tuning: `entropy_weight = 0.4`.

Note
----
`compute_trajectory_entropy` applies the same Shannon-entropy formula as
`calibra.metrics.kinematics.compute_action_entropy` but scoped to a *single*
trajectory rather than the full dataset. The per-trajectory view is what matters
for coreset selection: we want to keep the episodes with the most internal
variety, not just the episodes whose mean action is far from the dataset mean.
"""
from __future__ import annotations

import numpy as np

from calibra.schema.episode import EpisodeBatch


def compute_trajectory_entropy(actions: np.ndarray, num_bins: int = 20) -> float:
    """
    Mean per-dimension Shannon entropy of a single trajectory's action sequence.

    Higher entropy → more diverse actions within this episode → more
    informational richness → higher priority for retention in a coreset.

    Parameters
    ----------
    actions  : np.ndarray, shape (T, D) or (T,)
        Action sequence for a single episode.
    num_bins : int
        Number of histogram bins per action dimension. Default 20. Larger
        values give finer resolution but require more steps to be reliable.
        A heuristic minimum: T ≥ 3 × num_bins for stable entropy estimates.

    Returns
    -------
    float — mean Shannon entropy in bits per action dimension. Range [0, log2(num_bins)].

    Notes
    -----
    The entropy is computed per dimension independently (marginal entropy, not
    joint). This is fast and sufficient for detecting near-duplicate trajectories
    where all dimensions are simultaneously low-entropy.
    """
    acts = np.asarray(actions, dtype=np.float32)
    if acts.ndim == 1:
        acts = acts[:, np.newaxis]

    _T, D = acts.shape
    entropies: list[float] = []

    for d in range(D):
        col = acts[:, d]
        hist, _ = np.histogram(col, bins=num_bins)
        total = hist.sum()
        if total == 0:
            entropies.append(0.0)
            continue
        probs = hist / total
        probs = probs[probs > 0]
        entropies.append(float(-np.sum(probs * np.log2(probs))))

    return float(np.mean(entropies)) if entropies else 0.0


def score_batch_entropy(
    batch: EpisodeBatch,
    num_bins: int = 20,
) -> dict[str, float]:
    """
    Compute per-episode trajectory entropy for all episodes in a batch.

    Parameters
    ----------
    batch    : EpisodeBatch to score.
    num_bins : histogram bin count passed to compute_trajectory_entropy.

    Returns
    -------
    dict mapping episode_id → entropy score (bits/dim).
    """
    return {
        ep.metadata.episode_id: compute_trajectory_entropy(ep.actions, num_bins=num_bins)
        for ep in batch.episodes
    }


def rank_by_entropy(
    batch: EpisodeBatch,
    num_bins: int = 20,
    descending: bool = True,
) -> list[tuple[str, float]]:
    """
    Rank episodes by trajectory entropy.

    Parameters
    ----------
    batch      : EpisodeBatch to rank.
    num_bins   : histogram bin count.
    descending : True (default) returns highest-entropy (most diverse) episodes
                 first. Set False to surface the most redundant episodes first
                 (useful for identifying which episodes to drop).

    Returns
    -------
    List of (episode_id, entropy_bits_per_dim) tuples, sorted by entropy.
    """
    scores = score_batch_entropy(batch, num_bins=num_bins)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=descending)
