"""
calibra.pruning — Coreset selection for robotic imitation learning datasets.

Addresses the "data saturation" bottleneck: large datasets are highly redundant.
Naively training on all collected episodes inflates GPU cost and can degrade
policy performance through noise and redundant mode reinforcement.

This module implements a two-stage pruning pipeline:

  Stage 1 — Quality Filtering
    Discard episodes that fail quality thresholds (high jerk, dropout, etc.)
    using per-episode metrics from the diagnostic pipeline.

  Stage 2 — Greedy Maximum-Coverage Diversity Selection
    From the quality-passing pool, select a mathematically optimal coreset of K
    episodes that maximizes the minimum pairwise behavioral distance (greedy
    k-center / farthest-point sampling). Episodes are represented by their
    action-space statistics so similar demonstrations are clustered and only the
    most representative sample from each cluster is retained.

Two selectors are provided:

  CoresetSelector          — exact greedy k-center. O(N × K). Suitable up to
                             ~50k episodes.
  ApproximateCoresetSelector — MiniBatch approximation. O(N × B) where B is the
                             batch_size (default 1 000). Handles 500k+ episodes.
                             Automatically used when N > 50 000 if --approximate
                             is passed on the CLI.

Usage
-----
    from calibra.pruning import CoresetSelector, ApproximateCoresetSelector
    from calibra.pipeline import Pipeline

    batch  = load(...)
    report = Pipeline().run(batch)

    selector = CoresetSelector(keep_fraction=0.3)
    result   = selector.select(batch, report)

    print(result.summary())
    # → Write result.keep_episode_ids to a file, then filter your dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import DiagnosticReport
from calibra.comparison.comparator import _extract_ep_data


# ── quality thresholds (conservative defaults) ────────────────────────────────

_DEFAULT_MAX_SPIKE_RATE    = 0.10   # 10% of steps have jerk spikes
_DEFAULT_MAX_VEL_DISC_RATE = 0.25   # 25% of steps are discontinuous
_DEFAULT_MAX_DROPOUT       = 0.10   # 10% of frames dropped
_DEFAULT_MIN_LDLJ          = -30.0  # catastrophically jerky
_DEFAULT_MIN_LENGTH        = 10     # fewer than 10 steps is noise


# ── result schema ─────────────────────────────────────────────────────────────

@dataclass
class PruningResult:
    """
    Output of CoresetSelector.select().

    Attributes
    ----------
    keep_episode_ids        : episode IDs to retain (pass these to your dataset filter).
    quality_fail_ids        : episode IDs removed in Stage 1 (quality threshold failures).
    diversity_pruned_ids    : episode IDs removed in Stage 2 (redundant under max-coverage).
    quality_scores          : per-episode composite quality score (lower = cleaner).
    diversity_scores        : per-episode min-distance-to-selected score after greedy selection.
    n_original              : total episodes before pruning.
    n_kept                  : episodes in the coreset.
    n_quality_failures      : episodes removed for quality.
    n_diversity_pruned      : episodes removed for redundancy.
    keep_fraction_actual    : actual fraction kept (may differ from requested if quality failures
                              reduce the quality-passing pool).
    method                  : always "quality_filter + greedy_max_coverage".
    """
    keep_episode_ids:     list[str]
    quality_fail_ids:     list[str]
    diversity_pruned_ids: list[str]
    quality_scores:       dict[str, float]
    diversity_scores:     dict[str, float]
    n_original:           int
    n_kept:               int
    n_quality_failures:   int
    n_diversity_pruned:   int
    keep_fraction_actual: float
    method:               str = "quality_filter + greedy_max_coverage"

    def summary(self) -> str:
        lines = [
            "━" * 56,
            "  CALIBRA PRUNING SUMMARY",
            "━" * 56,
            f"  Original episodes  : {self.n_original}",
            f"  Quality failures   : {self.n_quality_failures}  (removed in Stage 1)",
            f"  Diversity pruned   : {self.n_diversity_pruned}  (removed in Stage 2)",
            f"  Coreset size       : {self.n_kept}  ({self.keep_fraction_actual:.1%} of original)",
            f"  Method             : {self.method}",
            "─" * 56,
            "  To use: add --export-dataset <dir> to write a ready-to-train dataset.",
            "    calibra prune <path> --keep 0.3 --export-dataset ./coreset/",
            "    python train.py --dataset ./coreset/",
            "━" * 56,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "method":              self.method,
            "n_original":          self.n_original,
            "n_kept":              self.n_kept,
            "n_quality_failures":  self.n_quality_failures,
            "n_diversity_pruned":  self.n_diversity_pruned,
            "keep_fraction_actual": self.keep_fraction_actual,
            "keep_episode_ids":    self.keep_episode_ids,
            "quality_fail_ids":    self.quality_fail_ids,
            "diversity_pruned_ids": self.diversity_pruned_ids,
            "quality_scores":      self.quality_scores,
            "diversity_scores":    self.diversity_scores,
        }


# ── selector ──────────────────────────────────────────────────────────────────

@dataclass
class CoresetSelector:
    """
    Two-stage coreset selection engine.

    Parameters
    ----------
    keep_fraction : target fraction of episodes to keep. Applied after quality
                    filtering; the actual fraction of the original dataset will
                    be <= keep_fraction if quality failures reduce the pool.
    max_spike_rate : Stage 1 threshold — episodes with jerk spike rate above
                     this are removed regardless of diversity. Default 0.10.
    max_vel_disc_rate : Stage 1 threshold — velocity discontinuity rate. Default 0.25.
    max_dropout_fraction : Stage 1 threshold — frame dropout fraction. Default 0.10.
    min_ldlj : Stage 1 threshold — minimum LDLJ (more negative = worse). Default -30.0.
    min_length : Stage 1 threshold — minimum episode length in steps. Default 10.
    quality_only : skip Stage 2; return all quality-passing episodes without
                   diversity selection. Equivalent to keep_fraction=1.0 in Stage 2.
    diversity_weight : float in [0, 1] controlling the blend between action-space
                       diversity features (1.0) and quality-metric features (0.0).
                       Default 0.7 — primarily action diversity, quality as tie-breaker.
    """

    keep_fraction:        float = 0.5
    max_spike_rate:       float = _DEFAULT_MAX_SPIKE_RATE
    max_vel_disc_rate:    float = _DEFAULT_MAX_VEL_DISC_RATE
    max_dropout_fraction: float = _DEFAULT_MAX_DROPOUT
    min_ldlj:             float = _DEFAULT_MIN_LDLJ
    min_length:           int   = _DEFAULT_MIN_LENGTH
    quality_only:         bool  = False
    diversity_weight:     float = 0.7
    entropy_weight:       float = 0.0

    def select(
        self,
        batch: EpisodeBatch,
        report: DiagnosticReport,
    ) -> PruningResult:
        """
        Run the two-stage pruning pipeline.

        Parameters
        ----------
        batch  : the EpisodeBatch to prune (must correspond to report).
        report : pre-computed DiagnosticReport from Pipeline().run(batch).

        Returns
        -------
        PruningResult with keep/remove episode ID lists and diagnostic scores.
        """
        ep_data = _extract_ep_data(report)
        episodes = batch.episodes
        n = len(episodes)

        # ── Stage 1: quality filtering ────────────────────────────────────────
        quality_scores = _compute_quality_scores(episodes, ep_data)
        quality_fail_indices = _quality_filter(episodes, ep_data, self)
        quality_fail_set = set(quality_fail_indices)
        quality_pass_indices = [i for i in range(n) if i not in quality_fail_set]

        if not quality_pass_indices:
            # Everything failed quality — return empty coreset
            return PruningResult(
                keep_episode_ids=[],
                quality_fail_ids=[episodes[i].metadata.episode_id for i in range(n)],
                diversity_pruned_ids=[],
                quality_scores=quality_scores,
                diversity_scores={},
                n_original=n,
                n_kept=0,
                n_quality_failures=n,
                n_diversity_pruned=0,
                keep_fraction_actual=0.0,
            )

        # ── Stage 2: diversity selection ──────────────────────────────────────
        k = max(1, round(n * self.keep_fraction))

        if self.quality_only or k >= len(quality_pass_indices):
            # No diversity pruning needed — return all quality-passing episodes
            keep_indices = quality_pass_indices
            diversity_pruned_indices: list[int] = []
            diversity_scores: dict[str, float] = {}
        else:
            entropy_scores = _compute_entropy_scores(episodes) if self.entropy_weight > 0 else {}
            features = _build_feature_matrix(
                episodes, quality_pass_indices, ep_data,
                self.diversity_weight, entropy_scores, self.entropy_weight,
            )
            selected_local = _greedy_max_coverage(features, k)
            selected_global = [quality_pass_indices[i] for i in selected_local]
            selected_set = set(selected_global)

            keep_indices = selected_global
            diversity_pruned_indices = [
                i for i in quality_pass_indices if i not in selected_set
            ]

            # Compute diversity scores: min distance from each episode to selected set
            diversity_scores = _diversity_score_map(
                episodes, quality_pass_indices, features, selected_local
            )

        keep_ids       = [episodes[i].metadata.episode_id for i in keep_indices]
        fail_ids       = [episodes[i].metadata.episode_id for i in quality_fail_indices]
        div_pruned_ids = [episodes[i].metadata.episode_id for i in diversity_pruned_indices]

        return PruningResult(
            keep_episode_ids=keep_ids,
            quality_fail_ids=fail_ids,
            diversity_pruned_ids=div_pruned_ids,
            quality_scores=quality_scores,
            diversity_scores=diversity_scores,
            n_original=n,
            n_kept=len(keep_ids),
            n_quality_failures=len(fail_ids),
            n_diversity_pruned=len(div_pruned_ids),
            keep_fraction_actual=len(keep_ids) / max(n, 1),
        )


# ── Stage 1: quality scoring ──────────────────────────────────────────────────

def _compute_quality_scores(
    episodes: list[Episode],
    ep_data: dict[str, list],
) -> dict[str, float]:
    """
    Composite quality score per episode. Lower = cleaner.

    Weighted combination of normalised quality metrics. All weights sum to 1.
    Missing metrics contribute 0 to the composite (treated as clean).
    """
    spike_rates   = ep_data.get("per_episode_spike_rate", [])
    disc_rates    = ep_data.get("per_episode_vel_disc_rate", [])
    dropouts      = ep_data.get("per_episode_dropout_fraction", [])
    ldlj_values   = ep_data.get("per_episode_ldlj", [])

    scores: dict[str, float] = {}
    for i, ep in enumerate(episodes):
        s = 0.0

        spike = _safe_get(spike_rates, i)
        if spike is not None:
            s += 0.35 * min(spike / 0.10, 1.0)

        disc = _safe_get(disc_rates, i)
        if disc is not None:
            s += 0.35 * min(disc / 0.25, 1.0)

        drop = _safe_get(dropouts, i)
        if drop is not None:
            s += 0.20 * min(drop / 0.10, 1.0)

        ldlj = _safe_get(ldlj_values, i)
        if ldlj is not None:
            # Map LDLJ: -30 → 1.0 (worst), -3 → 0.0 (best)
            normalised = max(0.0, min(1.0, (ldlj - (-3.0)) / (-30.0 - (-3.0))))
            s += 0.10 * normalised

        scores[ep.metadata.episode_id] = round(s, 6)

    return scores


def _quality_filter(
    episodes: list[Episode],
    ep_data: dict[str, list],
    cfg: "CoresetSelector",
) -> list[int]:
    """Return indices of episodes that fail quality thresholds (to be removed)."""
    spike_rates = ep_data.get("per_episode_spike_rate", [])
    disc_rates  = ep_data.get("per_episode_vel_disc_rate", [])
    dropouts    = ep_data.get("per_episode_dropout_fraction", [])
    ldlj_values = ep_data.get("per_episode_ldlj", [])

    fail: list[int] = []
    for i, ep in enumerate(episodes):
        if ep.n_steps < cfg.min_length:
            fail.append(i)
            continue

        spike = _safe_get(spike_rates, i)
        if spike is not None and spike > cfg.max_spike_rate:
            fail.append(i)
            continue

        disc = _safe_get(disc_rates, i)
        if disc is not None and disc > cfg.max_vel_disc_rate:
            fail.append(i)
            continue

        drop = _safe_get(dropouts, i)
        if drop is not None and drop > cfg.max_dropout_fraction:
            fail.append(i)
            continue

        ldlj = _safe_get(ldlj_values, i)
        if ldlj is not None and ldlj < cfg.min_ldlj:
            fail.append(i)

    return fail


# ── Stage 2: behavioral feature extraction ────────────────────────────────────

def _compute_entropy_scores(episodes: list[Episode]) -> dict[int, float]:
    """Return {episode_index: trajectory_entropy} for all episodes."""
    from calibra.curation.entropy import compute_trajectory_entropy
    return {i: compute_trajectory_entropy(ep.actions) for i, ep in enumerate(episodes)}


def _build_feature_matrix(
    episodes: list[Episode],
    candidate_indices: list[int],
    ep_data: dict[str, list],
    diversity_weight: float,
    entropy_scores: dict[int, float] | None = None,
    entropy_weight: float = 0.0,
) -> np.ndarray:
    """
    Build a (len(candidate_indices), F) feature matrix for diversity selection.

    Features (normalised to [0, 1]):
      - Action-space statistics (mean + std per dimension) — behavioral diversity
      - Quality metrics (spike_rate, vel_disc_rate) — quality diversity tie-breaker
      - Episode length (normalised)

    diversity_weight controls the blend: 1.0 = action stats only; 0.0 = quality
    metrics only. Default 0.7.
    """
    spike_rates = ep_data.get("per_episode_spike_rate", [])
    disc_rates  = ep_data.get("per_episode_vel_disc_rate", [])
    lengths     = ep_data.get("per_episode_length", [])

    rows: list[np.ndarray] = []
    for i in candidate_indices:
        ep = episodes[i]
        acts = ep.actions
        if acts.ndim == 1:
            acts = acts[:, np.newaxis]

        # Behavioral: action mean and std per dim
        action_mean = np.mean(acts, axis=0)
        action_std  = np.std(acts, axis=0)
        action_feat = np.concatenate([action_mean, action_std])

        # Quality metrics as secondary features
        spike = _safe_get(spike_rates, i) or 0.0
        disc  = _safe_get(disc_rates, i) or 0.0
        length_raw = _safe_get(lengths, i) or float(ep.n_steps)
        quality_feat = np.array([spike, disc, length_raw / 1000.0])

        # Entropy feature: per-trajectory action diversity (bits/dim).
        entropy_feat = np.array([entropy_scores.get(i, 0.0) if entropy_scores else 0.0])

        # Blend: diversity_weight for action stats, entropy_weight for entropy,
        # remaining for quality metrics.
        q_scale = max(0.0, 1.0 - diversity_weight - entropy_weight)
        row = np.concatenate([
            action_feat  * diversity_weight,
            quality_feat * q_scale,
            entropy_feat * entropy_weight,
        ])
        rows.append(row)

    if not rows:
        return np.zeros((0, 1))

    mat = np.stack(rows, axis=0)

    # Normalise each feature to [0, 1] across the candidate set
    col_mins = mat.min(axis=0)
    col_maxs = mat.max(axis=0)
    scale = col_maxs - col_mins
    scale[scale == 0] = 1.0  # constant features contribute nothing
    return (mat - col_mins) / scale


def _greedy_max_coverage(features: np.ndarray, k: int) -> list[int]:
    """
    Greedy k-center (farthest-point sampling): select k indices from features
    that maximise the minimum pairwise distance.

    Time: O(N × K)  — suitable for N up to ~50k, K up to ~5k.
    For larger datasets, consider approximate methods (random projection + greedy).

    Seed: the episode farthest from the empirical centroid of the candidate set.
    """
    n = len(features)
    if k >= n:
        return list(range(n))

    centroid = features.mean(axis=0)
    dists_to_centroid = np.linalg.norm(features - centroid, axis=1)
    seed = int(np.argmax(dists_to_centroid))

    selected = [seed]
    # min_dists[i] = min distance from candidate i to any already-selected episode
    min_dists = np.linalg.norm(features - features[seed], axis=1).copy()
    min_dists[seed] = -np.inf  # already selected

    for _ in range(k - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        dists_to_new = np.linalg.norm(features - features[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[next_idx] = -np.inf

    return selected


def _diversity_score_map(
    episodes: list[Episode],
    candidate_indices: list[int],
    features: np.ndarray,
    selected_local: list[int],
) -> dict[str, float]:
    """Return {episode_id: min_distance_to_selected_set} for all candidates."""
    selected_set = set(selected_local)
    selected_feats = features[sorted(selected_set)]
    scores: dict[str, float] = {}

    for local_idx, global_idx in enumerate(candidate_indices):
        ep_id = episodes[global_idx].metadata.episode_id
        feat = features[local_idx]
        dists = np.linalg.norm(selected_feats - feat, axis=1)
        scores[ep_id] = float(np.min(dists))

    return scores


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_get(lst: list, i: int):
    """Return lst[i] if in bounds and not None, else None."""
    if lst and i < len(lst) and lst[i] is not None:
        return lst[i]
    return None


# ── approximate coreset selector ──────────────────────────────────────────────

@dataclass
class ApproximateCoresetSelector(CoresetSelector):
    """
    Two-stage coreset selector with approximate Stage 2 diversity selection.

    Replaces the exact greedy k-center (O(N × K)) with a MiniBatch tournament
    algorithm that runs in O(N × B / R) time, where B=batch_size and
    R = ⌈N / B⌉ rounds. Handles datasets of 500k+ episodes.

    Algorithm (Stage 2):
        1. Shuffle all quality-passing episodes.
        2. Split into batches of size B.
        3. Run exact greedy k-center within each batch → local candidates.
        4. Tournament merge: from each batch's top-B/R candidates, keep the one
           farthest from the current global selected set.
        5. Repeat until K episodes are selected.

    Accuracy trade-off:
        The approximate selector may miss globally optimal coverage but
        consistently selects diverse representatives. In practice, quality
        metrics of the coreset are indistinguishable from the exact selector
        at batch_size ≥ 500.

    Parameters
    ----------
    batch_size : number of episodes processed per round. Larger = more accurate
                 but slower. Default 1 000 is a good balance for up to 1M episodes.
    """

    batch_size: int = 1000

    def select(
        self,
        batch: EpisodeBatch,
        report: DiagnosticReport,
    ) -> PruningResult:
        ep_data = _extract_ep_data(report)
        episodes = batch.episodes
        n = len(episodes)

        # Stage 1 is identical to the exact selector
        quality_scores = _compute_quality_scores(episodes, ep_data)
        quality_fail_indices = _quality_filter(episodes, ep_data, self)
        quality_fail_set = set(quality_fail_indices)
        quality_pass_indices = [i for i in range(n) if i not in quality_fail_set]

        if not quality_pass_indices:
            return PruningResult(
                keep_episode_ids=[],
                quality_fail_ids=[episodes[i].metadata.episode_id for i in range(n)],
                diversity_pruned_ids=[],
                quality_scores=quality_scores,
                diversity_scores={},
                n_original=n,
                n_kept=0,
                n_quality_failures=n,
                n_diversity_pruned=0,
                keep_fraction_actual=0.0,
                method="quality_filter + approximate_minibatch_coverage",
            )

        k = max(1, round(n * self.keep_fraction))

        if self.quality_only or k >= len(quality_pass_indices):
            keep_indices = quality_pass_indices
            diversity_pruned_indices: list[int] = []
            diversity_scores: dict[str, float] = {}
        else:
            entropy_scores = _compute_entropy_scores(episodes) if self.entropy_weight > 0 else {}
            features = _build_feature_matrix(
                episodes, quality_pass_indices, ep_data,
                self.diversity_weight, entropy_scores, self.entropy_weight,
            )
            selected_local = _approximate_max_coverage(features, k, self.batch_size)
            selected_global = [quality_pass_indices[i] for i in selected_local]
            selected_set = set(selected_global)

            keep_indices = selected_global
            diversity_pruned_indices = [
                i for i in quality_pass_indices if i not in selected_set
            ]
            diversity_scores = _diversity_score_map(
                episodes, quality_pass_indices, features, selected_local
            )

        keep_ids       = [episodes[i].metadata.episode_id for i in keep_indices]
        fail_ids       = [episodes[i].metadata.episode_id for i in quality_fail_indices]
        div_pruned_ids = [episodes[i].metadata.episode_id for i in diversity_pruned_indices]

        return PruningResult(
            keep_episode_ids=keep_ids,
            quality_fail_ids=fail_ids,
            diversity_pruned_ids=div_pruned_ids,
            quality_scores=quality_scores,
            diversity_scores=diversity_scores,
            n_original=n,
            n_kept=len(keep_ids),
            n_quality_failures=len(fail_ids),
            n_diversity_pruned=len(div_pruned_ids),
            keep_fraction_actual=len(keep_ids) / max(n, 1),
            method="quality_filter + approximate_minibatch_coverage",
        )


def _approximate_max_coverage(
    features: np.ndarray,
    k: int,
    batch_size: int,
) -> list[int]:
    """
    MiniBatch approximate greedy k-center.

    Splits N candidates into batches of `batch_size`, runs exact greedy within
    each batch, then merges candidates via a tournament that greedily adds the
    episode farthest from the current selected set.

    Time: O(N × B) where B = batch_size. Suitable for N up to ~1M.
    """
    n = len(features)
    if k >= n:
        return list(range(n))

    rng = np.random.default_rng(seed=42)
    order = rng.permutation(n)

    # Per-batch greedy selection: keep ceil(k * batch_size / n) from each batch.
    # Minimum 1 candidate per batch, maximum k.
    candidates_per_batch = max(1, min(k, round(k * batch_size / n) + 1))

    batch_candidates: list[int] = []
    for start in range(0, n, batch_size):
        batch_indices = order[start : start + batch_size].tolist()
        if not batch_indices:
            continue
        batch_feat = features[batch_indices]
        n_select = min(candidates_per_batch, len(batch_indices))
        local_selected = _greedy_max_coverage(batch_feat, n_select)
        batch_candidates.extend(batch_indices[i] for i in local_selected)

    if not batch_candidates:
        return list(range(min(k, n)))

    # Tournament merge: run exact greedy on the reduced candidate pool.
    candidate_feat = features[batch_candidates]
    if len(batch_candidates) <= k:
        return batch_candidates

    local_selected = _greedy_max_coverage(candidate_feat, k)
    return [batch_candidates[i] for i in local_selected]
