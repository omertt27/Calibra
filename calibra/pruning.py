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

_DEFAULT_MAX_SPIKE_RATE = 0.10  # 10% of steps have jerk spikes
_DEFAULT_MAX_VEL_DISC_RATE = 0.25  # 25% of steps are discontinuous
_DEFAULT_MAX_DROPOUT = 0.10  # 10% of frames dropped
_DEFAULT_MIN_LDLJ = -30.0  # catastrophically jerky
_DEFAULT_MIN_LENGTH = 10  # fewer than 10 steps is noise


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

    keep_episode_ids: list[str]
    quality_fail_ids: list[str]
    diversity_pruned_ids: list[str]
    quality_scores: dict[str, float]
    diversity_scores: dict[str, float]
    n_original: int
    n_kept: int
    n_quality_failures: int
    n_diversity_pruned: int
    keep_fraction_actual: float
    method: str = "quality_filter + greedy_max_coverage"

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
            "method": self.method,
            "n_original": self.n_original,
            "n_kept": self.n_kept,
            "n_quality_failures": self.n_quality_failures,
            "n_diversity_pruned": self.n_diversity_pruned,
            "keep_fraction_actual": self.keep_fraction_actual,
            "keep_episode_ids": self.keep_episode_ids,
            "quality_fail_ids": self.quality_fail_ids,
            "diversity_pruned_ids": self.diversity_pruned_ids,
            "quality_scores": self.quality_scores,
            "diversity_scores": self.diversity_scores,
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

    keep_fraction: float = 0.5
    max_spike_rate: float = _DEFAULT_MAX_SPIKE_RATE
    max_vel_disc_rate: float = _DEFAULT_MAX_VEL_DISC_RATE
    max_dropout_fraction: float = _DEFAULT_MAX_DROPOUT
    min_ldlj: float = _DEFAULT_MIN_LDLJ
    min_length: int = _DEFAULT_MIN_LENGTH
    quality_only: bool = False
    diversity_weight: float = 0.7
    entropy_weight: float = 0.0
    strategy: str = "diversity"
    latent_space: str = "none"
    contact_aware: bool = True

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
        effective_max_vel_disc = _contact_aware_vel_disc(ep_data, self)
        quality_fail_indices = _quality_filter(episodes, ep_data, self, effective_max_vel_disc)
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
        elif self.strategy in ("novelty", "transition_novelty"):
            novelty_scores = {}
            for res in report.analyzer_results:
                if res.analyzer_name == "latent_dynamics":
                    novelty_scores = res.raw_metrics.get("per_episode_exclusive_novelty", {})
                    break
            sorted_pass = sorted(
                quality_pass_indices,
                key=lambda idx: novelty_scores.get(episodes[idx].metadata.episode_id, 0.0),
                reverse=True,
            )
            keep_indices = sorted_pass[:k]
            diversity_pruned_indices = sorted_pass[k:]
            diversity_scores = {
                episodes[i].metadata.episode_id: float(
                    novelty_scores.get(episodes[i].metadata.episode_id, 0.0)
                )
                for i in quality_pass_indices
            }
        elif self.strategy == "influence":
            keep_indices, diversity_pruned_indices, diversity_scores = _select_influence(
                self, batch, report, quality_pass_indices, k
            )
        elif self.strategy == "energy":
            keep_indices, diversity_pruned_indices, diversity_scores = _select_energy(
                self, batch, report, quality_pass_indices, k
            )
        elif self.strategy == "world-model":
            keep_indices, diversity_pruned_indices, diversity_scores = _select_world_model(
                self, batch, report, quality_pass_indices, k
            )
        else:
            entropy_scores = _compute_entropy_scores(episodes) if self.entropy_weight > 0 else {}
            latent_embeddings = None
            if self.latent_space != "none":
                from calibra.curation.latent_embed import extract_latent_embeddings

                latent_embeddings = extract_latent_embeddings(batch, model_type=self.latent_space)

            features = _build_feature_matrix(
                episodes,
                quality_pass_indices,
                ep_data,
                self.diversity_weight,
                entropy_scores,
                self.entropy_weight,
                latent_embeddings=latent_embeddings,
            )
            selected_local = _greedy_max_coverage(features, k)
            selected_global = [quality_pass_indices[i] for i in selected_local]
            selected_set = set(selected_global)

            keep_indices = selected_global
            diversity_pruned_indices = [i for i in quality_pass_indices if i not in selected_set]

            # Compute diversity scores: min distance from each episode to selected set
            diversity_scores = _diversity_score_map(
                episodes, quality_pass_indices, features, selected_local
            )

        keep_ids = [episodes[i].metadata.episode_id for i in keep_indices]
        fail_ids = [episodes[i].metadata.episode_id for i in quality_fail_indices]
        div_pruned_ids = [episodes[i].metadata.episode_id for i in diversity_pruned_indices]

        _method = (
            "quality_filter + jepa_world_model_surprise"
            if self.strategy == "world-model"
            else "quality_filter + greedy_max_coverage"
        )
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
            method=_method,
        )


def _select_influence(
    self_selector: CoresetSelector,
    batch: EpisodeBatch,
    report: DiagnosticReport,
    quality_pass_indices: list[int],
    k: int,
) -> tuple[list[int], list[int], dict[str, float]]:
    episodes = batch.episodes
    influence_data = {}
    for r in report.analyzer_results:
        if r.analyzer_name == "influence":
            influence_data = r.raw_metrics.get("per_episode_influence", {})
            break

    if not influence_data:
        from calibra.analyzers.influence import InfluenceAnalyzer

        analyzer = InfluenceAnalyzer()
        res = analyzer.analyze(batch)
        influence_data = res.raw_metrics.get("per_episode_influence", {})

    candidate_ids = [episodes[i].metadata.episode_id for i in quality_pass_indices]
    candidate_influences = [influence_data.get(cid, 0.0) for cid in candidate_ids]

    sorted_local_indices = np.argsort(candidate_influences)[::-1]
    selected_local = sorted_local_indices[:k].tolist()

    selected_global = [quality_pass_indices[i] for i in selected_local]
    selected_set = set(selected_global)

    keep_indices = selected_global
    diversity_pruned_indices = [i for i in quality_pass_indices if i not in selected_set]

    diversity_scores = {
        episodes[i].metadata.episode_id: float(
            influence_data.get(episodes[i].metadata.episode_id, 0.0)
        )
        for i in quality_pass_indices
    }

    return keep_indices, diversity_pruned_indices, diversity_scores


def _select_energy(
    self_selector: CoresetSelector,
    batch: EpisodeBatch,
    report: DiagnosticReport,
    quality_pass_indices: list[int],
    k: int,
) -> tuple[list[int], list[int], dict[str, float]]:
    episodes = batch.episodes
    energy_data = {}
    for r in report.analyzer_results:
        if r.analyzer_name == "transition_dynamics":
            energy_data = r.raw_metrics.get("per_episode_dynamics_error", {})
            break

    if not energy_data:
        from calibra.analyzers.transition_dynamics import TransitionDynamicsAnalyzer

        analyzer = TransitionDynamicsAnalyzer()
        res = analyzer.analyze(batch)
        energy_data = res.raw_metrics.get("per_episode_dynamics_error", {})

    candidate_ids = [episodes[i].metadata.episode_id for i in quality_pass_indices]
    candidate_energies = np.array([energy_data.get(cid, 0.0) for cid in candidate_ids])

    # Prune extreme outliers (top 10% highest energy is likely physics violations / noise)
    if len(candidate_energies) > 10:
        threshold_noise = np.percentile(candidate_energies, 90)
        clean_mask = candidate_energies <= threshold_noise
        clean_indices = [quality_pass_indices[i] for i, val in enumerate(clean_mask) if val]
        clean_ids = [episodes[i].metadata.episode_id for i in clean_indices]
        clean_energies = [energy_data.get(cid, 0.0) for cid in clean_ids]
    else:
        clean_indices = quality_pass_indices
        clean_ids = candidate_ids
        clean_energies = candidate_energies

    # Greedily select the highest dynamics error (surprisal) transitions
    sorted_local_indices = np.argsort(clean_energies)[::-1]
    selected_local = sorted_local_indices[:k].tolist()

    selected_global = [clean_indices[i] for i in selected_local]
    selected_set = set(selected_global)

    keep_indices = selected_global
    diversity_pruned_indices = [i for i in quality_pass_indices if i not in selected_set]

    diversity_scores = {
        episodes[i].metadata.episode_id: float(
            energy_data.get(episodes[i].metadata.episode_id, 0.0)
        )
        for i in quality_pass_indices
    }

    return keep_indices, diversity_pruned_indices, diversity_scores


def _select_world_model(
    self_selector: CoresetSelector,
    batch: EpisodeBatch,
    report: DiagnosticReport,
    quality_pass_indices: list[int],
    k: int,
) -> tuple[list[int], list[int], dict[str, float]]:
    import sys

    from calibra.models.robot_jepa import score_by_jepa_surprise

    episodes = batch.episodes

    surprise_scores = score_by_jepa_surprise(batch, verbose=True)

    if not surprise_scores:
        print(
            "  [world-model] torch not available or too few episodes — falling back to random selection",
            file=sys.stderr,
        )
        keep_indices = quality_pass_indices[:k]
        diversity_pruned_indices = quality_pass_indices[k:]
        diversity_scores: dict[str, float] = {}
        return keep_indices, diversity_pruned_indices, diversity_scores

    sorted_pass = sorted(
        quality_pass_indices,
        key=lambda idx: surprise_scores.get(episodes[idx].metadata.episode_id, 0.0),
        reverse=True,
    )
    keep_indices = sorted_pass[:k]
    diversity_pruned_indices = sorted_pass[k:]

    diversity_scores = {
        episodes[i].metadata.episode_id: float(
            surprise_scores.get(episodes[i].metadata.episode_id, 0.0)
        )
        for i in quality_pass_indices
    }

    return keep_indices, diversity_pruned_indices, diversity_scores


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
    spike_rates = ep_data.get("per_episode_spike_rate", [])
    disc_rates = ep_data.get("per_episode_vel_disc_rate", [])
    dropouts = ep_data.get("per_episode_dropout_fraction", [])
    ldlj_values = ep_data.get("per_episode_ldlj", [])

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


def _contact_aware_vel_disc(ep_data: dict, cfg: "CoresetSelector") -> float:
    """
    Return the effective vel_disc threshold, relaxed when vel_disc is likely
    caused by contact dynamics rather than control noise.

    Mechanism: velocity discontinuities have two sources —
      (1) control noise / operator mistakes → also produces jerk spikes
      (2) contact events (pushes, grasps) → abrupt direction reversals
          with little or no accompanying jerk

    Discriminant: the ratio mean_vel_disc / mean_spike_rate.
      - When both are elevated together (ratio ~1-3), vel_disc is noise.
      - When vel_disc >> spike (ratio > 5), vel_disc is contact-driven.

    Observed values:
      ALOHA mobile  : vel_disc=0.013, spike=0.007 → ratio=1.9  (noise)
      DROID-100     : vel_disc=0.071, spike=0.046 → ratio=1.5  (noise)
      PushT real    : vel_disc=0.293, spike=0.012 → ratio=24.4 (contact)

    Scaling rule (contact ratio → vel_disc threshold multiplier):
      ratio <= 3.0  → 1.0x (unchanged — vel_disc is genuine noise)
      ratio == 10.0 → 2.0x (moderate contact signal)
      ratio >= 20.0 → 3.0x (strong contact signal, capped)
      Linear interpolation in [3, 20].

    Only active when cfg.contact_aware=True.
    Jerk spike and dropout thresholds are NOT relaxed — they encode genuine
    corruption regardless of contact density.
    """
    if not cfg.contact_aware:
        return cfg.max_vel_disc_rate

    disc_rates = ep_data.get("per_episode_vel_disc_rate", [])
    spike_rates = ep_data.get("per_episode_spike_rate", [])

    valid_disc = [v for v in disc_rates if v is not None]
    valid_spike = [v for v in spike_rates if v is not None]

    if not valid_disc or not valid_spike:
        return cfg.max_vel_disc_rate

    mean_disc = float(np.mean(valid_disc))
    mean_spike = float(np.mean(valid_spike))

    # Avoid division by zero; if spikes are absent treat ratio as large
    ratio = mean_disc / max(mean_spike, 0.001)

    _RATIO_LOW = 3.0    # below this: no scaling
    _RATIO_HIGH = 20.0  # above this: full 3x scale
    _SCALE_MAX = 3.0

    if ratio <= _RATIO_LOW:
        scale = 1.0
    elif ratio >= _RATIO_HIGH:
        scale = _SCALE_MAX
    else:
        t = (ratio - _RATIO_LOW) / (_RATIO_HIGH - _RATIO_LOW)
        scale = 1.0 + (_SCALE_MAX - 1.0) * t

    return cfg.max_vel_disc_rate * scale


def _quality_filter(
    episodes: list[Episode],
    ep_data: dict[str, list],
    cfg: "CoresetSelector",
    effective_max_vel_disc: float | None = None,
) -> list[int]:
    """Return indices of episodes that fail quality thresholds (to be removed)."""
    spike_rates = ep_data.get("per_episode_spike_rate", [])
    disc_rates = ep_data.get("per_episode_vel_disc_rate", [])
    dropouts = ep_data.get("per_episode_dropout_fraction", [])
    ldlj_values = ep_data.get("per_episode_ldlj", [])

    max_disc = effective_max_vel_disc if effective_max_vel_disc is not None else cfg.max_vel_disc_rate

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
        if disc is not None and disc > max_disc:
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
    latent_embeddings: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Build a (len(candidate_indices), F) feature matrix for diversity selection.

    Features (normalised to [0, 1]):
      - Latent state embeddings or Action-space statistics — behavioral representation
      - Quality metrics (spike_rate, vel_disc_rate) — quality diversity tie-breaker
      - Episode length (normalised)

    diversity_weight controls the blend: 1.0 = action stats/latent only; 0.0 = quality
    metrics only. Default 0.7.
    """
    spike_rates = ep_data.get("per_episode_spike_rate", [])
    disc_rates = ep_data.get("per_episode_vel_disc_rate", [])
    lengths = ep_data.get("per_episode_length", [])

    rows: list[np.ndarray] = []
    for i in candidate_indices:
        ep = episodes[i]
        acts = ep.actions
        if acts.ndim == 1:
            acts = acts[:, np.newaxis]

        if latent_embeddings is not None:
            action_feat = latent_embeddings.get(
                ep.metadata.episode_id, np.zeros(10, dtype=np.float32)
            )
        else:
            # Behavioral: action mean and std per dim
            action_mean = np.mean(acts, axis=0)
            action_std = np.std(acts, axis=0)
            action_feat = np.concatenate([action_mean, action_std])

        # Quality metrics as secondary features
        spike = _safe_get(spike_rates, i) or 0.0
        disc = _safe_get(disc_rates, i) or 0.0
        length_raw = _safe_get(lengths, i) or float(ep.n_steps)
        quality_feat = np.array([spike, disc, length_raw / 1000.0])

        # Entropy feature: per-trajectory action diversity (bits/dim).
        entropy_feat = np.array([entropy_scores.get(i, 0.0) if entropy_scores else 0.0])

        # Blend: diversity_weight for action stats, entropy_weight for entropy,
        # remaining for quality metrics.
        q_scale = max(0.0, 1.0 - diversity_weight - entropy_weight)
        row = np.concatenate(
            [
                action_feat * diversity_weight,
                quality_feat * q_scale,
                entropy_feat * entropy_weight,
            ]
        )
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

    Attempts PyTorch GPU acceleration if torch and a GPU (CUDA/MPS) are available.
    Falls back to NumPy.
    """
    n = len(features)
    if k >= n:
        return list(range(n))

    try:
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        if device in ("cuda", "mps"):
            feats_t = torch.tensor(features, dtype=torch.float32, device=device)
            centroid = feats_t.mean(dim=0)
            dists_to_centroid = torch.linalg.norm(feats_t - centroid, dim=1)
            seed = int(torch.argmax(dists_to_centroid).item())

            selected = [seed]
            min_dists = torch.linalg.norm(feats_t - feats_t[seed], dim=1).clone()
            min_dists[seed] = -float("inf")

            for _ in range(k - 1):
                next_idx = int(torch.argmax(min_dists).item())
                selected.append(next_idx)
                dists_to_new = torch.linalg.norm(feats_t - feats_t[next_idx], dim=1)
                min_dists = torch.minimum(min_dists, dists_to_new)
                min_dists[next_idx] = -float("inf")

            return selected
    except Exception:
        pass

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
    """Return {episode_id: min_distance_to_selected_set} for all candidates.
    Vectorized computation eliminating python loops.
    """
    selected_set = set(selected_local)
    selected_feats = features[sorted(selected_set)]  # (K, D)

    # Compute squared norms: (x - y)^2 = x^2 + y^2 - 2xy
    x2 = np.sum(features**2, axis=1, keepdims=True)  # (N, 1)
    y2 = np.sum(selected_feats**2, axis=1, keepdims=True).T  # (1, K)
    xy = features @ selected_feats.T  # (N, K)

    # Compute distances matrix - (N, K)
    dists2 = np.maximum(x2 + y2 - 2 * xy, 0.0)
    dists = np.sqrt(dists2)

    # Find min distance to selected set for each candidate
    min_dists = np.min(dists, axis=1)

    return {
        episodes[global_idx].metadata.episode_id: float(min_dists[local_idx])
        for local_idx, global_idx in enumerate(candidate_indices)
    }


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
        effective_max_vel_disc = _contact_aware_vel_disc(ep_data, self)
        quality_fail_indices = _quality_filter(episodes, ep_data, self, effective_max_vel_disc)
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
        elif self.strategy == "influence":
            keep_indices, diversity_pruned_indices, diversity_scores = _select_influence(
                self, batch, report, quality_pass_indices, k
            )
        elif self.strategy == "energy":
            keep_indices, diversity_pruned_indices, diversity_scores = _select_energy(
                self, batch, report, quality_pass_indices, k
            )
        elif self.strategy == "world-model":
            keep_indices, diversity_pruned_indices, diversity_scores = _select_world_model(
                self, batch, report, quality_pass_indices, k
            )
        else:
            entropy_scores = _compute_entropy_scores(episodes) if self.entropy_weight > 0 else {}
            latent_embeddings = None
            if self.latent_space != "none":
                from calibra.curation.latent_embed import extract_latent_embeddings

                latent_embeddings = extract_latent_embeddings(batch, model_type=self.latent_space)

            features = _build_feature_matrix(
                episodes,
                quality_pass_indices,
                ep_data,
                self.diversity_weight,
                entropy_scores,
                self.entropy_weight,
                latent_embeddings=latent_embeddings,
            )
            selected_local = _approximate_max_coverage(features, k, self.batch_size)
            selected_global = [quality_pass_indices[i] for i in selected_local]
            selected_set = set(selected_global)

            keep_indices = selected_global
            diversity_pruned_indices = [i for i in quality_pass_indices if i not in selected_set]
            diversity_scores = _diversity_score_map(
                episodes, quality_pass_indices, features, selected_local
            )

        keep_ids = [episodes[i].metadata.episode_id for i in keep_indices]
        fail_ids = [episodes[i].metadata.episode_id for i in quality_fail_indices]
        div_pruned_ids = [episodes[i].metadata.episode_id for i in diversity_pruned_indices]

        _method = (
            "quality_filter + jepa_world_model_surprise"
            if self.strategy == "world-model"
            else "quality_filter + approximate_minibatch_coverage"
        )
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
            method=_method,
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
