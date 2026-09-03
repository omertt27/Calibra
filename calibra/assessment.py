"""
Per-episode assessment — separates three concepts that a single outlier
score collapses together:

  anomaly_score   : how statistically unusual this episode is, on any metric.
  quality_risk    : how likely there's an actual recording/execution problem.
  coverage_value  : how much unique behavioral coverage this episode contributes.

A statistically unusual episode is not necessarily bad — it may be a recovery
behavior, a rare task state, or a successful-but-unconventional trajectory.
Ranking episodes for review by anomaly_score alone (duration/jerk/velocity
IQR outliers) is table stakes; combining it with quality_risk and
coverage_value tells a user whether an unusual episode is worth excluding or
worth keeping.

This module adds no new metrics. It re-combines per-episode signals already
computed by the pipeline's analyzers:
  - quality_risk    reuses pruning.compute_quality_scores_for_ids() (the same
                     composite pruning already uses to decide what to drop).
  - coverage_value  reuses InfluenceAnalyzer's per_episode_influence (novelty +
                     entropy + contact-density composite, 0-1).
  - anomaly_score   is new: the largest percentile-rank extremity (distance
                     from the batch median) among the per_episode_* metrics on
                     which this episode is in the decile tail (p<=0.1 or
                     p>=0.9). It fires on any dimension without presupposing
                     which one matters; an episode mid-pack on every metric
                     scores 0.0. Weak signal on datasets with fewer than ~100
                     episodes, where rank tails are coarse.

Percentile ranking is global by default, which can misfire on datasets that
mix multiple tasks/robots/sessions — a harder task's episodes can look like
uniform outliers just for being compared against an easier one. Pass `batch`
and `group_by` to compute_episode_assessments() to rank within each group
(e.g. per task) instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from calibra.comparison.comparator import _extract_ep_data
from calibra.pruning import compute_quality_scores_for_ids
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport

# per_episode_* raw_metrics keys considered for anomaly detection. Each is a
# list aligned with report.episode_ids (see comparator._extract_ep_data).
_ANOMALY_METRICS = (
    "per_episode_spike_rate",
    "per_episode_vel_disc_rate",
    "per_episode_dropout_fraction",
    "per_episode_ldlj",
    "per_episode_jitter_cv",
    "per_episode_length",
    "per_episode_dynamics_error",
    "per_episode_contact_fraction",
    "per_episode_transition_entropy",
)

# An episode counts toward `anomaly_score` on a metric only if it is BOTH in
# that metric's rank tail AND this many robust-MAD deviations from the batch
# median. The rank test alone saturated to ~1.0 for every episode once the
# batch was small relative to the metric count (every episode is min/max on
# something); requiring real magnitude separation keeps the score meaningful.
_ANOMALY_Z_THRESHOLD = 2.5


@dataclass
class AnomalyReason:
    """One metric that put an episode in the tail of the batch distribution."""

    metric: str
    percentile: float  # 0-100; distance from 50 indicates how extreme


@dataclass
class EpisodeAssessment:
    """
    Three-axis review signal for a single episode. See module docstring.

    review_priority combines the three axes into one sortable number for a
    ranked review queue — high anomaly/quality_risk raise it, high
    coverage_value lowers it (a rare-but-valuable episode is deprioritized
    for exclusion even if it's also anomalous).
    """

    episode_id: str
    anomaly_score: float  # 0-1
    quality_risk: float  # 0-1
    coverage_value: Optional[float] = None  # 0-1, None if no InfluenceAnalyzer result
    reasons: list[AnomalyReason] = field(default_factory=list)

    @property
    def review_priority(self) -> float:
        score = 0.5 * self.quality_risk + 0.5 * self.anomaly_score
        if self.coverage_value is not None:
            score -= 0.3 * self.coverage_value
        return round(score, 6)


def summarize_assessments(assessments: Sequence[EpisodeAssessment]) -> dict[str, Optional[float]]:
    """
    Roll up a batch of per-episode EpisodeAssessment into dataset-level means,
    for logging alongside a design-partner experiment record (see
    calibra.experiment_log.ExperimentLog.record). coverage_value is averaged
    only over episodes where it was actually computed (InfluenceAnalyzer ran);
    None if it never did.
    """
    if not assessments:
        return {"mean_anomaly_score": None, "mean_quality_risk": None, "mean_coverage_value": None}
    coverage = [a.coverage_value for a in assessments if a.coverage_value is not None]
    return {
        "mean_anomaly_score": float(np.mean([a.anomaly_score for a in assessments])),
        "mean_quality_risk": float(np.mean([a.quality_risk for a in assessments])),
        "mean_coverage_value": float(np.mean(coverage)) if coverage else None,
    }


def episode_calibra_score(quality_risk: Optional[float]) -> Optional[float]:
    """
    Per-episode 0–100 cleanliness score: ``100 · (1 − quality_risk)``. A clean
    episode scores 100, a maximally risky one 0. None when quality_risk is
    unavailable.

    This is NOT a per-episode decomposition of the dataset-level Calibra Score
    (`calibra.score.compute_score`), which is a quality-gated blend of temporal,
    smoothness, coverage and task-structure dimensions. Averaging these
    per-episode scores does not reproduce it. This one is quality-risk only.
    """
    if quality_risk is None:
        return None
    return round(100.0 * (1.0 - quality_risk), 1)


def episode_redundancy(coverage_value: Optional[float]) -> Optional[float]:
    """
    Per-episode redundancy: ``1 − coverage_value``, the complement of how much
    unique behavioral coverage the episode adds (novelty + entropy +
    contact-density, via InfluenceAnalyzer). High = its behavior is already
    well represented elsewhere. None without InfluenceAnalyzer. Perfectly
    anti-correlated with coverage_value by construction — not an independent
    feature, a human-readable restatement.
    """
    if coverage_value is None:
        return None
    return round(1.0 - coverage_value, 4)


def _percentile_ranks(values: list) -> list[Optional[float]]:
    """
    Rank each value against the others in the same list as a 0-1 fraction
    (fraction of the batch at or below this value). None where the source
    metric is missing for that episode, or where the metric is constant
    across the batch (zero variance carries no anomaly signal — without this
    guard, a metric that's 0.0 for every episode, e.g. no dropout anywhere,
    degenerates to "100th percentile" for everyone via `<=`).
    """
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if len(valid_idx) < 2:
        return [None] * len(values)
    valid_vals = np.array([values[i] for i in valid_idx], dtype=np.float64)
    if valid_vals.max() - valid_vals.min() < 1e-9:
        return [None] * len(values)
    out: list[Optional[float]] = [None] * len(values)
    for idx, v in zip(valid_idx, valid_vals):
        out[idx] = float(np.mean(valid_vals <= v))
    return out


def _abs_robust_z(values: list) -> list[Optional[float]]:
    """
    |z| of each value against the batch median, scaled by MAD (falling back to
    std, then giving up on a constant metric). None where the metric is
    missing or there are fewer than 3 comparable episodes. Used to require
    that a rank-tail episode is also genuinely far from typical before it
    counts toward anomaly_score.
    """
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if len(valid_idx) < 3:
        return [None] * len(values)
    vv = np.array([values[i] for i in valid_idx], dtype=np.float64)
    median = float(np.median(vv))
    scale = 1.4826 * float(np.median(np.abs(vv - median)))
    if scale < 1e-9:
        scale = float(vv.std())
    if scale < 1e-9:
        return [None] * len(values)
    out: list[Optional[float]] = [None] * len(values)
    for idx, v in zip(valid_idx, vv):
        out[idx] = abs(float(v) - median) / scale
    return out


def _resolve_group_key(episode, group_by: Sequence[str]) -> tuple:
    """
    Resolve one episode's grouping key. "task" reads the first-class
    task_description field; anything else is looked up in metadata.extra
    (format-specific bag) — e.g. "robot", "operator", "session", once an
    adapter populates those. Missing keys resolve to None, which buckets
    episodes without that metadata into their own group rather than
    silently mixing them with a real value.
    """
    key = []
    for field_name in group_by:
        if field_name == "task":
            key.append(episode.metadata.task_description)
        else:
            key.append(episode.metadata.extra.get(field_name))
    return tuple(key)


def _group_indices(
    batch: EpisodeBatch, episode_ids: list[str], group_by: Sequence[str]
) -> list[list[int]]:
    """Bucket episode indices by their resolved group key."""
    groups: dict[tuple, list[int]] = {}
    for i, ep in enumerate(batch.episodes):
        groups.setdefault(_resolve_group_key(ep, group_by), []).append(i)
    return list(groups.values())


def compute_episode_assessments(
    report: DiagnosticReport,
    batch: Optional[EpisodeBatch] = None,
    group_by: Optional[Sequence[str]] = None,
) -> list[EpisodeAssessment]:
    """
    Derive a three-axis EpisodeAssessment per episode from a DiagnosticReport.

    Requires the report to have been produced with the default per-episode
    analyzers (temporal, smoothness, coverage, transition_dynamics) for
    quality_risk/anomaly signals, and InfluenceAnalyzer for coverage_value —
    axes silently degrade (quality_risk=0.0, coverage_value=None, fewer
    anomaly reasons) when the underlying analyzer didn't run or was skipped
    for lacking a required capability.

    Pass `batch` and `group_by` (e.g. ["task"], or ["task", "robot"] once an
    adapter populates EpisodeMetadata.extra with those keys) to rank anomaly
    percentiles within each group instead of across the whole dataset — a
    global IQR/percentile threshold can otherwise flag an entire valid
    subpopulation (e.g. a harder task with naturally higher jerk) as
    anomalous relative to the rest. quality_risk and coverage_value are
    absolute/pairwise metrics already and aren't affected by grouping.
    """
    episode_ids = report.episode_ids
    n = len(episode_ids)
    if n == 0:
        return []

    ep_data = _extract_ep_data(report)
    quality_risk = compute_quality_scores_for_ids(episode_ids, ep_data)

    coverage_value_by_id: dict[str, float] = {}
    for result in report.analyzer_results:
        if "per_episode_influence" in result.raw_metrics:
            coverage_value_by_id = result.raw_metrics["per_episode_influence"]
            break

    if group_by and batch is not None:
        groups = _group_indices(batch, episode_ids, group_by)
    else:
        groups = [list(range(n))]

    percentiles_by_metric: dict[str, list[Optional[float]]] = {}
    absz_by_metric: dict[str, list[Optional[float]]] = {}
    for metric in _ANOMALY_METRICS:
        values = ep_data.get(metric)
        if not values or len(values) != n:
            continue
        merged_p: list[Optional[float]] = [None] * n
        merged_z: list[Optional[float]] = [None] * n
        for group_indices in groups:
            group_values = [values[i] for i in group_indices]
            group_percentiles = _percentile_ranks(group_values)
            group_absz = _abs_robust_z(group_values)
            for local_i, global_i in enumerate(group_indices):
                merged_p[global_i] = group_percentiles[local_i]
                merged_z[global_i] = group_absz[local_i]
        percentiles_by_metric[metric] = merged_p
        absz_by_metric[metric] = merged_z

    assessments = []
    for i, episode_id in enumerate(episode_ids):
        reasons: list[AnomalyReason] = []
        anomaly = 0.0
        for metric, percentiles in percentiles_by_metric.items():
            p = percentiles[i]
            if p is None:
                continue
            if not (p >= 0.9 or p <= 0.1):
                continue
            reasons.append(AnomalyReason(metric=metric, percentile=round(p * 100, 1)))
            # A rank-tail episode only counts toward anomaly_score if it is also
            # genuinely far from the batch median (robust MAD z). The rank test
            # alone saturated to ~1.0 for every episode once the batch was small
            # relative to the ~9 metrics — every episode is min/max on something.
            z = absz_by_metric[metric][i]
            if z is not None and z >= _ANOMALY_Z_THRESHOLD:
                extremity = abs(p - 0.5) * 2.0  # 0 at the median, 1 at either tail
                anomaly = max(anomaly, extremity)
        reasons.sort(key=lambda r: abs(r.percentile - 50.0), reverse=True)

        assessments.append(
            EpisodeAssessment(
                episode_id=episode_id,
                anomaly_score=round(anomaly, 4),
                quality_risk=quality_risk.get(episode_id, 0.0),
                coverage_value=coverage_value_by_id.get(episode_id),
                reasons=reasons,
            )
        )
    return assessments


def rank_for_review(assessments: list[EpisodeAssessment]) -> list[EpisodeAssessment]:
    """Sort episodes by review_priority, most in need of human review first."""
    return sorted(assessments, key=lambda a: a.review_priority, reverse=True)
