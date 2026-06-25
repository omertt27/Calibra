"""
Episode-level anomaly detection.

Identifies individual episodes that are statistical outliers within their own
dataset — not relative to a reference, but relative to the dataset's own
median. This is what catches silent corruption that aggregate metrics miss:
sync loss events, end-of-session fatigue, random hardware failures.

Method: median absolute deviation (MAD). An episode is flagged when its
per-episode metric deviates from the dataset median by more than
OUTLIER_K × MAD. MAD is robust to outliers themselves, so a cluster of
bad episodes doesn't inflate the threshold and hide itself.

OUTLIER_K=3.0 means "three median-absolute-deviations from the median" —
roughly equivalent to 3σ on a normal distribution but distribution-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibra.schema.report import DiagnosticReport

OUTLIER_K = 3.0

# (analyzer_name, raw_metrics_key, display_label, higher_is_worse)
_EPISODE_METRICS: list[tuple[str, str, str, bool]] = [
    ("temporal_stability", "per_episode_jitter_cv", "jitter_cv", True),
    ("temporal_stability", "per_episode_dropout_fraction", "dropout_rate", True),
    ("control_smoothness", "per_episode_spike_rate", "spike_rate", True),
    ("control_smoothness", "per_episode_vel_disc_rate", "vel_disc_rate", True),
    ("control_smoothness", "per_episode_ldlj", "ldlj", False),
]

# How many MADs above/below median before we call it an outlier.
# Separate thresholds for LDLJ (already noisy across episodes).
_OUTLIER_K_BY_METRIC: dict[str, float] = {
    "ldlj": 4.0,  # LDLJ has high natural variance; be more conservative
}


@dataclass
class EpisodeFlag:
    """A single per-episode anomaly signal."""

    episode_idx: int
    episode_id: str
    metric: str
    observed: float
    median: float
    deviation_mads: float
    higher_is_worse: bool

    @property
    def multiple(self) -> float:
        """How many MADs from the median."""
        return self.deviation_mads

    @property
    def direction(self) -> str:
        return "above" if self.observed > self.median else "below"


@dataclass
class EpisodeAnomaly:
    """All flags for a single episode, grouped."""

    episode_idx: int
    episode_id: str
    flags: list[EpisodeFlag] = field(default_factory=list)

    @property
    def severity(self) -> float:
        return max((f.deviation_mads for f in self.flags), default=0.0)

    @property
    def metrics(self) -> list[str]:
        return [f.metric for f in self.flags]


def find_outliers(
    report: DiagnosticReport,
    k: float = OUTLIER_K,
) -> list[EpisodeAnomaly]:
    """
    Return a list of EpisodeAnomaly objects, sorted by severity descending.
    Only episodes with at least one metric deviation > k MADs are returned.
    """
    raw_by_analyzer: dict[str, dict] = {
        r.analyzer_name: r.raw_metrics for r in report.analyzer_results
    }

    # episode_id lookup: fall back to string index if ids weren't populated
    n = report.n_episodes
    ids = report.episode_ids if report.episode_ids else [str(i) for i in range(n)]

    # Collect all flags keyed by episode index
    flags_by_ep: dict[int, list[EpisodeFlag]] = {}

    for analyzer_name, raw_key, label, higher_is_worse in _EPISODE_METRICS:
        raw = raw_by_analyzer.get(analyzer_name, {})
        values_raw = raw.get(raw_key, [])
        if not values_raw:
            continue

        arr = np.array(
            [v if v is not None else np.nan for v in values_raw],
            dtype=np.float64,
        )
        valid = arr[~np.isnan(arr)]
        if len(valid) < 5:
            continue

        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))

        # When MAD=0, most episodes share the same value — any deviation is
        # unambiguous. Fall back to std so we can still quantify the outlier.
        if mad < 1e-12:
            std = float(np.std(valid))
            if std < 1e-12:
                continue  # truly constant — nothing to flag
            scale = std
        else:
            scale = mad

        threshold_k = _OUTLIER_K_BY_METRIC.get(label, k)

        for idx, v in enumerate(arr):
            if np.isnan(v):
                continue
            deviation = (v - median) / scale
            is_anomaly = (higher_is_worse and deviation > threshold_k) or (
                not higher_is_worse and deviation < -threshold_k
            )
            if not is_anomaly:
                continue

            ep_id = ids[idx] if idx < len(ids) else str(idx)
            flag = EpisodeFlag(
                episode_idx=idx,
                episode_id=ep_id,
                metric=label,
                observed=float(v),
                median=median,
                deviation_mads=abs(deviation),
                higher_is_worse=higher_is_worse,
            )
            flags_by_ep.setdefault(idx, []).append(flag)

    anomalies = [
        EpisodeAnomaly(
            episode_idx=idx,
            episode_id=ids[idx] if idx < len(ids) else str(idx),
            flags=flags,
        )
        for idx, flags in flags_by_ep.items()
    ]
    return sorted(anomalies, key=lambda a: a.severity, reverse=True)


# ── rendering ─────────────────────────────────────────────────────────────────


def _consecutive_groups(indices: list[int]) -> list[list[int]]:
    """Group a sorted list of integers into consecutive runs."""
    if not indices:
        return []
    groups: list[list[int]] = [[indices[0]]]
    for i in indices[1:]:
        if i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _heuristic_label(
    anomalies_in_group: list[EpisodeAnomaly],
    n_episodes: int,
    group_size: int,
) -> str:
    metrics_seen = {m for a in anomalies_in_group for m in a.metrics}
    first_idx = anomalies_in_group[0].episode_idx
    last_idx = anomalies_in_group[-1].episode_idx

    if group_size >= 3:
        if "jitter_cv" in metrics_seen or "dropout_rate" in metrics_seen:
            return "cluster — possible sync loss or recording interruption"
        return "cluster — possible operator fatigue or recording artifact"

    if last_idx >= int(n_episodes * 0.90):
        return "end of dataset — possible fatigue or equipment drift"
    if first_idx <= int(n_episodes * 0.05):
        return "start of dataset — possible equipment warm-up artifact"
    if "jitter_cv" in metrics_seen:
        return "possible sync loss or timestamp irregularity"
    if "dropout_rate" in metrics_seen:
        return "possible frame drop or recording gap"
    return "investigate before training"


def render(anomalies: list[EpisodeAnomaly], n_episodes: int) -> str:
    if not anomalies:
        return ""

    n_flagged = len(anomalies)
    lines = [
        "─── Episode Anomalies " + "─" * 36,
        f"{n_flagged} of {n_episodes} episodes are outliers within this dataset:",
        "",
    ]

    # Group by consecutive indices for display
    by_idx = {a.episode_idx: a for a in anomalies}
    sorted_idxs = sorted(by_idx)
    groups = _consecutive_groups(sorted_idxs)

    for group in groups:
        group_anomalies = [by_idx[i] for i in group]
        label = _heuristic_label(group_anomalies, n_episodes, len(group))

        if len(group) == 1:
            a = group_anomalies[0]
            metric_strs = [
                f"{f.metric} {f.deviation_mads:.1f}× MAD"
                for f in sorted(a.flags, key=lambda f: f.deviation_mads, reverse=True)
            ]
            lines.append(f"  ep_{a.episode_id:<8}  {', '.join(metric_strs)}")
        else:
            id_range = f"ep_{group_anomalies[0].episode_id}–{group_anomalies[-1].episode_id}"
            # Collect all metrics across the cluster, show worst multiple
            all_flags: list[EpisodeFlag] = [f for a in group_anomalies for f in a.flags]
            by_metric: dict[str, float] = {}
            for f in all_flags:
                by_metric[f.metric] = max(by_metric.get(f.metric, 0), f.deviation_mads)
            metric_strs = [
                f"{m} {v:.1f}× MAD"
                for m, v in sorted(by_metric.items(), key=lambda x: x[1], reverse=True)
            ]
            lines.append(f"  {id_range:<20}  {', '.join(metric_strs)}")

        lines.append(f"    → {label}")
        lines.append("")

    lines.append(
        "Inspect flagged episodes before training. "
        "Remove with: calibra.comparison.curator.EpisodeCurator"
    )
    lines.append("─" * 58)
    return "\n".join(lines)
