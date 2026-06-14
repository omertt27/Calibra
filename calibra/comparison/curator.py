"""
EpisodeCurator: filter episodes based on per-episode Phase 1 signals.

Accepts a pre-computed DiagnosticReport alongside the raw EpisodeBatch so
that per-episode metrics don't need to be recomputed. Every removal decision
is recorded in the returned CurationReport so users can audit which signal
triggered each drop and adjust thresholds accordingly.

Typical usage:
    pipeline = Pipeline()
    report   = pipeline.run(batch)
    curator  = EpisodeCurator(min_length=50, max_jitter_cv=0.20)
    filtered_batch, curation_report = curator.curate(batch, report)
    print(curation_report.summary())
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import DiagnosticReport
from calibra.schema.comparison import CurationReport, EpisodeFlag
from calibra.comparison.comparator import _extract_ep_data


@dataclass
class EpisodeCurator:
    """
    Filter episodes from a batch based on per-episode Phase 1 quality signals.

    Each configured threshold is checked independently; an episode is dropped
    if it violates ANY threshold. The CurationReport lists every (episode,
    metric) violation so users can audit and override individual decisions.

    Parameters
    ----------
    min_length : drop episodes with fewer than this many steps.
                 Applied from the batch directly; no analyzer required.
    max_jitter_cv : drop if per-episode timestamp jitter CV exceeds this.
                    Requires TemporalAnalyzer in the pipeline.
    max_dropout_fraction : drop if per-episode dropout fraction exceeds this.
                           Requires TemporalAnalyzer in the pipeline.
    min_ldlj : drop if per-episode LDLJ is below this value (LDLJ is negative;
               more negative = worse). E.g. min_ldlj=-15.0 drops the jerkiest
               episodes. Requires ControlSmoothnessAnalyzer in the pipeline.
    max_spike_rate : drop if per-episode jerk spike fraction exceeds this.
                     Requires ControlSmoothnessAnalyzer in the pipeline.
    max_vel_disc_rate : drop if per-episode velocity discontinuity rate exceeds this.
                        Requires ControlSmoothnessAnalyzer in the pipeline.
    """

    min_length:           Optional[int]   = None
    max_jitter_cv:        Optional[float] = None
    max_dropout_fraction: Optional[float] = None
    min_ldlj:             Optional[float] = None
    max_spike_rate:       Optional[float] = None
    max_vel_disc_rate:    Optional[float] = None

    def curate(
        self,
        batch: EpisodeBatch,
        report: DiagnosticReport,
    ) -> tuple[EpisodeBatch, CurationReport]:
        """
        Filter a batch using a pre-computed DiagnosticReport for per-episode signals.

        The report must have been computed from the same batch (same episode order).
        A UserWarning is emitted if report.n_episodes != batch.n_episodes, since
        per-episode arrays will be misaligned in that case.

        Returns
        -------
        filtered_batch   : EpisodeBatch with only retained episodes.
        curation_report  : CurationReport listing every removal decision.
        """
        if report.n_episodes != batch.n_episodes:
            warnings.warn(
                f"report.n_episodes ({report.n_episodes}) != "
                f"batch.n_episodes ({batch.n_episodes}). "
                "Per-episode arrays may be misaligned — curate() results "
                "should not be trusted.",
                stacklevel=2,
            )

        ep_data = _extract_ep_data(report)
        n = batch.n_episodes
        per_ep_flags: list[list[EpisodeFlag]] = [[] for _ in range(n)]

        # ── apply each configured threshold ──────────────────────────────────

        if self.min_length is not None:
            for i, ep in enumerate(batch.episodes):
                if ep.n_steps < self.min_length:
                    per_ep_flags[i].append(EpisodeFlag(
                        episode_index=i,
                        episode_id=ep.metadata.episode_id,
                        metric="length",
                        observed_value=float(ep.n_steps),
                        threshold=float(self.min_length),
                        direction="too_short",
                        interpretation=(
                            f"Episode has {ep.n_steps} steps, "
                            f"below min_length={self.min_length}."
                        ),
                    ))

        _flag_upper(per_ep_flags, batch.episodes,
                    ep_data.get("per_episode_jitter_cv", []),
                    "timestamp_jitter_cv", self.max_jitter_cv,
                    "Timestamp jitter CV")

        _flag_upper(per_ep_flags, batch.episodes,
                    ep_data.get("per_episode_dropout_fraction", []),
                    "timestamp_dropout_rate", self.max_dropout_fraction,
                    "Timestamp dropout fraction")

        _flag_lower(per_ep_flags, batch.episodes,
                    ep_data.get("per_episode_ldlj", []),
                    "ldlj", self.min_ldlj,
                    "LDLJ smoothness score")

        _flag_upper(per_ep_flags, batch.episodes,
                    ep_data.get("per_episode_spike_rate", []),
                    "jerk_spike_rate", self.max_spike_rate,
                    "Jerk spike rate")

        _flag_upper(per_ep_flags, batch.episodes,
                    ep_data.get("per_episode_vel_disc_rate", []),
                    "velocity_discontinuity_rate", self.max_vel_disc_rate,
                    "Velocity discontinuity rate")

        # ── split retained / dropped ──────────────────────────────────────────

        retained_indices: list[int] = []
        dropped_indices: list[int] = []
        flat_flags: list[EpisodeFlag] = []

        for i in range(n):
            if per_ep_flags[i]:
                dropped_indices.append(i)
                flat_flags.extend(per_ep_flags[i])
            else:
                retained_indices.append(i)

        filtered_batch = EpisodeBatch(
            episodes=[batch.episodes[i] for i in retained_indices],
            dataset_name=batch.dataset_name + "_curated",
            format=batch.format,
            source_path=batch.source_path,
        )

        curation_report = CurationReport(
            original_n_episodes=n,
            retained_n_episodes=len(retained_indices),
            retained_indices=retained_indices,
            dropped_indices=dropped_indices,
            episode_flags=flat_flags,
        )

        return filtered_batch, curation_report


# ── threshold helpers ─────────────────────────────────────────────────────────

def _flag_upper(
    per_ep_flags: list[list[EpisodeFlag]],
    episodes: list[Episode],
    ep_values: list,
    metric: str,
    threshold: Optional[float],
    label: str,
) -> None:
    """Flag episode i if ep_values[i] > threshold."""
    if threshold is None or not ep_values:
        return
    for i, val in enumerate(ep_values[: len(episodes)]):
        if val is None:
            continue
        if float(val) > threshold:
            per_ep_flags[i].append(EpisodeFlag(
                episode_index=i,
                episode_id=episodes[i].metadata.episode_id,
                metric=metric,
                observed_value=float(val),
                threshold=float(threshold),
                direction="too_high",
                interpretation=(
                    f"{label} = {float(val):.4g} exceeds "
                    f"max threshold {threshold:.4g}."
                ),
            ))


def _flag_lower(
    per_ep_flags: list[list[EpisodeFlag]],
    episodes: list[Episode],
    ep_values: list,
    metric: str,
    threshold: Optional[float],
    label: str,
) -> None:
    """Flag episode i if ep_values[i] < threshold."""
    if threshold is None or not ep_values:
        return
    for i, val in enumerate(ep_values[: len(episodes)]):
        if val is None:
            continue
        if float(val) < threshold:
            per_ep_flags[i].append(EpisodeFlag(
                episode_index=i,
                episode_id=episodes[i].metadata.episode_id,
                metric=metric,
                observed_value=float(val),
                threshold=float(threshold),
                direction="too_low",
                interpretation=(
                    f"{label} = {float(val):.4g} is below "
                    f"min threshold {threshold:.4g}."
                ),
            ))
