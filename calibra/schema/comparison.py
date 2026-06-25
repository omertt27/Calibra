"""
Output-facing schema for Phase 2 Comparative Analysis Layer.

DriftFlag      : a single metric comparison between two DiagnosticReports.
ComparisonReport : collection of DriftFlags produced by DatasetComparator.
EpisodeFlag    : a quality signal attached to one episode in a curation pass.
CurationReport : audit trail from EpisodeCurator.curate().
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from calibra.schema.report import ObservedValue, RiskLevel

_LEVEL_ICONS = {
    RiskLevel.CRITICAL: "❌",
    RiskLevel.WARNING: "⚠️ ",
    RiskLevel.OK: "✅",
    RiskLevel.INFO: "ℹ️ ",
}

_DIRECTION_ICONS = {
    "degraded": "↓",
    "improved": "↑",
    "ambiguous": "~",
}


class DriftFlag(BaseModel):
    """
    A metric-level comparison between a baseline and a candidate DiagnosticReport.

    `direction` is "degraded" when the candidate is worse than the baseline
    for this metric, "improved" when it is better, and "ambiguous" when the
    metric has no inherent good direction (e.g. contact_density).

    `significant` is True when the difference is statistically meaningful:
    either p_value < alpha (permutation test), or the two bootstrap CIs do
    not overlap (fallback when per-episode data is unavailable).
    """

    metric: str
    analyzer_name: str
    baseline_observed: ObservedValue
    candidate_observed: ObservedValue
    delta: Optional[float] = None  # candidate.value - baseline.value
    relative_change: Optional[float] = None  # delta / |baseline.value|
    p_value: Optional[float] = None  # permutation test; None = CI fallback
    significant: bool = False
    direction: str = "ambiguous"  # "degraded" | "improved" | "ambiguous"
    level: RiskLevel = RiskLevel.INFO
    interpretation: str = ""
    implication: str = ""

    def render(self) -> str:
        icon = _LEVEL_ICONS.get(self.level, "")
        dir_icon = _DIRECTION_ICONS.get(self.direction, "~")
        sig = "*" if self.significant else ""
        header = (
            f"{icon} {dir_icon}{sig} {self.metric}: "
            f"{self.baseline_observed} → {self.candidate_observed}"
        )
        if self.delta is not None:
            rel = f" ({self.relative_change:+.1%})" if self.relative_change is not None else ""
            header += f"  Δ={self.delta:+.4g}{rel}"
        if self.p_value is not None:
            header += f"  p={self.p_value:.3f}"
        if self.implication:
            header += f"\n   → {self.implication}"
        return header


class ComparisonReport(BaseModel):
    """
    Report comparing two DiagnosticReports metric by metric.

    Produced by DatasetComparator.compare(baseline_report, candidate_report).
    Only includes metrics present in both reports.
    """

    baseline_name: str
    candidate_name: str
    baseline_n_episodes: int
    candidate_n_episodes: int
    n_permutations: int = 199
    alpha: float = 0.05
    drift_flags: list[DriftFlag] = []

    @property
    def degraded(self) -> list[DriftFlag]:
        """Significant regressions: direction == 'degraded' and significant."""
        return [f for f in self.drift_flags if f.direction == "degraded" and f.significant]

    @property
    def improved(self) -> list[DriftFlag]:
        """Significant improvements: direction == 'improved' and significant."""
        return [f for f in self.drift_flags if f.direction == "improved" and f.significant]

    def summary(self) -> str:
        lines = [
            "=== Calibra Comparison Report ===",
            f"Baseline  : {self.baseline_name} ({self.baseline_n_episodes} episodes)",
            f"Candidate : {self.candidate_name} ({self.candidate_n_episodes} episodes)",
            f"Test      : permutation (n={self.n_permutations}), α={self.alpha}",
            "",
        ]

        if self.drift_flags:
            lines.append("--- Drift Flags ---")
            for flag in self.drift_flags:
                lines.append(flag.render())
                lines.append("")

        n_deg = len(self.degraded)
        n_imp = len(self.improved)
        lines.append(f"{n_deg} significant regressions  ·  {n_imp} significant improvements")
        return "\n".join(lines)


# ── curation schema ───────────────────────────────────────────────────────────


class EpisodeFlag(BaseModel):
    """
    A quality signal attached to a single episode in a curation pass.

    Records exactly which metric, threshold, and direction triggered the
    episode's removal so users can audit or override decisions.
    """

    episode_index: int  # position in the original EpisodeBatch
    episode_id: str  # from EpisodeMetadata
    metric: str  # e.g. "timestamp_jitter_cv", "length"
    observed_value: float  # the episode-level value that triggered removal
    threshold: float  # the configured threshold
    direction: str  # "too_high" | "too_low" | "too_short"
    interpretation: str  # human-readable explanation


class CurationReport(BaseModel):
    """
    Audit trail from EpisodeCurator.curate().

    Records which episodes were retained and which were dropped, and for
    each dropped episode, which signal triggered the removal. Inspect
    episode_flags to understand and override individual decisions.
    """

    original_n_episodes: int
    retained_n_episodes: int
    retained_indices: list[int] = []  # indices in original batch that are kept
    dropped_indices: list[int] = []  # indices in original batch that are dropped
    episode_flags: list[EpisodeFlag] = []  # one per (episode, metric) violation

    @property
    def drop_fraction(self) -> float:
        if self.original_n_episodes == 0:
            return 0.0
        return len(self.dropped_indices) / self.original_n_episodes

    def flags_for_episode(self, episode_index: int) -> list[EpisodeFlag]:
        return [f for f in self.episode_flags if f.episode_index == episode_index]

    def summary(self) -> str:
        lines = [
            "=== Calibra Curation Report ===",
            f"Original  : {self.original_n_episodes} episodes",
            f"Retained  : {self.retained_n_episodes} episodes",
            f"Dropped   : {len(self.dropped_indices)} ({self.drop_fraction:.1%})",
            "",
        ]
        if self.dropped_indices:
            lines.append("--- Dropped Episodes ---")
            for idx in self.dropped_indices:
                ep_flags = self.flags_for_episode(idx)
                if ep_flags:
                    ep_id = ep_flags[0].episode_id
                    reasons = "; ".join(
                        f"{f.metric}={f.observed_value:.4g} "
                        f"({f.direction}, threshold={f.threshold:.4g})"
                        for f in ep_flags
                    )
                    lines.append(f"  [{idx}] {ep_id}: {reasons}")
        return "\n".join(lines)
