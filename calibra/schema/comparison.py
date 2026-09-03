"""
Output-facing schema for Phase 2 Comparative Analysis Layer.

DriftFlag      : a single metric comparison between two DiagnosticReports.
ComparisonReport : collection of DriftFlags produced by DatasetComparator.
EpisodeFlag    : a quality signal attached to one episode in a curation pass.
CurationReport : audit trail from EpisodeCurator.curate().
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator

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


class Disposition(str, Enum):
    """
    What Calibra recommends be done with an episode (ADR-011).

    Closed set — adding a value is a deliberate schema change. Downstream
    tooling (dataset exporters, CI gates) switches on these, so the vocabulary
    must stay small and stable.

    KEEP / DOWNWEIGHT / ANNOTATE all mean "include the episode in training";
    they differ only in how it is weighted or annotated. DROP and RECOLLECT
    mean "not in this training set".
    """

    KEEP = "KEEP"  # in the coreset as-is; the default when no signal fires
    DROP = "DROP"  # exclude: integrity failure, or redundancy with no coverage value
    DOWNWEIGHT = "DOWNWEIGHT"  # include with a reduced sample weight (see .weight)
    ANNOTATE = "ANNOTATE"  # include; emit characterization as conditioning metadata
    REVIEW = "REVIEW"  # needs human inspection before a decision — unusual, not clearly bad
    RECOLLECT = "RECOLLECT"  # reserved: data-acquisition guidance, not emitted yet


# Dispositions that mean "the episode is part of the training set".
KEEP_LIKE: frozenset[Disposition] = frozenset(
    {Disposition.KEEP, Disposition.DOWNWEIGHT, Disposition.ANNOTATE}
)


class EpisodeCharacterization(BaseModel):
    """
    Per-episode disposition plus the signals behind it (ADR-011).

    Characterization is computed for every episode regardless of its
    disposition: prune mode materialises the KEEP-like set, annotate mode
    emits this whole record as a LeRobot-compatible sidecar for
    metadata-conditioned training.

    Every numeric field is Optional — it is None when the analyzer that
    produces it did not run (e.g. no coverage_value without InfluenceAnalyzer).
    """

    episode_index: int  # position in the original EpisodeBatch
    episode_id: str  # from EpisodeMetadata
    disposition: Disposition = Disposition.KEEP

    # characterization signals — the union of what the pipeline already computes
    n_steps: Optional[int] = None
    success: Optional[bool] = None
    calibra_score: Optional[float] = None
    anomaly_score: Optional[float] = None  # 0-1, how statistically unusual
    quality_risk: Optional[float] = None  # 0-1, likelihood of a real recording/execution problem
    coverage_value: Optional[float] = None  # 0-1, unique behavioral coverage contributed
    redundancy: Optional[float] = None  # 0-1, how near-duplicate of other episodes
    integrity_flags: list[str] = []  # metric names that failed an integrity check

    weight: Optional[float] = None  # only meaningful when disposition == DOWNWEIGHT

    reasons: list[str] = []  # human-readable, one per triggering signal


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

    ADR-011: the primary output is `dispositions` — one EpisodeCharacterization
    per episode, carrying its decision (KEEP / DROP / DOWNWEIGHT / ANNOTATE /
    REVIEW / RECOLLECT) and the signals behind it.

    `retained_indices` / `dropped_indices` are the legacy keep/drop view, kept
    for backward compatibility. They are derived from `dispositions` (KEEP-like
    → retained, everything else → dropped) when only `dispositions` is
    supplied, and vice versa — so a caller can populate either and read both.
    Prefer `dispositions` / `by_disposition()` in new code.
    """

    original_n_episodes: int
    retained_n_episodes: int
    retained_indices: list[int] = []  # legacy: indices in original batch that are kept
    dropped_indices: list[int] = []  # legacy: indices in original batch that are dropped
    episode_flags: list[EpisodeFlag] = []  # one per (episode, metric) violation
    dispositions: list[EpisodeCharacterization] = []  # ADR-011: per-episode decision + signals

    @model_validator(mode="after")
    def _sync_dispositions_and_indices(self) -> "CurationReport":
        """
        Keep `dispositions` and the legacy `retained_indices` /
        `dropped_indices` consistent regardless of which the caller supplied.
        If both are given they are left untouched (the caller owns the split).
        """
        has_indices = bool(self.retained_indices or self.dropped_indices)
        if self.dispositions and not has_indices:
            self.retained_indices = [
                d.episode_index for d in self.dispositions if d.disposition in KEEP_LIKE
            ]
            self.dropped_indices = [
                d.episode_index for d in self.dispositions if d.disposition not in KEEP_LIKE
            ]
        elif has_indices and not self.dispositions:
            retained = set(self.retained_indices)
            self.dispositions = [
                EpisodeCharacterization(
                    episode_index=i,
                    episode_id=str(i),
                    disposition=Disposition.KEEP if i in retained else Disposition.DROP,
                )
                for i in sorted(retained | set(self.dropped_indices))
            ]
        return self

    @property
    def drop_fraction(self) -> float:
        if self.original_n_episodes == 0:
            return 0.0
        return len(self.dropped_indices) / self.original_n_episodes

    def flags_for_episode(self, episode_index: int) -> list[EpisodeFlag]:
        return [f for f in self.episode_flags if f.episode_index == episode_index]

    def by_disposition(self, disposition: Disposition) -> list[EpisodeCharacterization]:
        """All per-episode records with the given disposition."""
        return [d for d in self.dispositions if d.disposition == disposition]

    def disposition_counts(self) -> dict[str, int]:
        """Episode count per disposition, e.g. {'KEEP': 90, 'DROP': 10}."""
        counts: dict[str, int] = {}
        for d in self.dispositions:
            counts[d.disposition.value] = counts.get(d.disposition.value, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            "=== Calibra Curation Report ===",
            f"Original  : {self.original_n_episodes} episodes",
            f"Retained  : {self.retained_n_episodes} episodes",
            f"Dropped   : {len(self.dropped_indices)} ({self.drop_fraction:.1%})",
        ]
        other = {k: v for k, v in self.disposition_counts().items() if k not in ("KEEP", "DROP")}
        if other:
            lines.append(
                "Other     : " + ", ".join(f"{k.lower()}={v}" for k, v in sorted(other.items()))
            )
        lines.append("")

        char_by_idx = {d.episode_index: d for d in self.dispositions}
        if self.dropped_indices:
            detail: list[str] = []
            for idx in self.dropped_indices:
                ep_flags = self.flags_for_episode(idx)
                if ep_flags:
                    ep_id = ep_flags[0].episode_id
                    reasons = "; ".join(
                        f"{f.metric}={f.observed_value:.4g} "
                        f"({f.direction}, threshold={f.threshold:.4g})"
                        for f in ep_flags
                    )
                    detail.append(f"  [{idx}] {ep_id}: {reasons}")
                elif idx in char_by_idx:
                    c = char_by_idx[idx]
                    reasons = "; ".join(c.reasons) if c.reasons else c.disposition.value
                    detail.append(f"  [{idx}] {c.episode_id}: {reasons}")
            if detail:
                lines.append("--- Dropped Episodes ---")
                lines.extend(detail)
        return "\n".join(lines)
