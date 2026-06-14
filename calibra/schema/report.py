"""
Output-facing diagnostic report schema.

All fields are Pydantic models so reports serialize cleanly to JSON.
Analyzers produce AnalyzerResult objects; the pipeline assembles them into
a DiagnosticReport.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, model_validator


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"
    OK       = "OK"
    INFO     = "INFO"


_LEVEL_ORDER = {RiskLevel.CRITICAL: 0, RiskLevel.WARNING: 1,
                RiskLevel.OK: 2, RiskLevel.INFO: 3}
_ICONS = {
    RiskLevel.CRITICAL: "❌",
    RiskLevel.WARNING:  "⚠️ ",
    RiskLevel.OK:       "✅",
    RiskLevel.INFO:     "ℹ️ ",
}


class ObservedValue(BaseModel):
    """
    A scalar measurement with an optional bootstrap confidence interval.

    Use `unit` for physical units ("ms", "bits", "fraction").
    Omit ci_lower/ci_upper when the measurement is deterministic (e.g. a
    single episode count) or when sample size makes bootstrapping meaningless.
    """

    value: Optional[float] = None   # None = metric could not be computed
    unit: str = ""
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    ci_method: str = ""   # "bootstrap", "t-distribution", …

    @model_validator(mode="after")
    def _bounds_ordered(self) -> "ObservedValue":
        if self.ci_lower is not None and self.ci_upper is not None:
            if self.ci_lower > self.ci_upper:
                raise ValueError("ci_lower must be ≤ ci_upper")
        return self

    def __str__(self) -> str:
        if self.value is None:
            return "n/a"
        s = f"{self.value:.4g}"
        if self.unit:
            s += f" {self.unit}"
        if self.ci_lower is not None and self.ci_upper is not None:
            s += (
                f" [{self.ci_lower:.4g}, {self.ci_upper:.4g}]"
                f" ({self.ci_level:.0%} CI via {self.ci_method or 'unknown'})"
            )
        return s


class RiskFlag(BaseModel):
    """
    A single diagnostic finding produced by an analyzer.

    `interpretation` explains what was measured.
    `implication`    explains the downstream risk for training.
    `affected_fraction` is fraction of episodes or steps that triggered this flag.
    """

    level: RiskLevel
    metric: str
    observed: ObservedValue
    threshold: Optional[float] = None
    interpretation: str
    implication: str
    affected_fraction: Optional[float] = None

    def render(self) -> str:
        icon = _ICONS[self.level]
        header = f"{icon} {self.level.value}: {self.metric} = {self.observed}"
        if self.threshold is not None:
            header += f" (threshold: {self.threshold:.4g} {self.observed.unit})"
        if self.affected_fraction is not None:
            header += f"  [{self.affected_fraction:.1%} of samples]"
        return f"{header}\n   → {self.implication}"


class CompatibilityHint(BaseModel):
    """
    Structural compatibility signal for a specific policy family.

    `compatible` is a tri-state: True / False / None (ambiguous).
    Hints are only emitted when a `policy_family` is specified at analysis time.
    """

    policy_family: str
    compatible: Optional[bool] = None
    explanation: str
    caveats: list[str] = []


class AnalyzerResult(BaseModel):
    """Output contract that every Analyzer must satisfy."""

    analyzer_name: str
    flags: list[RiskFlag] = []
    hints: list[CompatibilityHint] = []
    raw_metrics: dict[str, Any] = {}   # untyped bag for downstream consumers


class DiagnosticReport(BaseModel):
    """
    Top-level report assembled by the pipeline from one or more AnalyzerResults.
    """

    dataset_name: str
    source_path: str
    format: str
    n_episodes: int
    n_samples: int
    analyzer_results: list[AnalyzerResult] = []
    policy_family: Optional[str] = None

    # ── convenience accessors ───────────────────────────────────────────────

    @property
    def flags(self) -> list[RiskFlag]:
        return [f for r in self.analyzer_results for f in r.flags]

    @property
    def hints(self) -> list[CompatibilityHint]:
        return [h for r in self.analyzer_results for h in r.hints]

    def flags_at_level(self, level: RiskLevel) -> list[RiskFlag]:
        return [f for f in self.flags if f.level == level]

    # ── rendering ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines: list[str] = [
            "=== Calibra Diagnostic Report ===",
            f"Dataset  : {self.dataset_name}",
            f"Format   : {self.format}",
            f"Episodes : {self.n_episodes}  |  Samples: {self.n_samples}",
        ]
        if self.policy_family:
            lines.append(f"Policy   : {self.policy_family}")
        lines.append("")

        sorted_flags = sorted(self.flags, key=lambda f: _LEVEL_ORDER[f.level])
        for flag in sorted_flags:
            lines.append(flag.render())
            lines.append("")

        if self.hints:
            lines.append("--- Policy Compatibility ---")
            compat_icon = {True: "✅", False: "❌", None: "⚠️ "}
            for hint in self.hints:
                lines.append(
                    f"{compat_icon[hint.compatible]} {hint.policy_family}: {hint.explanation}"
                )
                for caveat in hint.caveats:
                    lines.append(f"     • {caveat}")
            lines.append("")

        n_crit = len(self.flags_at_level(RiskLevel.CRITICAL))
        n_warn = len(self.flags_at_level(RiskLevel.WARNING))
        lines.append(f"{n_crit} critical  ·  {n_warn} warnings")
        return "\n".join(lines)
