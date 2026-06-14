"""Calibra — dataset reliability and risk profiling for robotics IL pipelines."""

from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import (
    AnalyzerResult,
    CompatibilityHint,
    DiagnosticReport,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)
from calibra.ingestion.registry import load, registered_formats
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.coverage import CoverageEntropyAnalyzer
from calibra.pipeline import Pipeline

__all__ = [
    "Episode",
    "EpisodeBatch",
    "EpisodeMetadata",
    "AnalyzerResult",
    "CompatibilityHint",
    "DiagnosticReport",
    "ObservedValue",
    "RiskFlag",
    "RiskLevel",
    "load",
    "registered_formats",
    "TemporalAnalyzer",
    "ControlSmoothnessAnalyzer",
    "CoverageEntropyAnalyzer",
    "Pipeline",
]
