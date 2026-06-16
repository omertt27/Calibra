__version__ = "0.4.0"

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
from calibra.schema.comparison import (
    ComparisonReport,
    CurationReport,
    DriftFlag,
    EpisodeFlag,
)
from calibra.ingestion.registry import load, registered_formats
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.coverage import CoverageEntropyAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.pipeline import Pipeline
from calibra.comparison import DatasetComparator, EpisodeCurator

__all__ = [
    # schema
    "Episode",
    "EpisodeBatch",
    "EpisodeMetadata",
    "AnalyzerResult",
    "CompatibilityHint",
    "DiagnosticReport",
    "ObservedValue",
    "RiskFlag",
    "RiskLevel",
    # phase 2 schema
    "ComparisonReport",
    "CurationReport",
    "DriftFlag",
    "EpisodeFlag",
    # ingestion
    "load",
    "registered_formats",
    # analyzers
    "TemporalAnalyzer",
    "ControlSmoothnessAnalyzer",
    "CoverageEntropyAnalyzer",
    "TaskStructureAnalyzer",
    # pipeline
    "Pipeline",
    # phase 2
    "DatasetComparator",
    "EpisodeCurator",
]
