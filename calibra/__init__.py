"""Calibra — dataset reliability and risk profiling for robotics IL pipelines."""

__version__ = "0.5.0"

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
from calibra.analyzers.gr00t import GR00TCompatibilityAnalyzer
from calibra.analyzers.pi0 import Pi0CompatibilityAnalyzer
from calibra.analyzers.openvla import OpenVLACompatibilityAnalyzer
from calibra.analyzers.octo import OctoCompatibilityAnalyzer
from calibra.pipeline import Pipeline
from calibra.comparison import DatasetComparator, EpisodeCurator
from calibra.score import compute_score
from calibra.predict import predict_outcome
from calibra.sim2real import analyze_gap
from calibra.transfer import analyze_transfer

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
    "GR00TCompatibilityAnalyzer",
    "Pi0CompatibilityAnalyzer",
    "OpenVLACompatibilityAnalyzer",
    "OctoCompatibilityAnalyzer",
    # pipeline
    "Pipeline",
    # phase 2
    "DatasetComparator",
    "EpisodeCurator",
    # scoring / analysis
    "compute_score",
    "predict_outcome",
    "analyze_gap",
    "analyze_transfer",
]
