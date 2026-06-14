"""
Calibra Pipeline — assembles multiple analyzers into a single DiagnosticReport.

Usage:
    from calibra.pipeline import Pipeline

    report = Pipeline().run(batch, policy_family="diffusion")
    print(report.summary())

    # Or from a file path (auto-detects format):
    report = Pipeline().analyze_path("/data/my_dataset.h5", policy_family="act")
"""
from __future__ import annotations

from typing import Optional

from calibra.analyzers.base import Analyzer
from calibra.analyzers.coverage import CoverageEntropyAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport


def _default_analyzers() -> list[Analyzer]:
    return [
        TemporalAnalyzer(),
        ControlSmoothnessAnalyzer(),
        CoverageEntropyAnalyzer(),
        TaskStructureAnalyzer(),
    ]


class Pipeline:
    """
    Runs a configurable list of analyzers over an EpisodeBatch and
    assembles the results into a single DiagnosticReport.

    Parameters
    ----------
    analyzers : list of Analyzer instances to run, in order.
                Defaults to [TemporalAnalyzer, ControlSmoothnessAnalyzer,
                CoverageEntropyAnalyzer].
    """

    def __init__(self, analyzers: Optional[list[Analyzer]] = None) -> None:
        self.analyzers: list[Analyzer] = (
            analyzers if analyzers is not None else _default_analyzers()
        )

    def run(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> DiagnosticReport:
        """
        Run all analyzers over `batch` and return a DiagnosticReport.

        Parameters
        ----------
        batch         : normalized dataset from the ingestion layer.
        policy_family : optional target policy for conditioned hints
                        (e.g. "diffusion", "act", "transformer").
        """
        results = [
            analyzer.analyze(batch, policy_family=policy_family)
            for analyzer in self.analyzers
        ]
        return DiagnosticReport(
            dataset_name=batch.dataset_name,
            source_path=batch.source_path,
            format=batch.format,
            n_episodes=batch.n_episodes,
            n_samples=batch.n_samples,
            analyzer_results=results,
            policy_family=policy_family,
            episode_ids=[ep.metadata.episode_id for ep in batch.episodes],
        )

    def analyze_path(
        self,
        path: str,
        policy_family: Optional[str] = None,
        reader=None,
    ) -> DiagnosticReport:
        """
        Load a dataset from `path` (auto-detecting format) and run the pipeline.

        Parameters
        ----------
        path          : filesystem path to the dataset directory or file.
        policy_family : optional target policy for conditioned hints.
        reader        : optional DatasetReader instance to bypass auto-detection.
        """
        from calibra.ingestion.registry import load
        batch = load(path, reader=reader)
        return self.run(batch, policy_family=policy_family)
