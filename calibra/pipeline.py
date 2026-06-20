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

import time
from typing import Optional

from calibra.analyzers.base import Analyzer
from calibra.analyzers.coverage import CoverageEntropyAnalyzer
from calibra.analyzers.gr00t import GR00TCompatibilityAnalyzer
from calibra.analyzers.influence import InfluenceAnalyzer
from calibra.analyzers.octo import OctoCompatibilityAnalyzer
from calibra.analyzers.openvla import OpenVLACompatibilityAnalyzer
from calibra.analyzers.phase_balance import PhaseBalanceAnalyzer
from calibra.analyzers.pi0 import Pi0CompatibilityAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.analyzers.transition_dynamics import TransitionDynamicsAnalyzer
from calibra.analyzers.latent_dynamics import LatentDynamicsAnalyzer
from calibra.analyzers.ssl_embed import SSLTrajectoryEmbedderAnalyzer
from calibra.analyzers.force_torque import ForceTorqueContactAnalyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport


def _default_analyzers() -> list[Analyzer]:
    return [
        TemporalAnalyzer(),
        ControlSmoothnessAnalyzer(),
        CoverageEntropyAnalyzer(),
        TaskStructureAnalyzer(),
        PhaseBalanceAnalyzer(),
        InfluenceAnalyzer(),
        TransitionDynamicsAnalyzer(),
        LatentDynamicsAnalyzer(),
        SSLTrajectoryEmbedderAnalyzer(),
        ForceTorqueContactAnalyzer(),
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
        analyzers = list(self.analyzers)
        pf_lower = policy_family.lower() if policy_family else ""
        if pf_lower and "gr00t" in pf_lower:
            analyzers.append(GR00TCompatibilityAnalyzer())
        if pf_lower and "pi0" in pf_lower:
            analyzers.append(Pi0CompatibilityAnalyzer())
        if pf_lower and "openvla" in pf_lower:
            analyzers.append(OpenVLACompatibilityAnalyzer())
        if pf_lower and "octo" in pf_lower:
            analyzers.append(OctoCompatibilityAnalyzer())

        results = []
        timing: dict[str, float] = {}
        for analyzer in analyzers:
            t0 = time.perf_counter()
            results.append(analyzer.analyze(batch, policy_family=policy_family))
            timing[analyzer.name] = round(time.perf_counter() - t0, 4)

        return DiagnosticReport(
            dataset_name=batch.dataset_name,
            source_path=batch.source_path,
            format=batch.format,
            n_episodes=batch.n_episodes,
            n_samples=batch.n_samples,
            analyzer_results=results,
            policy_family=policy_family,
            episode_ids=[ep.metadata.episode_id for ep in batch.episodes],
            timing=timing,
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
