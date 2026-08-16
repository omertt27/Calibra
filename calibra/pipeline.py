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

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Optional

from calibra.analyzers.base import Analyzer
from calibra.analyzers.coverage import CoverageEntropyAnalyzer
from calibra.analyzers.force_torque import ForceTorqueContactAnalyzer
from calibra.analyzers.gr00t import GR00TCompatibilityAnalyzer
from calibra.analyzers.influence import InfluenceAnalyzer
from calibra.analyzers.latent_dynamics import LatentDynamicsAnalyzer
from calibra.analyzers.octo import OctoCompatibilityAnalyzer
from calibra.analyzers.openvla import OpenVLACompatibilityAnalyzer
from calibra.analyzers.phase_balance import PhaseBalanceAnalyzer
from calibra.analyzers.pi0 import Pi0CompatibilityAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.ssl_embed import SSLTrajectoryEmbedderAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.analyzers.transition_dynamics import TransitionDynamicsAnalyzer
from calibra.analyzers.world_model import WorldModelConsistencyAnalyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport


def _config_hash(calibra_version: str, policy_family: Optional[str], analyzer_versions: dict) -> str:
    """
    Deterministic fingerprint of "what produced this report" — the Calibra
    version, target policy family, and the exact analyzer/version set that
    ran. Two reports with matching config_hash were built by the same
    analysis logic, so their numbers are safe to compare directly; a mismatch
    is a signal to check *why* before treating a delta as a real finding.
    """
    payload = {
        "calibra_version": calibra_version,
        "policy_family": policy_family or "",
        "analyzers": sorted(analyzer_versions.items()),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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


def _fast_analyzers() -> list[Analyzer]:
    """
    Action/timestamp-only diagnostics: no pairwise embedding distances
    (SSLTrajectoryEmbedderAnalyzer, InfluenceAnalyzer), no model fitting
    (TransitionDynamicsAnalyzer, LatentDynamicsAnalyzer). Cheap, linear-time
    metrics only — trades away coverage_value and some anomaly reasons for
    speed on very large datasets.
    """
    return [
        TemporalAnalyzer(),
        ControlSmoothnessAnalyzer(),
        CoverageEntropyAnalyzer(),
    ]


class Pipeline:
    """
    Runs a configurable list of analyzers over an EpisodeBatch and
    assembles the results into a single DiagnosticReport.

    Parameters
    ----------
    analyzers : list of Analyzer instances to run, in order. Overrides `mode`.
    mode      : "full" (default) runs every analyzer; "fast" restricts to
                cheap, linear-time action/timestamp diagnostics (see
                _fast_analyzers). Ignored when `analyzers` is given.
    """

    def __init__(
        self,
        analyzers: Optional[list[Analyzer]] = None,
        world_model: bool = False,
        mode: str = "full",
    ) -> None:
        if analyzers is not None:
            self.analyzers: list[Analyzer] = analyzers
        elif mode == "fast":
            self.analyzers = _fast_analyzers()
        elif mode == "full":
            self.analyzers = _default_analyzers()
        else:
            raise ValueError(f"mode must be 'fast' or 'full', got {mode!r}")
        if world_model:
            self.analyzers = list(self.analyzers) + [WorldModelConsistencyAnalyzer()]

    def run(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
        cache=None,
    ) -> DiagnosticReport:
        """
        Run all analyzers over `batch` and return a DiagnosticReport.

        Parameters
        ----------
        batch         : normalized dataset from the ingestion layer.
        policy_family : optional target policy for conditioned hints
                        (e.g. "diffusion", "act", "transformer").
        cache         : optional AuditCache instance. On hit, returns cached
                        result instantly. On miss, runs pipeline and stores result.
        """
        if cache is not None:
            fingerprint = cache.fingerprint(batch, policy_family)
            cached = cache.get(fingerprint)
            if cached is not None:
                return cached

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

        capabilities = batch.capabilities
        results = []
        timing: dict[str, float] = {}
        skipped: list[str] = []
        analyzer_versions: dict[str, str] = {}
        for analyzer in analyzers:
            if not analyzer.requires <= capabilities:
                skipped.append(analyzer.name)
                continue
            t0 = time.perf_counter()
            results.append(analyzer.analyze(batch, policy_family=policy_family))
            timing[analyzer.name] = round(time.perf_counter() - t0, 4)
            analyzer_versions[analyzer.name] = analyzer.version

        from calibra import __version__ as calibra_version

        report = DiagnosticReport(
            dataset_name=batch.dataset_name,
            source_path=batch.source_path,
            format=batch.format,
            n_episodes=batch.n_episodes,
            n_samples=batch.n_samples,
            analyzer_results=results,
            policy_family=policy_family,
            episode_ids=[ep.metadata.episode_id for ep in batch.episodes],
            timing=timing,
            skipped_analyzers=skipped,
            calibra_version=calibra_version,
            analyzer_versions=analyzer_versions,
            config_hash=_config_hash(calibra_version, policy_family, analyzer_versions),
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        if cache is not None:
            cache.put(fingerprint, report)

        return report

    def analyze_path(
        self,
        path: str,
        policy_family: Optional[str] = None,
        reader=None,
        cache=None,
    ) -> DiagnosticReport:
        """
        Load a dataset from `path` (auto-detecting format) and run the pipeline.

        Parameters
        ----------
        path          : filesystem path to the dataset directory or file.
        policy_family : optional target policy for conditioned hints.
        reader        : optional DatasetReader instance to bypass auto-detection.
        cache         : optional AuditCache. The loaded batch is stored in
                        ``self._last_batch`` for callers that need episode hashes.
        """
        from calibra.ingestion.registry import load

        batch = load(path, reader=reader)
        self._last_batch = batch
        return self.run(batch, policy_family=policy_family, cache=cache)
