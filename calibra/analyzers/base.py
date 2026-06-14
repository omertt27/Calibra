"""
Analyzer base class. Every diagnostic module implements this contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult


class Analyzer(ABC):
    """
    Stateless diagnostic unit.

    Each analyzer receives an EpisodeBatch, computes metrics, and returns
    an AnalyzerResult containing RiskFlags and optional CompatibilityHints.

    Analyzers must not modify the EpisodeBatch or retain state between calls.
    """

    @abstractmethod
    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        """
        Run diagnostics on `batch`.

        Parameters
        ----------
        batch         : normalized dataset from the ingestion layer.
        policy_family : optional target policy (e.g. "diffusion", "act",
                        "transformer"). When provided, emit CompatibilityHints
                        tailored to that policy's inductive biases.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this analyzer, used in AnalyzerResult."""
        ...
