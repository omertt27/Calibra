"""
DatasetReader — the only interface between format-specific adapters and the
diagnostic layer. Analyzers receive EpisodeBatch; they never see this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from calibra.schema.episode import EpisodeBatch


class DatasetReader(ABC):
    """
    Abstract adapter that maps a native format to EpisodeBatch.

    Subclasses must implement:
      read(path)   — load and normalize
      can_read(path) — fast heuristic to check format compatibility
      format_name  — short string identifier

    No format-specific state or logic should exist outside this class hierarchy.
    """

    @abstractmethod
    def read(self, path: str) -> EpisodeBatch:
        """Load the dataset at `path` and return a normalized EpisodeBatch."""
        ...

    @classmethod
    @abstractmethod
    def can_read(cls, path: str) -> bool:
        """Return True if this reader can handle the given path."""
        ...

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Short format identifier matching EpisodeBatch.format."""
        ...
