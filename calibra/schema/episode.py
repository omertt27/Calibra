"""
Internal normalized episode representation. All analyzers consume this type.
No format-specific logic belongs here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EpisodeMetadata:
    episode_id: str
    task_description: Optional[str] = None
    success: Optional[bool] = None
    source_file: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Episode:
    """
    Single normalized episode.

    timestamps     : master clock, shape (T,), seconds (relative or epoch).
    observations   : modality name → array (T, ...). Keys are format-agnostic
                     names like "camera_rgb", "proprio", "depth".
    obs_timestamps : per-modality timestamps when the sensor clock differs from
                     the master clock. Absent key means use `timestamps`.
    actions        : shape (T, action_dim).
    action_timestamps : when the action log has its own clock; None → use
                        `timestamps`.
    """

    metadata: EpisodeMetadata
    timestamps: np.ndarray                        # (T,)
    observations: dict[str, np.ndarray]
    actions: np.ndarray                           # (T, action_dim)
    obs_timestamps: dict[str, np.ndarray] = field(default_factory=dict)
    action_timestamps: Optional[np.ndarray] = None

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        if self.n_steps < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1] if self.actions.ndim > 1 else 1


@dataclass
class EpisodeBatch:
    """
    Collection of normalized episodes from a single dataset load.
    This is the only type that analyzers accept; they never see raw format data.
    """

    episodes: list[Episode]
    dataset_name: str
    format: str        # "rlds" | "lerobot" | "hdf5" | "mcap"
    source_path: str
    extra: dict = field(default_factory=dict)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def n_samples(self) -> int:
        return sum(ep.n_steps for ep in self.episodes)

    @property
    def modalities(self) -> set[str]:
        keys: set[str] = set()
        for ep in self.episodes:
            keys.update(ep.observations.keys())
        return keys
