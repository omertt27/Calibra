"""
Internal normalized episode representation. All analyzers consume this type.
No format-specific logic belongs here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Iterable
from collections.abc import Sequence

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


class LazyEpisodeList(Sequence):
    """
    A list-like interface that loads episodes lazily on demand.
    Supports random access via loader_fn(idx) and sequential streaming
    via iterator_fn() to optimize tf.data.Dataset iterations.
    """

    def __init__(
        self,
        loader_fn: Callable[[int], Episode],
        length: int,
        iterator_fn: Optional[Callable[[], Iterable[Episode]]] = None,
        cache_size: int = 100,
    ) -> None:
        self._loader_fn = loader_fn
        self._length = length
        self._iterator_fn = iterator_fn
        self._cache: dict[int, Episode] = {}
        self._cache_size = cache_size
        self._access_order: list[int] = []

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterable[Episode]:
        if self._iterator_fn is not None:
            yield from self._iterator_fn()
        else:
            for i in range(self._length):
                yield self[i]

    def __getitem__(self, idx: int | slice) -> Union[Episode, list[Episode]]:
        if isinstance(idx, slice):
            indices = idx.indices(self._length)
            # Safe downcast to handle slice
            res = [self[i] for i in range(*indices)]
            return res  # type: ignore
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("Episode index out of range")

        if idx in self._cache:
            self._access_order.remove(idx)
            self._access_order.append(idx)
            return self._cache[idx]

        ep = self._loader_fn(idx)
        self._cache[idx] = ep
        self._access_order.append(idx)

        if len(self._cache) > self._cache_size:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        return ep


@dataclass
class EpisodeBatch:
    """
    Collection of normalized episodes from a single dataset load.
    This is the only type that analyzers accept; they never see raw format data.
    """

    episodes: Union[list[Episode], LazyEpisodeList]
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
