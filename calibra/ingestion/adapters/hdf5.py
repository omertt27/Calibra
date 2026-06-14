"""
HDF5 adapter.

Expects datasets in one of two common robotics conventions:

  Convention A — flat group per episode:
    /episode_0/observations/camera_rgb   (T, H, W, C)
    /episode_0/observations/proprio      (T, D)
    /episode_0/actions                   (T, action_dim)
    /episode_0/timestamps                (T,)

  Convention B — parallel datasets at root:
    /observations/camera_rgb             (N, H, W, C)   # N = total steps
    /observations/proprio                (N, D)
    /actions                             (N, action_dim)
    /timestamps                          (N,)
    /episode_ends                        (E,)            # cumulative step counts

If neither convention is detected, a ValueError is raised with a description
of what was found so the user can write a thin adapter subclass.

Dependency: pip install 'calibra[hdf5]'  (h5py)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

if TYPE_CHECKING:
    import h5py as _h5py


def _require_h5py() -> "_h5py":
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for the HDF5 adapter.\n"
            "Install it with: pip install 'calibra[hdf5]'"
        ) from None


@register
class HDF5Reader(DatasetReader):
    """Reads robotics HDF5 datasets (convention A or B)."""

    @property
    def format_name(self) -> str:
        return "hdf5"

    @classmethod
    def can_read(cls, path: str) -> bool:
        p = Path(path)
        if p.is_file() and p.suffix in (".hdf5", ".h5"):
            return True
        if p.is_dir():
            return any(p.glob("*.hdf5")) or any(p.glob("*.h5"))
        return False

    def read(self, path: str) -> EpisodeBatch:
        h5py = _require_h5py()
        p = Path(path)

        # If path is a directory, collect all .hdf5 / .h5 files.
        if p.is_dir():
            files = sorted(p.glob("*.hdf5")) + sorted(p.glob("*.h5"))
            if not files:
                raise ValueError(f"No HDF5 files found in {path}")
            episodes: list[Episode] = []
            for f in files:
                eps = self._read_file(h5py, str(f))
                episodes.extend(eps)
            dataset_name = p.name
        else:
            episodes = self._read_file(h5py, str(p))
            dataset_name = p.stem

        return EpisodeBatch(
            episodes=episodes,
            dataset_name=dataset_name,
            format=self.format_name,
            source_path=str(p),
        )

    # ── internal ────────────────────────────────────────────────────────────

    def _read_file(self, h5py: "_h5py", path: str) -> list[Episode]:
        with h5py.File(path, "r") as f:
            if self._is_convention_a(f):
                return self._read_convention_a(f, path)
            elif self._is_convention_b(f):
                return self._read_convention_b(f, path)
            else:
                keys = list(f.keys())
                raise ValueError(
                    f"Unrecognised HDF5 layout in '{path}'.\n"
                    f"Top-level keys: {keys}\n"
                    "Expected convention A (episode_N groups with observations/, "
                    "actions, timestamps) or convention B (parallel flat arrays "
                    "with episode_ends index).\n"
                    "Subclass HDF5Reader and override _read_file to handle "
                    "custom layouts."
                )

    @staticmethod
    def _is_convention_a(f: "_h5py.File") -> bool:
        return any(k.startswith("episode") for k in f.keys())

    @staticmethod
    def _is_convention_b(f: "_h5py.File") -> bool:
        return "episode_ends" in f or ("actions" in f and "timestamps" in f)

    def _read_convention_a(self, f: "_h5py.File", source: str) -> list[Episode]:
        episodes: list[Episode] = []
        ep_keys = sorted(k for k in f.keys() if k.startswith("episode"))
        for key in ep_keys:
            grp = f[key]
            timestamps = self._load_timestamps(grp)
            actions = np.array(grp["actions"], dtype=np.float32)
            observations = self._load_obs_group(grp)
            success = bool(grp.attrs["success"]) if "success" in grp.attrs else None
            task = str(grp.attrs["task"]) if "task" in grp.attrs else None
            meta = EpisodeMetadata(
                episode_id=key,
                task_description=task,
                success=success,
                source_file=source,
            )
            episodes.append(Episode(
                metadata=meta,
                timestamps=timestamps,
                observations=observations,
                actions=actions,
            ))
        return episodes

    def _read_convention_b(self, f: "_h5py.File", source: str) -> list[Episode]:
        actions_all = np.array(f["actions"], dtype=np.float32)
        timestamps_all = self._load_timestamps(f)
        obs_all = self._load_obs_group(f)
        episode_ends: np.ndarray = np.array(f["episode_ends"], dtype=np.int64)

        episodes: list[Episode] = []
        starts = np.concatenate([[0], episode_ends[:-1]])
        for i, (start, end) in enumerate(zip(starts, episode_ends)):
            sl = slice(int(start), int(end))
            obs = {k: v[sl] for k, v in obs_all.items()}
            meta = EpisodeMetadata(
                episode_id=f"episode_{i}",
                source_file=source,
            )
            episodes.append(Episode(
                metadata=meta,
                timestamps=timestamps_all[sl],
                observations=obs,
                actions=actions_all[sl],
            ))
        return episodes

    @staticmethod
    def _load_timestamps(grp: "_h5py.Group | _h5py.File") -> np.ndarray:
        for key in ("timestamps", "timestamp", "t"):
            if key in grp:
                return np.array(grp[key], dtype=np.float64)
        # Synthesize from step count if timestamps are absent.
        n = len(next(iter(grp.values())))  # type: ignore[arg-type]
        return np.arange(n, dtype=np.float64)

    @staticmethod
    def _load_obs_group(
        grp: "_h5py.Group | _h5py.File",
    ) -> dict[str, np.ndarray]:
        import h5py
        obs: dict[str, np.ndarray] = {}
        root = grp["observations"] if "observations" in grp else grp
        if isinstance(root, h5py.Group):
            for k in root.keys():
                if isinstance(root[k], h5py.Dataset):
                    obs[k] = np.array(root[k])
        return obs
