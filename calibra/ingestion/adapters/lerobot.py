"""
LeRobot adapter.

LeRobot stores datasets as HuggingFace datasets (Parquet shards) with a
standard schema:

  observation.images.<camera>   (T,) — encoded frames or paths
  observation.state             (T, state_dim)
  action                        (T, action_dim)
  timestamp                     (T,)
  episode_index                 (T,)   — integer episode ID per step

Dataset root contains a metadata.json with fps and task descriptions.

Dependency: pip install 'calibra[lerobot]'  (datasets, pyarrow)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

if TYPE_CHECKING:
    import datasets as _hf_datasets


def _require_datasets() -> "_hf_datasets":
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for the LeRobot adapter.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None


@register
class LeRobotReader(DatasetReader):
    """Reads LeRobot-format HuggingFace datasets."""

    @property
    def format_name(self) -> str:
        return "lerobot"

    @classmethod
    def can_read(cls, path: str) -> bool:
        p = Path(path)
        return p.is_dir() and (
            (p / "metadata.json").exists()
            or (p / "dataset_dict.json").exists()
            or any(p.glob("*.parquet"))
        )

    def read(self, path: str) -> EpisodeBatch:
        hf = _require_datasets()
        p = Path(path)

        ds = hf.load_from_disk(str(p))
        meta_path = p / "metadata.json"
        task = None
        if meta_path.exists():
            with open(meta_path) as f:
                task = json.load(f).get("task_description")

        # Group rows by episode_index.
        episode_col = "episode_index"
        if episode_col not in ds.column_names:
            raise ValueError(
                f"Expected column '{episode_col}' in LeRobot dataset at '{path}'.\n"
                f"Available columns: {ds.column_names}"
            )

        episodes: list[Episode] = []
        for ep_id in sorted(set(ds[episode_col])):
            subset = ds.filter(lambda row, eid=ep_id: row[episode_col] == eid)
            episodes.append(self._episode_from_subset(subset, ep_id, task, path))

        return EpisodeBatch(
            episodes=episodes,
            dataset_name=p.name,
            format=self.format_name,
            source_path=str(p),
        )

    @staticmethod
    def _episode_from_subset(subset: "_hf_datasets.Dataset", ep_id: int,
                              task: str | None, source: str) -> Episode:
        timestamps = np.array(subset["timestamp"], dtype=np.float64)
        actions = np.array(subset["action"], dtype=np.float32)

        obs: dict[str, np.ndarray] = {}
        for col in subset.column_names:
            if col.startswith("observation."):
                key = col.removeprefix("observation.")
                obs[key] = np.array(subset[col])

        return Episode(
            metadata=EpisodeMetadata(
                episode_id=str(ep_id),
                task_description=task,
                source_file=source,
            ),
            timestamps=timestamps,
            observations=obs,
            actions=actions,
        )
