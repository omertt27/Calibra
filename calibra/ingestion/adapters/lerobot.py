"""
LeRobot adapter.

LeRobot stores datasets as HuggingFace datasets (Parquet shards) with a
standard schema:

  observation.images.<camera>   (T,) — encoded frames or paths
  observation.state             (T, state_dim)
  action                        (T, action_dim)
  timestamp                     (T,)
  episode_index                 (T,)   — integer episode ID per step
  frame_index                   (T,)   — step index within episode

Supports:
  - HuggingFace Hub repo IDs:  "lerobot/pusht"
  - Local disk (v1):  directory with metadata.json or dataset_dict.json
  - Local disk (v2):  directory with meta/info.json

Image feature columns are skipped automatically; only scalar/sequence columns
are loaded into the EpisodeBatch.

Dependency: pip install 'calibra[lerobot]'  (datasets, pyarrow)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

if TYPE_CHECKING:
    import datasets as _hf_datasets
    import pandas as _pd


def _require_datasets() -> "_hf_datasets":
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for the LeRobot adapter.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None


def _is_hub_id(path: str) -> bool:
    """
    True for strings like "lerobot/pusht" that are Hub repo IDs rather than
    local filesystem paths. Heuristic: path doesn't exist locally, contains
    exactly one "/" with non-empty parts on both sides, and no filesystem
    indicators (backslash, drive letter, known extensions).
    """
    p = Path(path)
    if p.exists():
        return False
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        return False
    return not any(c in path for c in ("\\", ":", ".parquet", ".h5", ".hdf5", ".json"))


@register
class LeRobotReader(DatasetReader):
    """Reads LeRobot-format HuggingFace datasets from Hub or local disk."""

    @property
    def format_name(self) -> str:
        return "lerobot"

    @classmethod
    def can_read(cls, path: str) -> bool:
        if _is_hub_id(path):
            return True
        p = Path(path)
        return p.is_dir() and (
            (p / "metadata.json").exists()
            or (p / "dataset_dict.json").exists()
            or (p / "meta" / "info.json").exists()
            or any(p.glob("*.parquet"))
            or any(p.glob("data/*.parquet"))
        )

    def read(self, path: str) -> EpisodeBatch:
        hf = _require_datasets()

        if _is_hub_id(path):
            ds, dataset_name, task = self._load_hub(hf, path)
        else:
            ds, dataset_name, task = self._load_local(hf, Path(path))

        # Drop image feature columns — we only need scalar/sequence data.
        scalar_cols = self._scalar_columns(hf, ds)
        ds = ds.select_columns(scalar_cols)

        episode_col = "episode_index"
        if episode_col not in ds.column_names:
            raise ValueError(
                f"Expected column '{episode_col}' in LeRobot dataset at '{path}'.\n"
                f"Available columns: {ds.column_names}"
            )

        # O(n) groupby via pandas — avoids O(n_episodes × n_steps) repeated filter.
        df = ds.to_pandas()
        episodes: list[Episode] = []
        for ep_id, group in df.groupby(episode_col, sort=True):
            episodes.append(self._episode_from_group(group, ep_id, task, path))

        return EpisodeBatch(
            episodes=episodes,
            dataset_name=dataset_name,
            format=self.format_name,
            source_path=path,
        )

    # ── loading helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_hub(
        hf: "_hf_datasets", path: str
    ) -> tuple["_hf_datasets.Dataset", str, Optional[str]]:
        dataset_name = path.split("/")[-1]
        try:
            ds = hf.load_dataset(path, split="train")
        except Exception:
            # Some Hub datasets use a different split name; fall back to all splits.
            dd = hf.load_dataset(path)
            split = next(iter(dd))
            ds = dd[split]
        return ds, dataset_name, None

    @staticmethod
    def _load_local(
        hf: "_hf_datasets", p: Path
    ) -> tuple["_hf_datasets.Dataset", str, Optional[str]]:
        # v2 format: meta/info.json with parquet in data/
        if (p / "meta" / "info.json").exists():
            parquet_files = sorted(p.glob("data/**/*.parquet"))
            if not parquet_files:
                parquet_files = sorted(p.glob("*.parquet"))
            ds = hf.load_dataset(
                "parquet",
                data_files={"train": [str(f) for f in parquet_files]},
                split="train",
            )
            task = _read_task_v2(p)
        else:
            ds = hf.load_from_disk(str(p))
            task = _read_task_v1(p)

        # load_from_disk returns DatasetDict for some saved formats
        if hasattr(ds, "keys"):
            split = next(iter(ds))
            ds = ds[split]

        return ds, p.name, task

    # ── column filtering ─────────────────────────────────────────────────────

    @staticmethod
    def _scalar_columns(hf: "_hf_datasets", ds: "_hf_datasets.Dataset") -> list[str]:
        """Return columns that are not HuggingFace Image features."""
        try:
            from datasets import Image as HFImage
            return [
                col for col, feat in ds.features.items()
                if not isinstance(feat, HFImage)
            ]
        except ImportError:
            return [col for col in ds.column_names if "image" not in col.lower()]

    # ── episode construction ─────────────────────────────────────────────────

    @staticmethod
    def _episode_from_group(
        group: "_pd.DataFrame",
        ep_id: int,
        task: Optional[str],
        source: str,
    ) -> Episode:
        if "frame_index" in group.columns:
            group = group.sort_values("frame_index")

        timestamps = group["timestamp"].to_numpy(dtype=np.float64)
        actions = np.array(group["action"].tolist(), dtype=np.float32)

        obs: dict[str, np.ndarray] = {}
        for col in group.columns:
            if col.startswith("observation."):
                key = col.removeprefix("observation.")
                try:
                    obs[key] = np.array(group[col].tolist(), dtype=np.float32)
                except (ValueError, TypeError):
                    pass  # skip non-numeric residuals (shouldn't reach here after column filter)

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


# ── metadata readers ─────────────────────────────────────────────────────────

def _read_task_v1(p: Path) -> Optional[str]:
    meta = p / "metadata.json"
    if meta.exists():
        with open(meta) as f:
            return json.load(f).get("task_description")
    return None


def _read_task_v2(p: Path) -> Optional[str]:
    info = p / "meta" / "info.json"
    if info.exists():
        with open(info) as f:
            data = json.load(f)
            return data.get("task_description") or data.get("description")
    tasks = p / "meta" / "tasks.jsonl"
    if tasks.exists():
        with open(tasks) as f:
            first = f.readline()
            if first:
                return json.loads(first).get("task")
    return None