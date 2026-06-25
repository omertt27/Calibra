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
  - HuggingFace Hub URIs:      "hf://lerobot/pusht"
  - Local disk (v1):  directory with metadata.json or dataset_dict.json
  - Local disk (v2):  directory with meta/info.json + Parquet shards
                      (fast path: DuckDB reads Parquet directly without
                       loading image columns into RAM)

Image feature columns are skipped automatically; only scalar/sequence columns
are loaded into the EpisodeBatch.

Dependencies:
  pip install 'calibra[lerobot]'  (datasets, pyarrow, duckdb)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.normalization import normalize_obs_keys

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


def _require_duckdb():
    try:
        import duckdb

        return duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for fast local LeRobot Parquet scanning.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None


def _is_hub_uri(path: str) -> bool:
    """True for 'hf://lerobot/pusht' style URIs."""
    return path.startswith("hf://")


def _strip_hf_prefix(path: str) -> str:
    """Remove 'hf://' prefix if present, returning the bare repo ID."""
    return path[len("hf://") :] if _is_hub_uri(path) else path


def _is_hub_id(path: str) -> bool:
    """
    True for strings like "lerobot/pusht" that are Hub repo IDs rather than
    local filesystem paths. Heuristic: path doesn't exist locally, contains
    exactly one "/" with non-empty parts on both sides, and no filesystem
    indicators (backslash, drive letter, known extensions).
    """
    if _is_hub_uri(path):
        return True
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
        p = Path(_strip_hf_prefix(path))
        return p.is_dir() and (
            (p / "metadata.json").exists()
            or (p / "dataset_dict.json").exists()
            or (p / "meta" / "info.json").exists()
            or any(p.glob("*.parquet"))
            or any(p.glob("data/*.parquet"))
        )

    def read(self, path: str) -> EpisodeBatch:
        bare = _strip_hf_prefix(path)

        if _is_hub_id(path):
            hf = _require_datasets()
            ds, dataset_name, task = self._load_hub(hf, bare)
            scalar_cols = self._scalar_columns(hf, ds)
            ds = ds.select_columns(scalar_cols)
            df = ds.to_pandas()
            episodes = self._episodes_from_df(df, task, path)
        else:
            p = Path(bare)
            if (p / "meta" / "info.json").exists():
                # v2 format: DuckDB fast path, pyarrow fallback
                try:
                    return self._read_local_v2_duckdb(p, path)
                except ImportError:
                    return self._read_local_v2_pyarrow(p, path)
            else:
                hf = _require_datasets()
                ds, dataset_name, task = self._load_local(hf, p)
                scalar_cols = self._scalar_columns(hf, ds)
                ds = ds.select_columns(scalar_cols)
                df = ds.to_pandas()
                episodes = self._episodes_from_df(df, task, path)
                return EpisodeBatch(
                    episodes=episodes,
                    dataset_name=p.name,
                    format=self.format_name,
                    source_path=path,
                )

        dataset_name = bare.split("/")[-1]
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=dataset_name,
            format=self.format_name,
            source_path=path,
        )

    # ── DuckDB v2 fast path ──────────────────────────────────────────────────

    def _read_local_v2_duckdb(self, p: Path, source: str) -> EpisodeBatch:
        """
        Read a v2 LeRobot dataset (meta/info.json + Parquet shards) using DuckDB.

        Advantages over the HuggingFace datasets path:
          - Reads Parquet natively; no Python object conversion overhead.
          - Image columns are excluded via a SQL projection before any data
            leaves the Parquet pages — they never enter RAM.
          - Global aggregate queries (episode count, action bounds) run in
            under a second on multi-terabyte datasets via DuckDB's push-down.
          - For large datasets (e.g. DROID with 76k episodes), use
            iter_episodes_lazy() to query one episode at a time.
        """
        conn = self._build_duckdb_conn(p)
        task = _read_task_v2(p)

        df = conn.execute("SELECT * FROM dataset").df()
        conn.close()

        episodes = self._episodes_from_df(df, task, source)
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=p.name,
            format=self.format_name,
            source_path=source,
        )

    def _read_local_v2_pyarrow(self, p: Path, source: str) -> EpisodeBatch:
        """
        Fallback reader for v2 LeRobot datasets when DuckDB is not installed.

        Uses pyarrow.parquet with column projection to exclude image columns,
        so image bytes never enter RAM even without DuckDB.
        """
        import pyarrow.parquet as pq

        info_path = p / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        image_cols = _image_columns_from_info(info)
        parquet_files = sorted(p.glob("data/**/*.parquet"))
        if not parquet_files:
            parquet_files = sorted(p.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in {p}")

        # Determine scalar columns from first file's schema
        schema = pq.read_schema(str(parquet_files[0]))
        scalar_cols = [c for c in schema.names if c not in image_cols]

        tables = [pq.read_table(str(f), columns=scalar_cols) for f in parquet_files]

        import pyarrow as pa

        combined = pa.concat_tables(tables)
        df = combined.to_pandas()
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        task = _read_task_v2(p)
        episodes = self._episodes_from_df(df, task, source)
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=p.name,
            format=self.format_name,
            source_path=source,
        )

    def iter_episodes_lazy(self, path: str):
        """
        Yield Episode objects one at a time without loading the full dataset.

        Use this for multi-terabyte datasets where loading everything into
        RAM is not feasible (e.g. DROID, BridgeData V2 full splits).
        Each episode is fetched with a single SQL WHERE clause, so only that
        episode's rows are transferred from Parquet into Python memory.

        Usage::

            reader = LeRobotReader()
            for ep in reader.iter_episodes_lazy("/data/droid"):
                result = TemporalAnalyzer().analyze_episode(ep)
                ...

        Parameters
        ----------
        path : local v2 dataset directory (must have meta/info.json).
               Hub IDs and v1 formats are not supported by this method.
        """
        p = Path(_strip_hf_prefix(path))
        if not (p / "meta" / "info.json").exists():
            raise ValueError(
                f"iter_episodes_lazy requires a v2 local dataset (meta/info.json). "
                f"'{path}' does not appear to be v2 format."
            )
        conn = self._build_duckdb_conn(p)
        task = _read_task_v2(p)

        episode_ids: list[int] = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT episode_index FROM dataset ORDER BY episode_index"
            ).fetchall()
        ]

        for ep_id in episode_ids:
            df = conn.execute(
                f"SELECT * FROM dataset WHERE episode_index = {ep_id} ORDER BY frame_index"
            ).df()
            yield self._episode_from_group(df, ep_id, task, path)

        conn.close()

    def _build_duckdb_conn(self, p: Path):
        """
        Create a DuckDB in-memory connection with a 'dataset' view over all
        Parquet shards in `p`, projecting out image columns.
        """
        duckdb = _require_duckdb()
        conn = duckdb.connect(":memory:")

        info_path = p / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        image_cols = _image_columns_from_info(info)

        parquet_files = sorted(p.glob("data/**/*.parquet"))
        if not parquet_files:
            parquet_files = sorted(p.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in {p}")

        file_list_sql = ", ".join(f"'{f}'" for f in parquet_files)
        conn.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet([{file_list_sql}])")

        all_cols: list[str] = [row[0] for row in conn.execute("DESCRIBE raw").fetchall()]
        scalar_cols = [c for c in all_cols if c not in image_cols]
        col_sql = ", ".join(f'"{c}"' for c in scalar_cols)
        conn.execute(
            f"CREATE VIEW dataset AS SELECT {col_sql} FROM raw ORDER BY episode_index, frame_index"
        )
        return conn

    # ── loading helpers (HF hub / v1 disk) ───────────────────────────────────

    @staticmethod
    def _load_hub(
        hf: "_hf_datasets", path: str
    ) -> tuple["_hf_datasets.Dataset", str, Optional[str]]:
        dataset_name = path.split("/")[-1]
        try:
            ds = hf.load_dataset(path, split="train")
        except Exception:
            dd = hf.load_dataset(path)
            split = next(iter(dd))
            ds = dd[split]
        return ds, dataset_name, None

    @staticmethod
    def _load_local(
        hf: "_hf_datasets", p: Path
    ) -> tuple["_hf_datasets.Dataset", str, Optional[str]]:
        ds = hf.load_from_disk(str(p))
        task = _read_task_v1(p)
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

            return [col for col, feat in ds.features.items() if not isinstance(feat, HFImage)]
        except ImportError:
            return [col for col in ds.column_names if "image" not in col.lower()]

    # ── episode construction ─────────────────────────────────────────────────

    @staticmethod
    def _episodes_from_df(
        df: "_pd.DataFrame",
        task: Optional[str],
        source: str,
    ) -> list[Episode]:
        """Split a full-dataset DataFrame into per-episode Episode objects."""
        episode_col = "episode_index"
        if episode_col not in df.columns:
            raise ValueError(
                f"Expected column '{episode_col}' in LeRobot dataset.\n"
                f"Available columns: {list(df.columns)}"
            )
        episodes: list[Episode] = []
        for ep_id, group in df.groupby(episode_col, sort=True):
            episodes.append(LeRobotReader._episode_from_group(group, ep_id, task, source))
        return episodes

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

        raw_obs: dict[str, np.ndarray] = {}
        for col in group.columns:
            if col.startswith("observation."):
                key = col.removeprefix("observation.")
                try:
                    raw_obs[key] = np.array(group[col].tolist(), dtype=np.float32)
                except (ValueError, TypeError):
                    pass

        obs = normalize_obs_keys(raw_obs)

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


def _image_columns_from_info(info: dict) -> set[str]:
    """
    Return the set of column names that carry image/video data in a v2 info.json.

    LeRobot v2 info.json has a top-level "features" dict.  Each entry whose
    dtype is "image" or "video" — or whose nested "info" dict has an "encoding"
    key — is an image column.  We also fall back to a name-heuristic for
    datasets that deviate from the spec.
    """
    image_cols: set[str] = set()
    for col_name, feat in info.get("features", {}).items():
        if not isinstance(feat, dict):
            continue
        dtype = feat.get("dtype", "")
        if dtype in ("image", "video"):
            image_cols.add(col_name)
            continue
        if "encoding" in feat.get("info", {}):
            image_cols.add(col_name)
            continue
        # name heuristic: fall back for non-standard schemas
        if any(tok in col_name.lower() for tok in ("image", "video", "camera", "rgb", "depth")):
            image_cols.add(col_name)
    return image_cols


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
