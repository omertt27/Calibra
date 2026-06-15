"""
calibra.core.reader — SQL-level zero-copy interface for local v2 LeRobot datasets.

This module exposes LazyDatasetReader: a per-column, per-episode DuckDB query
interface for local datasets stored in the LeRobot v2 format (meta/info.json +
Parquet shards).

It is distinct from LeRobotReader in calibra.ingestion.adapters.lerobot:
  - LeRobotReader loads all scalar columns into an EpisodeBatch for pipeline use.
  - LazyDatasetReader exposes raw SQL projections, useful for notebooks, custom
    scripts, and profiling passes where you only want a few columns from one episode.

Supports:
  - Local v2 datasets (meta/info.json + data/**/*.parquet)
  - hf:// URI prefix stripping (but does NOT download from Hub — for Hub datasets
    use LeRobotReader which calls datasets.load_dataset)

Dependencies: pip install 'calibra[lerobot]'  (pyarrow, duckdb)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def _strip_hf(path: str) -> str:
    return path[len("hf://"):] if path.startswith("hf://") else path


class LazyDatasetReader:
    """
    Zero-copy robotic trajectory reader powered by DuckDB and PyArrow.

    Designed for profiling passes over large datasets where you need only a
    handful of columns from a single episode. Image/video columns are excluded
    automatically before any data leaves Parquet pages.

    Parameters
    ----------
    dataset_path : path to a local v2 LeRobot dataset directory.
                   hf:// URI prefixes are stripped automatically.

    Example
    -------
    >>> reader = LazyDatasetReader("/data/lerobot/aloha_mobile")
    >>> table = reader.query_proprioception_tensors(
    ...     ["observation.state", "action"], episode_idx=0
    ... )
    >>> arr = table["action"].to_pylist()
    """

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = _strip_hf(dataset_path)
        p = Path(self.dataset_path)

        info_path = p / "meta" / "info.json"
        if not info_path.exists():
            raise ValueError(
                f"LazyDatasetReader requires a v2 LeRobot dataset (meta/info.json). "
                f"'{dataset_path}' does not appear to be v2 format.\n"
                "For Hub datasets or v1 disk datasets, use "
                "calibra.ingestion.adapters.lerobot.LeRobotReader instead."
            )

        with open(info_path) as f:
            self.info: dict = json.load(f)

        self.fps: float = float(self.info.get("fps", 30))
        self.features: dict = self.info.get("features", {})
        self._p = p
        self._conn = None  # lazy — created on first query

    # ── public API ────────────────────────────────────────────────────────────

    def query_proprioception_tensors(
        self,
        columns: List[str],
        episode_idx: int,
    ):
        """
        Query specific proprioception columns for one episode without loading
        image or video data.

        Parameters
        ----------
        columns     : list of feature names (e.g. ["observation.state", "action"]).
                      Each name is resolved to its actual flat Parquet column(s).
        episode_idx : episode_index value to filter on.

        Returns
        -------
        pyarrow.Table — ordered by frame_index ascending.
        """
        conn = self._get_conn()
        resolved = []
        for col in columns:
            resolved.extend(self._resolve_feature_channels(col))

        col_projection = ", ".join(f'"{c}"' for c in resolved)
        query = (
            f"SELECT {col_projection} "
            f"FROM dataset "
            f"WHERE episode_index = {episode_idx} "
            f"ORDER BY frame_index ASC"
        )
        return conn.execute(query).arrow()

    def list_episodes(self) -> List[int]:
        """Return sorted list of all episode_index values in the dataset."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT episode_index FROM dataset ORDER BY episode_index"
        ).fetchall()
        return [r[0] for r in rows]

    def episode_count(self) -> int:
        conn = self._get_conn()
        return conn.execute(
            "SELECT COUNT(DISTINCT episode_index) FROM dataset"
        ).fetchone()[0]

    def get_fps(self) -> float:
        return self.fps

    def get_features(self) -> dict:
        return self.features

    def list_scalar_columns(self) -> List[str]:
        """Return all scalar (non-image/video) column names available."""
        conn = self._get_conn()
        return [row[0] for row in conn.execute("DESCRIBE dataset").fetchall()]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_conn(self):
        if self._conn is None:
            self._conn = self._build_conn()
        return self._conn

    def _build_conn(self):
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "duckdb is required for LazyDatasetReader.\n"
                "Install it with: pip install 'calibra[lerobot]'"
            ) from None

        conn = duckdb.connect(":memory:")
        image_cols = self._image_columns()

        parquet_files = sorted(self._p.glob("data/**/*.parquet"))
        if not parquet_files:
            parquet_files = sorted(self._p.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in {self._p}")

        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        conn.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet([{file_list}])")

        all_cols = [row[0] for row in conn.execute("DESCRIBE raw").fetchall()]
        scalar_cols = [c for c in all_cols if c not in image_cols]
        col_sql = ", ".join(f'"{c}"' for c in scalar_cols)
        conn.execute(
            f"CREATE VIEW dataset AS SELECT {col_sql} FROM raw "
            f"ORDER BY episode_index, frame_index"
        )
        return conn

    def _image_columns(self) -> set:
        image_cols: set = set()
        for col_name, feat in self.features.items():
            if not isinstance(feat, dict):
                continue
            if feat.get("dtype", "") in ("image", "video"):
                image_cols.add(col_name)
                continue
            if "encoding" in feat.get("info", {}):
                image_cols.add(col_name)
                continue
            if any(tok in col_name.lower() for tok in ("image", "video", "camera", "rgb", "depth")):
                image_cols.add(col_name)
        return image_cols

    def _resolve_feature_channels(self, feature_name: str) -> List[str]:
        """
        Map a high-level feature name to actual flat Parquet column names.

        LeRobot v2 stores multi-dimensional features as flat arrays in Parquet.
        For a feature with shape [14], the column name is simply the feature name
        and the value in each row is a list of 14 floats. For flattened formats
        where each dimension is a separate column (feature_0, feature_1, …),
        this method returns the expanded names.
        """
        # First check if the exact column exists
        try:
            conn = self._get_conn()
            existing = {row[0] for row in conn.execute("DESCRIBE dataset").fetchall()}
            if feature_name in existing:
                return [feature_name]
        except Exception:
            pass

        # Check features metadata for shape info
        feat_meta = self.features.get(feature_name, {})
        shape = feat_meta.get("shape", [1])

        # If it's a 1D feature with >1 element and individual columns exist, expand
        if len(shape) == 1 and shape[0] > 1:
            expanded = [f"{feature_name}_{i}" for i in range(shape[0])]
            try:
                existing_set = {row[0] for row in conn.execute("DESCRIBE dataset").fetchall()}
                if all(c in existing_set for c in expanded):
                    return expanded
            except Exception:
                pass

        # Fall back to the feature name as-is
        return [feature_name]