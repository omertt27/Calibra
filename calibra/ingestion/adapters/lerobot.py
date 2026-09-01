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
  - Local disk (v3):  directory with meta/info.json (codebase_version "v3.0")
                      + Parquet shards; same fast path as v2
                      (fast path: DuckDB reads Parquet directly without
                       loading image columns into RAM)

Image feature columns are skipped by default for performance. On the v1 path
(HuggingFace Hub / local `datasets`-saved directories) they can be decoded
instead via `LeRobotReader(decode_images=True)` — opt-in, since it changes
load time and memory characteristics. v2/v3 (video-encoded) datasets don't
support this yet; see `_read_local_v2_duckdb`/`_read_local_v2_pyarrow`.

Dependencies:
  pip install 'calibra[lerobot]'  (datasets, pyarrow, duckdb, pillow)
"""

from __future__ import annotations

import io
import json
import sys
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


def _require_pillow():
    try:
        from PIL import Image

        return Image
    except ImportError:
        raise ImportError(
            "Pillow is required to decode LeRobot image columns.\n"
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
    """
    Reads LeRobot-format HuggingFace datasets from Hub or local disk.

    Parameters
    ----------
    decode_images : if True, decode HuggingFace `Image`-feature columns into
                     (T, H, W, C) uint8 arrays instead of excluding them.
                     Only applies to the v1 path (Hub / local `datasets`-saved
                     directories) — opt-in because it increases load time and
                     memory use. v2/v3 (video-encoded) datasets are unaffected.
    """

    def __init__(self, decode_images: bool = False) -> None:
        self.decode_images = decode_images

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
            task_table = _fetch_hub_tasks_table(bare)
            scalar_cols = self._scalar_columns(hf, ds)
            ds = ds.select_columns(scalar_cols)
            df = ds.to_pandas()
            episodes = self._episodes_from_df(df, task, path, task_table)
        else:
            p = Path(bare)
            if (p / "meta" / "info.json").exists():
                # v2/v3 format (meta/info.json + Parquet shards): DuckDB fast path, pyarrow fallback
                if self.decode_images:
                    print(
                        "warning: --decode-images is not yet supported for v2/v3 "
                        "(video-encoded) LeRobot datasets; images will still be excluded.",
                        file=sys.stderr,
                    )
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
        task_table = _read_tasks_table_v2v3(p)

        df = conn.execute("SELECT * FROM dataset").df()
        conn.close()

        episodes = self._episodes_from_df(df, task, source, task_table)
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
        task_table = _read_tasks_table_v2v3(p)
        episodes = self._episodes_from_df(df, task, source, task_table)
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
        task_table = _read_tasks_table_v2v3(p)

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
            yield self._episode_from_group(df, ep_id, task, path, task_table)

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

    def _scalar_columns(self, hf: "_hf_datasets", ds: "_hf_datasets.Dataset") -> list[str]:
        """
        Return columns to load. HuggingFace Image features are excluded unless
        `self.decode_images` is set, in which case they're kept for decoding
        in `_episode_from_group`.
        """
        try:
            from datasets import Image as HFImage

            if self.decode_images:
                return list(ds.features.keys())
            return [col for col, feat in ds.features.items() if not isinstance(feat, HFImage)]
        except ImportError:
            if self.decode_images:
                return list(ds.column_names)
            return [col for col in ds.column_names if "image" not in col.lower()]

    # ── episode construction ─────────────────────────────────────────────────

    @staticmethod
    def _episodes_from_df(
        df: "_pd.DataFrame",
        task: Optional[str],
        source: str,
        task_table: Optional[dict[int, str]] = None,
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
            episodes.append(
                LeRobotReader._episode_from_group(group, ep_id, task, source, task_table)
            )
        return episodes

    @staticmethod
    def _episode_from_group(
        group: "_pd.DataFrame",
        ep_id: int,
        task: Optional[str],
        source: str,
        task_table: Optional[dict[int, str]] = None,
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
                    decoded = _decode_image_column(group[col].tolist())
                    if decoded is not None:
                        raw_obs[key] = decoded

        obs = normalize_obs_keys(raw_obs)
        resolved_task = _resolve_episode_task(group, task, task_table)

        return Episode(
            metadata=EpisodeMetadata(
                episode_id=str(ep_id),
                task_description=resolved_task,
                source_file=source,
            ),
            timestamps=timestamps,
            observations=obs,
            actions=actions,
        )


# ── image decoding (v1 only) ─────────────────────────────────────────────────


def _decode_image_column(cells: list) -> Optional[np.ndarray]:
    """
    Decode a HuggingFace Image-feature column into a (T, H, W, C) uint8 array.

    `Dataset.to_pandas()` yields `{"bytes": <encoded>, "path": <str|None>}`
    dicts per cell (verified empirically — it does NOT eagerly decode to
    pixels the way `Dataset.__getitem__` does). Handles that shape, plus an
    already-decoded PIL.Image cell defensively. Returns None (rather than
    raising) if the cells aren't image data, so callers can fall back to
    dropping the column exactly as they did before this column existed.
    """
    try:
        PILImage = _require_pillow()
    except ImportError:
        return None

    frames = []
    for cell in cells:
        try:
            if isinstance(cell, dict):
                data = cell.get("bytes")
                if data is None and cell.get("path"):
                    with open(cell["path"], "rb") as f:
                        data = f.read()
                if data is None:
                    return None
                img = PILImage.open(io.BytesIO(data)).convert("RGB")
            elif isinstance(cell, PILImage.Image):
                img = cell.convert("RGB")
            else:
                return None
        except Exception:
            return None
        frames.append(np.array(img, dtype=np.uint8))

    return np.stack(frames)


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


def _resolve_episode_task(
    group: "_pd.DataFrame",
    fallback_task: Optional[str],
    task_table: Optional[dict[int, str]],
) -> Optional[str]:
    """
    Per-episode task text, preferring a task_index lookup over a single
    dataset-wide fallback string.

    Multi-task LeRobot v2/v3 datasets carry a `task_index` column per frame
    plus a separate task table (`meta/tasks.parquet` in v3, `meta/tasks.jsonl`
    in v2) mapping index -> text. `fallback_task` (a single dataset-wide
    string, or None) is used when no table entry is available, matching the
    single-task behavior for datasets without a task table.
    """
    if task_table and "task_index" in group.columns:
        raw = group["task_index"].iloc[0]
        if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
            resolved = task_table.get(int(raw))
            if resolved is not None:
                return resolved
    return fallback_task


def _parse_tasks_parquet(path: Path) -> Optional[dict[int, str]]:
    """
    Parse a LeRobot v3 `meta/tasks.parquet` task table into {task_index: text}.

    LeRobotDatasetMetadata builds this as a DataFrame indexed by task text
    with a `task_index` column (`pd.DataFrame({"task_index": ...}, index=tasks)`).
    Also accept a plain `task_index`/`task` two-column layout defensively, in
    case of schema drift.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if "task_index" not in df.columns:
        return None
    if "task" in df.columns:
        return {int(ti): str(t) for ti, t in zip(df["task_index"], df["task"])}
    return {int(ti): str(t) for t, ti in zip(df.index, df["task_index"])}


def _parse_tasks_jsonl(path: Path) -> Optional[dict[int, str]]:
    """Parse a LeRobot v2 `meta/tasks.jsonl` task table into {task_index: text}."""
    table: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "task_index" in obj and "task" in obj:
                table[int(obj["task_index"])] = obj["task"]
    return table or None


def _read_tasks_table_v2v3(p: Path) -> Optional[dict[int, str]]:
    """Build the {task_index: text} table from a local v2/v3 dataset's meta/ dir."""
    parquet_path = p / "meta" / "tasks.parquet"
    if parquet_path.exists():
        table = _parse_tasks_parquet(parquet_path)
        if table:
            return table
    jsonl_path = p / "meta" / "tasks.jsonl"
    if jsonl_path.exists():
        return _parse_tasks_jsonl(jsonl_path)
    return None


def _fetch_hub_tasks_table(repo_id: str) -> Optional[dict[int, str]]:
    """
    Best-effort fetch of a Hub dataset's task table (v3 tasks.parquet, v2
    tasks.jsonl), so `LeRobotReader.read()` on a Hub ID (e.g. "lerobot/libero_10")
    gets the same per-episode task_description as the local v2/v3 path instead
    of always None. Returns None (never raises) if huggingface_hub is missing
    or the repo has no task table (e.g. single-task v1 datasets).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    for filename, parser in (
        ("meta/tasks.parquet", _parse_tasks_parquet),
        ("meta/tasks.jsonl", _parse_tasks_jsonl),
    ):
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        except Exception:
            continue
        table = parser(Path(local_path))
        if table:
            return table
    return None
