"""
calibra.curation.export — Materialise a pruned coreset as a ready-to-train dataset.

After ``calibra prune`` selects a coreset index (``coreset_index.json``), this
module writes the *actual* dataset files so training scripts can consume the
pruned data directly without any glue code.

Supported source formats
------------------------
* **LeRobot v2** (local Parquet shards + ``meta/info.json``) — full re-index,
  metadata update, and optional stats passthrough.
* **HDF5** (Isaac Lab / Robomimic) — copies kept episode groups into a new file.
* **LeRobot v1** (HuggingFace Datasets on disk) — filters and saves with
  ``save_to_disk``.

Hub IDs (``lerobot/pusht``, ``hf://…``) are *not* directly supported — download
the dataset locally first (``huggingface-cli download``), then run prune + export.

All formats re-number episode IDs from 0 to N-1 so the output is a self-contained,
valid dataset that training scripts (LeRobot ``train.py``, etc.) can consume
without modification.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

from calibra.pruning import PruningResult


def export_dataset(
    result: PruningResult,
    source_path: str,
    out_dir: str | Path,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Materialise the coreset identified by *result* as a dataset directory.

    Parameters
    ----------
    result      : output of ``CoresetSelector.select()``
    source_path : path of the original **local** dataset (Hub IDs not supported)
    out_dir     : directory to write the exported dataset into
    log         : optional callable(str) for progress messages

    Returns
    -------
    Path to the written dataset directory.

    Raises
    ------
    ValueError  : if source_path is a Hub ID or the format cannot be determined.
    RuntimeError: if required dependencies (pyarrow, h5py, datasets) are missing.
    """
    if log is None:
        log = lambda _: None  # noqa: E731

    from calibra.ingestion.adapters.lerobot import _is_hub_id, _strip_hf_prefix

    if _is_hub_id(source_path):
        raise ValueError(
            f"Hub IDs are not supported by --export-dataset. "
            f"Download '{source_path}' locally first:\n"
            f"  huggingface-cli download {source_path} --local-dir ./datasets/{source_path.split('/')[-1]}\n"
            f"Then re-run: calibra prune ./datasets/{source_path.split('/')[-1]} "
            f"--keep ... --export-dataset <out>"
        )

    bare = _strip_hf_prefix(source_path)
    p = Path(bare)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if (p / "meta" / "info.json").exists():
        return _export_lerobot_v2(p, out, result, log)

    if (p / "metadata.json").exists() or (p / "dataset_dict.json").exists():
        return _export_lerobot_v1(p, out, result, log)

    if p.suffix in (".h5", ".hdf5"):
        return _export_hdf5(p, out, result, log)

    h5_files = list(p.glob("**/*.h5")) + list(p.glob("**/*.hdf5"))
    if h5_files:
        return _export_hdf5(p, out, result, log)

    raise ValueError(
        f"Cannot determine format for '{source_path}'.\n"
        "Supported: LeRobot v2 (local Parquet), LeRobot v1 (HF Datasets on disk), HDF5."
    )


# ── LeRobot v2 ────────────────────────────────────────────────────────────────

def _export_lerobot_v2(
    src: Path,
    out: Path,
    result: PruningResult,
    log: Callable[[str], None],
) -> Path:
    """
    Export a LeRobot v2 dataset (Parquet shards) filtered to the coreset.

    Steps
    -----
    1. Read all Parquet shards with pyarrow.
    2. Filter rows to ``keep_episode_ids``.
    3. Remap ``episode_index`` to 0..N-1 (preserving original sort order).
    4. Write a single consolidated Parquet shard into ``out/data/chunk-000/``.
    5. Copy and update ``meta/`` files (info.json, episodes.jsonl, tasks.jsonl,
       stats.json).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError(
            "pyarrow is required for LeRobot v2 export.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None

    keep_int: set[int] = {int(eid) for eid in result.keep_episode_ids}
    # Sorted list of original episode IDs to keep, in ascending order.
    sorted_keep = sorted(keep_int)
    # Mapping: original episode_index → new 0-based index.
    ep_remap: dict[int, int] = {old: new for new, old in enumerate(sorted_keep)}

    # ── read and filter Parquet ───────────────────────────────────────────────
    parquet_files = sorted(src.glob("data/**/*.parquet"))
    if not parquet_files:
        parquet_files = sorted(src.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {src}")

    log(f"  Reading {len(parquet_files)} Parquet shard(s) …")

    tables: list[pa.Table] = []
    for pf in parquet_files:
        tbl = pq.read_table(pf)
        # Filter to kept episodes.
        ep_col = tbl.column("episode_index")
        mask = pa.array(
            [ep_col[i].as_py() in keep_int for i in range(len(ep_col))],
            type=pa.bool_(),
        )
        filtered = tbl.filter(mask)
        if len(filtered) > 0:
            tables.append(filtered)

    if not tables:
        raise ValueError("No rows remain after filtering — coreset is empty.")

    combined = pa.concat_tables(tables)
    log(f"  {len(combined)} rows kept from {sum(len(t) for t in tables)} total filtered rows")

    # ── remap episode_index ───────────────────────────────────────────────────
    orig_ep_col = combined.column("episode_index").to_pylist()
    new_ep_col = pa.array([ep_remap[v] for v in orig_ep_col], type=pa.int64())
    col_idx = combined.schema.get_field_index("episode_index")
    combined = combined.set_column(col_idx, "episode_index", new_ep_col)

    # ── write Parquet ─────────────────────────────────────────────────────────
    data_out = out / "data" / "chunk-000"
    data_out.mkdir(parents=True, exist_ok=True)
    out_parquet = data_out / "train-00000-of-00001.parquet"
    pq.write_table(combined, out_parquet)
    log(f"  Wrote {out_parquet}")

    # ── update meta/ ─────────────────────────────────────────────────────────
    meta_src = src / "meta"
    meta_out = out / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)

    _update_info_json(meta_src, meta_out, result, len(combined))
    _filter_episodes_jsonl(meta_src, meta_out, ep_remap)
    _copy_tasks_jsonl(meta_src, meta_out)
    _copy_stats_json(meta_src, meta_out)

    log(f"  Meta written to {meta_out}")
    return out


def _update_info_json(
    meta_src: Path,
    meta_out: Path,
    result: PruningResult,
    total_frames: int,
) -> None:
    info_path = meta_src / "info.json"
    if not info_path.exists():
        return
    with open(info_path) as f:
        info = json.load(f)

    info["total_episodes"] = result.n_kept
    info["total_frames"] = total_frames

    # Update splits if present (LeRobot v2 splits block).
    if "splits" in info:
        info["splits"] = {"train": f"0:{total_frames}"}

    with open(meta_out / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def _filter_episodes_jsonl(
    meta_src: Path,
    meta_out: Path,
    ep_remap: dict[int, int],
) -> None:
    src_path = meta_src / "episodes.jsonl"
    if not src_path.exists():
        return
    kept_lines: list[str] = []
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            orig_idx = obj.get("episode_index")
            if orig_idx in ep_remap:
                obj["episode_index"] = ep_remap[orig_idx]
                kept_lines.append(json.dumps(obj))
    # Sort by new episode_index for cleanliness.
    kept_lines.sort(key=lambda s: json.loads(s)["episode_index"])
    with open(meta_out / "episodes.jsonl", "w") as f:
        f.write("\n".join(kept_lines))
        if kept_lines:
            f.write("\n")


def _copy_tasks_jsonl(meta_src: Path, meta_out: Path) -> None:
    src = meta_src / "tasks.jsonl"
    if src.exists():
        shutil.copy2(src, meta_out / "tasks.jsonl")


def _copy_stats_json(meta_src: Path, meta_out: Path) -> None:
    """Copy stats.json as-is (approximate but training scripts expect it)."""
    src = meta_src / "stats.json"
    if src.exists():
        shutil.copy2(src, meta_out / "stats.json")


# ── LeRobot v1 (HuggingFace Datasets) ────────────────────────────────────────

def _export_lerobot_v1(
    src: Path,
    out: Path,
    result: PruningResult,
    log: Callable[[str], None],
) -> Path:
    try:
        import datasets as hf_datasets
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required for LeRobot v1 export.\n"
            "Install it with: pip install 'calibra[lerobot]'"
        ) from None

    keep_int: set[int] = {int(eid) for eid in result.keep_episode_ids}
    log(f"  Loading LeRobot v1 dataset from {src} …")
    ds = hf_datasets.load_from_disk(str(src))
    if hasattr(ds, "keys"):
        split_name = next(iter(ds))
        ds = ds[split_name]

    log(f"  Filtering to {len(keep_int)} episodes …")
    ds = ds.filter(lambda row: row["episode_index"] in keep_int)

    # Re-index episode_index 0..N-1.
    sorted_keep = sorted(keep_int)
    ep_remap = {old: new for new, old in enumerate(sorted_keep)}
    ds = ds.map(lambda row: {"episode_index": ep_remap[row["episode_index"]]})

    log(f"  Saving to {out} …")
    ds.save_to_disk(str(out))
    return out


# ── HDF5 ──────────────────────────────────────────────────────────────────────

def _export_hdf5(
    src: Path,
    out: Path,
    result: PruningResult,
    log: Callable[[str], None],
) -> Path:
    try:
        import h5py
    except ImportError:
        raise RuntimeError(
            "h5py is required for HDF5 export.\n"
            "Install it with: pip install 'calibra[hdf5]'"
        ) from None

    keep_ids: set[str] = set(result.keep_episode_ids)

    # Find the source HDF5 file.
    if src.suffix in (".h5", ".hdf5"):
        src_file = src
        out_file = out / src.name
    else:
        h5_files = sorted(src.glob("**/*.h5")) + sorted(src.glob("**/*.hdf5"))
        if not h5_files:
            raise ValueError(f"No HDF5 files found under {src}")
        src_file = h5_files[0]
        out_file = out / src_file.name

    log(f"  Copying kept episode groups from {src_file} …")
    with h5py.File(src_file, "r") as src_h5, h5py.File(out_file, "w") as dst_h5:
        # Copy top-level non-episode groups/datasets first (e.g. metadata).
        for key in src_h5.keys():
            if key not in keep_ids and not _is_episode_group(src_h5[key], keep_ids):
                src_h5.copy(key, dst_h5)

        # Copy kept episode groups, re-numbering from 0.
        sorted_keep = sorted(keep_ids, key=lambda x: int(x) if x.isdigit() else x)
        for new_idx, old_key in enumerate(sorted_keep):
            if old_key in src_h5:
                new_key = str(new_idx) if old_key.isdigit() else f"demo_{new_idx}"
                src_h5.copy(old_key, dst_h5, name=new_key)
            else:
                log(f"  Warning: episode '{old_key}' not found in source HDF5 — skipping.")

    log(f"  Wrote {out_file}")
    return out


def _is_episode_group(obj, episode_ids: set[str]) -> bool:
    """Heuristic: True if obj is an HDF5 group whose name matches an episode ID."""
    try:
        import h5py
        return isinstance(obj, h5py.Group) and obj.name.lstrip("/") in episode_ids
    except Exception:
        return False
