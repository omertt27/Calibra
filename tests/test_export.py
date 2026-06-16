"""Tests for calibra.curation.export — dataset materialisation after pruning."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from calibra.pruning import PruningResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_result(keep_ids: list[str], n_original: int = 10) -> PruningResult:
    all_ids = [str(i) for i in range(n_original)]
    quality_fail = []
    diversity_pruned = [eid for eid in all_ids if eid not in keep_ids]
    return PruningResult(
        keep_episode_ids=keep_ids,
        quality_fail_ids=quality_fail,
        diversity_pruned_ids=diversity_pruned,
        quality_scores={eid: float(i) * 0.1 for i, eid in enumerate(all_ids)},
        diversity_scores={eid: float(i) * 0.2 for i, eid in enumerate(all_ids)},
        n_original=n_original,
        n_kept=len(keep_ids),
        n_quality_failures=0,
        n_diversity_pruned=len(diversity_pruned),
        keep_fraction_actual=len(keep_ids) / n_original,
    )


def _make_lerobot_v2_dir(tmp_path: Path, n_episodes: int = 5, steps_per_ep: int = 10) -> Path:
    """
    Create a minimal LeRobot v2 dataset on disk (Parquet + meta).
    Uses pyarrow directly; skips if unavailable.
    """
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    import pyarrow as pa_mod
    import pyarrow.parquet as pq_mod

    ds_dir = tmp_path / "my_dataset"
    data_dir = ds_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    meta_dir = ds_dir / "meta"
    meta_dir.mkdir()

    rows: dict[str, list] = {
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "action": [],
        "observation.state": [],
    }
    for ep in range(n_episodes):
        for step in range(steps_per_ep):
            rows["episode_index"].append(ep)
            rows["frame_index"].append(step)
            rows["timestamp"].append(step * 0.02)
            rows["action"].append([float(ep), float(step)])
            rows["observation.state"].append([float(ep + 0.1), float(step + 0.1)])

    table = pa_mod.table({
        "episode_index": pa_mod.array(rows["episode_index"], type=pa_mod.int64()),
        "frame_index":   pa_mod.array(rows["frame_index"],   type=pa_mod.int64()),
        "timestamp":     pa_mod.array(rows["timestamp"],     type=pa_mod.float64()),
        "action":        pa_mod.array(rows["action"],        type=pa_mod.list_(pa_mod.float32())),
        "observation.state": pa_mod.array(
            rows["observation.state"], type=pa_mod.list_(pa_mod.float32())
        ),
    })
    pq_mod.write_table(table, data_dir / "train-00000-of-00001.parquet")

    info = {
        "total_episodes": n_episodes,
        "total_frames": n_episodes * steps_per_ep,
        "fps": 50,
        "features": {
            "action": {"dtype": "float32", "shape": [2]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "episode_index": {"dtype": "int64"},
            "frame_index": {"dtype": "int64"},
            "timestamp": {"dtype": "float64"},
        },
        "splits": {"train": f"0:{n_episodes * steps_per_ep}"},
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    episodes_lines = [
        json.dumps({"episode_index": ep, "length": steps_per_ep})
        for ep in range(n_episodes)
    ]
    (meta_dir / "episodes.jsonl").write_text("\n".join(episodes_lines) + "\n")
    (meta_dir / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick"}) + "\n")

    return ds_dir


# ── LeRobot v2 export ─────────────────────────────────────────────────────────

class TestExportLeRobotV2:
    def test_basic_export_creates_output(self, tmp_path):
        pytest.importorskip("pyarrow")
        from calibra.curation.export import export_dataset

        src = _make_lerobot_v2_dir(tmp_path, n_episodes=5)
        result = _make_result(keep_ids=["0", "2", "4"], n_original=5)

        out = tmp_path / "coreset"
        exported = export_dataset(result, str(src), out)

        assert exported.exists()
        assert (exported / "meta" / "info.json").exists()
        assert list((exported / "data").rglob("*.parquet"))

    def test_row_count_matches_kept_episodes(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        from calibra.curation.export import export_dataset

        src = _make_lerobot_v2_dir(tmp_path, n_episodes=5, steps_per_ep=10)
        keep_ids = ["1", "3"]
        result = _make_result(keep_ids=keep_ids, n_original=5)

        out = tmp_path / "coreset"
        exported = export_dataset(result, str(src), out)

        parquet_files = list((exported / "data").rglob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        assert len(table) == len(keep_ids) * 10  # 2 episodes × 10 steps

    def test_episode_index_remapped_to_zero_based(self, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        from calibra.curation.export import export_dataset

        src = _make_lerobot_v2_dir(tmp_path, n_episodes=5)
        result = _make_result(keep_ids=["2", "4"], n_original=5)

        out = tmp_path / "coreset"
        exported = export_dataset(result, str(src), out)

        parquet_files = list((exported / "data").rglob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        ep_indices = sorted(set(table.column("episode_index").to_pylist()))
        assert ep_indices == [0, 1]  # remapped, not original [2, 4]

    def test_info_json_updated(self, tmp_path):
        pytest.importorskip("pyarrow")
        from calibra.curation.export import export_dataset

        src = _make_lerobot_v2_dir(tmp_path, n_episodes=6, steps_per_ep=8)
        result = _make_result(keep_ids=["0", "3", "5"], n_original=6)

        out = tmp_path / "coreset"
        export_dataset(result, str(src), out)

        info = json.loads((out / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 3
        assert info["total_frames"] == 3 * 8

    def test_episodes_jsonl_filtered_and_reindexed(self, tmp_path):
        pytest.importorskip("pyarrow")
        from calibra.curation.export import export_dataset

        src = _make_lerobot_v2_dir(tmp_path, n_episodes=5)
        result = _make_result(keep_ids=["1", "4"], n_original=5)

        out = tmp_path / "coreset"
        export_dataset(result, str(src), out)

        lines = (out / "meta" / "episodes.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        indices = [json.loads(l)["episode_index"] for l in lines]
        assert sorted(indices) == [0, 1]

    def test_tasks_jsonl_copied(self, tmp_path):
        pytest.importorskip("pyarrow")
        from calibra.curation.export import export_dataset

        src = _make_lerobot_v2_dir(tmp_path, n_episodes=3)
        result = _make_result(keep_ids=["0", "1"], n_original=3)

        out = tmp_path / "coreset"
        export_dataset(result, str(src), out)

        assert (out / "meta" / "tasks.jsonl").exists()


# ── Hub ID guard ──────────────────────────────────────────────────────────────

class TestHubIdGuard:
    def test_hub_id_raises_valueerror(self, tmp_path):
        from calibra.curation.export import export_dataset

        result = _make_result(keep_ids=["0"], n_original=5)
        with pytest.raises(ValueError, match="Hub IDs are not supported"):
            export_dataset(result, "lerobot/pusht", tmp_path / "out")

    def test_hf_uri_raises_valueerror(self, tmp_path):
        from calibra.curation.export import export_dataset

        result = _make_result(keep_ids=["0"], n_original=5)
        with pytest.raises(ValueError, match="Hub IDs are not supported"):
            export_dataset(result, "hf://lerobot/pusht", tmp_path / "out")


# ── unknown format ────────────────────────────────────────────────────────────

class TestUnknownFormat:
    def test_unknown_path_raises_valueerror(self, tmp_path):
        from calibra.curation.export import export_dataset

        result = _make_result(keep_ids=["0"], n_original=5)
        unknown = tmp_path / "some_random_dir"
        unknown.mkdir()
        with pytest.raises(ValueError, match="Cannot determine format"):
            export_dataset(result, str(unknown), tmp_path / "out")
