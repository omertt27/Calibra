"""
Regression test for per-episode task identity on local LeRobot v2/v3 datasets.

Real multi-task v3 datasets (e.g. lerobot/libero_10) store task text in a
separate `meta/tasks.parquet` table, keyed by the per-frame `task_index`
column — there is no per-episode task string sitting in info.json. Before the
fix in calibra/ingestion/adapters/lerobot.py, LeRobotReader only ever read a
single dataset-wide task string (or None), so every episode of a multi-task
dataset silently got the same (or no) task_description.

This builds a minimal two-episode / two-task v3-shaped fixture entirely on
disk (no network) and asserts the two episodes come back with distinct task
descriptions that match their task_index.
"""

from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas", reason="pip install 'calibra-robotics[lerobot]'")
pytest.importorskip("pyarrow", reason="pip install 'calibra-robotics[lerobot]'")

from calibra.ingestion.adapters.lerobot import (  # noqa: E402
    LeRobotReader,
    _parse_tasks_parquet,
    _read_tasks_table_v2v3,
)

TASKS = ["pick up the red block", "open the drawer"]


def _write_v3_fixture(root, tasks_layout: str = "indexed") -> None:
    """
    Write a tiny two-episode, two-task v3-shaped dataset under `root`:
      meta/info.json
      meta/tasks.parquet   (task_index -> text)
      data/episodes.parquet  (episode_index, frame_index, timestamp, action, task_index)

    tasks_layout="indexed" matches the real LeRobotDatasetMetadata.tasks shape
    (DataFrame indexed by task text, with a task_index column).
    tasks_layout="columns" is the defensive plain two-column fallback.
    """
    meta_dir = root / "meta"
    data_dir = root / "data"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    (meta_dir / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "features": {}}))

    if tasks_layout == "indexed":
        tasks_df = pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(TASKS, name="task"))
    else:
        tasks_df = pd.DataFrame({"task_index": [0, 1], "task": TASKS})
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    n_steps = 5
    rows = []
    for ep_id, task_idx in enumerate([0, 1]):
        for step in range(n_steps):
            rows.append(
                {
                    "episode_index": ep_id,
                    "frame_index": step,
                    "timestamp": step * 0.1,
                    "action": [float(step), float(step) * 2.0],
                    "task_index": task_idx,
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(data_dir / "episodes.parquet")


class TestLocalV3TaskTable:
    def test_task_table_parsed_indexed_layout(self, tmp_path):
        _write_v3_fixture(tmp_path, tasks_layout="indexed")
        table = _read_tasks_table_v2v3(tmp_path)
        assert table == {0: TASKS[0], 1: TASKS[1]}

    def test_task_table_parsed_columns_layout(self, tmp_path):
        _write_v3_fixture(tmp_path, tasks_layout="columns")
        table = _parse_tasks_parquet(tmp_path / "meta" / "tasks.parquet")
        assert table == {0: TASKS[0], 1: TASKS[1]}

    def test_episodes_retain_distinct_task_descriptions(self, tmp_path):
        _write_v3_fixture(tmp_path, tasks_layout="indexed")
        batch = LeRobotReader().read(str(tmp_path))

        assert batch.n_episodes == 2
        by_id = {ep.metadata.episode_id: ep.metadata.task_description for ep in batch.episodes}
        descriptions = set(by_id.values())

        assert None not in descriptions, f"task_description missing for some episode: {by_id}"
        assert descriptions == set(TASKS), f"expected distinct tasks, got: {by_id}"

    def test_no_task_table_falls_back_to_none(self, tmp_path):
        """Datasets without a task table (or plain v2 single-task) keep prior behavior."""
        meta_dir = tmp_path / "meta"
        data_dir = tmp_path / "data"
        meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        (meta_dir / "info.json").write_text(json.dumps({"features": {}}))
        df = pd.DataFrame(
            {
                "episode_index": [0, 0],
                "frame_index": [0, 1],
                "timestamp": [0.0, 0.1],
                "action": [[0.0], [1.0]],
            }
        )
        df.to_parquet(data_dir / "episodes.parquet")

        batch = LeRobotReader().read(str(tmp_path))
        assert batch.episodes[0].metadata.task_description is None
