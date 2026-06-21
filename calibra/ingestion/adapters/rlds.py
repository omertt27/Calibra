"""
RLDS adapter (Reinforcement Learning Datasets).

RLDS datasets are TensorFlow Datasets (TFDS) following the RLDS spec:

  episode → steps → {observation, action, reward, is_terminal, …}

The adapter iterates episodes via tfds and converts to EpisodeBatch.
Timestamps are synthesized from step index × (1/fps) unless the step dict
contains an explicit "timestamp" key.

Dependency: pip install 'calibra[rlds]'  (tensorflow, tensorflow-datasets)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

if TYPE_CHECKING:
    import tensorflow as tf
    import tensorflow_datasets as tfds


def _require_tfds() -> tuple["tf", "tfds"]:
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
        return tf, tfds
    except ImportError:
        raise ImportError(
            "tensorflow and tensorflow-datasets are required for the RLDS adapter.\n"
            "Install them with: pip install 'calibra[rlds]'"
        ) from None


@register
class RLDSReader(DatasetReader):
    """
    Reads RLDS datasets from a local data_dir or a registered TFDS name.

    `path` can be:
      - A TFDS dataset name (e.g. "bridge" or "fractal20220817_data")
      - A local directory containing tfrecord shards
    """

    def __init__(self, fps: float = 10.0) -> None:
        self._fps = fps

    @property
    def format_name(self) -> str:
        return "rlds"

    @classmethod
    def can_read(cls, path: str) -> bool:
        p = Path(path)
        if p.is_dir() and any(p.glob("*.tfrecord*")):
            return True
        # Heuristic: no extension, not a file → might be a TFDS name.
        return not p.exists() and "." not in path

    def read(self, path: str) -> EpisodeBatch:
        tf, tfds = _require_tfds()
        p = Path(path)

        if p.is_dir():
            ds = tf.data.TFRecordDataset(list(str(f) for f in p.glob("*.tfrecord*")))
            dataset_name = p.name
            # RLDS TFRecords require knowing the feature spec; users should
            # subclass and provide it, or pass a pre-parsed tf.data.Dataset.
            raise NotImplementedError(
                "Reading raw RLDS TFRecord shards requires a feature spec. "
                "Pass a pre-parsed tf.data.Dataset or subclass RLDSReader and "
                "override `read` to supply the feature description."
            )
        else:
            dataset_name = path
            ds, info = tfds.load(path, split="train", with_info=True)
            n_episodes = info.splits["train"].num_examples

        def loader_fn(idx: int) -> Episode:
            fresh_ds = tfds.load(path, split=f"train[{idx}:{idx+1}]")
            ep = next(iter(fresh_ds))
            return self._episode_from_rlds(ep, idx, path, tf)

        def iterator_fn() -> Iterable[Episode]:
            for i, ep in enumerate(ds):
                yield self._episode_from_rlds(ep, i, path, tf)

        from calibra.schema.episode import LazyEpisodeList
        lazy_eps = LazyEpisodeList(
            loader_fn=loader_fn,
            length=n_episodes,
            iterator_fn=iterator_fn,
        )

        return EpisodeBatch(
            episodes=lazy_eps,
            dataset_name=dataset_name,
            format=self.format_name,
            source_path=path,
        )

    def _episode_from_rlds(
        self, ep: Any, ep_idx: int, source: str, tf: "tf"
    ) -> Episode:
        steps = ep["steps"]
        step_list = list(steps)

        obs_keys = [k for k in step_list[0]["observation"].keys()]
        obs: dict[str, list] = {k: [] for k in obs_keys}
        actions, timestamps = [], []

        for i, step in enumerate(step_list):
            for k in obs_keys:
                obs[k].append(step["observation"][k].numpy())
            actions.append(step["action"].numpy())
            ts = float(step["timestamp"].numpy()) if "timestamp" in step else i / self._fps
            timestamps.append(ts)

        return Episode(
            metadata=EpisodeMetadata(
                episode_id=str(ep_idx),
                source_file=source,
            ),
            timestamps=np.array(timestamps, dtype=np.float64),
            observations={k: np.array(v) for k, v in obs.items()},
            actions=np.array(actions, dtype=np.float32),
        )
