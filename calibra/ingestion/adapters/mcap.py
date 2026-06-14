"""
MCAP adapter (ROS2 bag format).

MCAP is the recommended bag format for ROS2. This adapter reads MCAP files
and maps ROS2 topics to EpisodeBatch structure.

Topic mapping convention (configurable via constructor):
  observations  ← configurable list of ROS2 topics (images, joint states, etc.)
  actions       ← /joint_trajectory or user-specified action topic
  timestamps    ← derived from ROS2 header stamps or log time

Each MCAP file is treated as one episode. If multiple files are in a
directory, they become separate episodes.

Dependency: pip install 'calibra[mcap]'  (mcap, mcap-ros2-support)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

if TYPE_CHECKING:
    from mcap.reader import McapReader as _McapReader


def _require_mcap() -> tuple:
    try:
        from mcap.reader import make_reader
        return (make_reader,)
    except ImportError:
        raise ImportError(
            "The 'mcap' package is required for the MCAP adapter.\n"
            "Install it with: pip install 'calibra[mcap]'"
        ) from None


@register
class MCAPReader(DatasetReader):
    """
    Reads MCAP (ROS2 bag) datasets.

    Parameters
    ----------
    obs_topics : list of ROS2 topic strings to treat as observations.
                 Each topic becomes one key in Episode.observations.
    action_topic : ROS2 topic to treat as the action stream.
    use_header_stamp : if True, use the ROS2 header stamp as the timestamp;
                       otherwise use log_time from the MCAP index.
    """

    def __init__(
        self,
        obs_topics: list[str] | None = None,
        action_topic: str = "/joint_trajectory",
        use_header_stamp: bool = True,
    ) -> None:
        self._obs_topics = obs_topics or []
        self._action_topic = action_topic
        self._use_header_stamp = use_header_stamp

    @property
    def format_name(self) -> str:
        return "mcap"

    @classmethod
    def can_read(cls, path: str) -> bool:
        p = Path(path)
        if p.is_file() and p.suffix == ".mcap":
            return True
        if p.is_dir():
            return any(p.glob("*.mcap"))
        return False

    def read(self, path: str) -> EpisodeBatch:
        (make_reader,) = _require_mcap()
        p = Path(path)

        files = [p] if p.is_file() else sorted(p.glob("*.mcap"))
        if not files:
            raise ValueError(f"No .mcap files found at '{path}'")

        episodes: list[Episode] = []
        for f in files:
            episodes.append(self._read_file(make_reader, str(f)))

        return EpisodeBatch(
            episodes=episodes,
            dataset_name=p.stem if p.is_file() else p.name,
            format=self.format_name,
            source_path=str(p),
        )

    def _read_file(self, make_reader: object, path: str) -> Episode:
        obs_data: dict[str, list] = {t: [] for t in self._obs_topics}
        action_data: list[np.ndarray] = []
        timestamps: list[float] = []

        with open(path, "rb") as stream:
            reader = make_reader(stream)  # type: ignore[operator]
            for schema, channel, message, decoded in reader.iter_decoded_messages():
                topic = channel.topic
                ts_ns = message.log_time
                if self._use_header_stamp and hasattr(decoded, "header"):
                    ts_ns = decoded.header.stamp.sec * 1_000_000_000 + decoded.header.stamp.nanosec

                ts_s = ts_ns * 1e-9

                if topic in obs_data:
                    obs_data[topic].append(self._msg_to_array(decoded))
                    timestamps.append(ts_s)
                elif topic == self._action_topic:
                    action_data.append(self._msg_to_array(decoded))

        if not timestamps:
            raise ValueError(
                f"No messages found for obs topics {self._obs_topics} in '{path}'.\n"
                "Check that topic names match the bag contents."
            )

        obs = {t: np.array(v) for t, v in obs_data.items() if v}
        actions = np.array(action_data, dtype=np.float32) if action_data else np.empty((0,))
        ts_arr = np.array(timestamps, dtype=np.float64)

        return Episode(
            metadata=EpisodeMetadata(
                episode_id=Path(path).stem,
                source_file=path,
            ),
            timestamps=ts_arr,
            observations=obs,
            actions=actions,
        )

    @staticmethod
    def _msg_to_array(msg: object) -> np.ndarray:
        """Best-effort conversion of a ROS2 message to a flat numpy array."""
        if hasattr(msg, "data"):
            return np.asarray(msg.data, dtype=np.float32)
        if hasattr(msg, "position"):
            return np.array(msg.position, dtype=np.float32)
        raise TypeError(
            f"Cannot convert message type '{type(msg).__name__}' to numpy array.\n"
            "Subclass MCAPReader and override _msg_to_array for custom message types."
        )
