"""
GRAIL adapter (Generating Humanoid Loco-Manipulation).

GRAIL motion datasets consist of directories containing pickled (.pkl) files
for robot trajectories and object trajectories. Each robot trajectory pickle 
file typically has:
  dof_pos      (T, num_dof)
  dof_vel      (T, num_dof)
  root_state   (T, 13)

This adapter reads a directory of pickled files (or a single .pkl trajectory)
and packages them into an EpisodeBatch.
"""
from __future__ import annotations

import pickle
import gzip
from pathlib import Path
from typing import Any

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata


@register
class GRAILReader(DatasetReader):
    """Reads GRAIL trajectory pickle files."""

    def __init__(self, fps: float = 50.0) -> None:
        self._fps = fps

    @property
    def format_name(self) -> str:
        return "grail"

    @classmethod
    def can_read(cls, path: str) -> bool:
        p = Path(path)
        if p.is_file() and p.suffix in (".pkl", ".gz"):
            try:
                data = cls._load_pickle(p)
                return isinstance(data, dict) and ("dof_pos" in data or "root_state" in data)
            except Exception:
                return False
        if p.is_dir():
            # Check for robot/*.pkl subfiles or just flat *.pkl files
            pkl_files = sorted(p.glob("**/*.pkl")) + sorted(p.glob("**/*.pkl.gz"))
            if not pkl_files:
                return False
            # Check the first file to confirm it has dof_pos or root_state
            try:
                data = cls._load_pickle(pkl_files[0])
                return isinstance(data, dict) and ("dof_pos" in data or "root_state" in data)
            except Exception:
                return False
        return False

    def read(self, path: str) -> EpisodeBatch:
        p = Path(path)
        
        if p.is_file():
            files = [p]
            dataset_name = p.stem
        else:
            files = sorted(p.glob("**/*.pkl")) + sorted(p.glob("**/*.pkl.gz"))
            dataset_name = p.name

        episodes: list[Episode] = []
        for i, file_path in enumerate(files):
            try:
                data = self._load_pickle(file_path)
            except Exception as e:
                # Log error and skip corrupt pickles
                print(f"Warning: Failed to load pickle {file_path}: {e}")
                continue
            
            # Extract keys
            dof_pos = data.get("dof_pos")
            if dof_pos is None:
                # Root states might exist without dof_pos (e.g. object trajectory),
                # but for robot tracking we require dof_pos or joint position.
                root_state = data.get("root_state")
                if root_state is None:
                    continue
                # If only root_state exists, synthesize a dummy dof_pos to avoid errors in downstream analysis
                dof_pos = np.zeros((len(root_state), 1))
            
            T = len(dof_pos)
            timestamps = np.arange(T, dtype=np.float64) / self._fps
            
            # observations: joint position, base root state, dof velocities
            observations = {
                "dof_pos": dof_pos,
            }
            if "dof_vel" in data:
                observations["dof_vel"] = data["dof_vel"]
            if "root_state" in data:
                observations["root_state"] = data["root_state"]
            
            # Since actions in trajectory data are target commands, let's treat
            # the dof_pos as actions if no explicit 'action' key is defined.
            actions = data.get("actions") or data.get("action")
            if actions is None:
                actions = dof_pos
            
            meta = EpisodeMetadata(
                episode_id=file_path.name.replace(".gz", "").replace(".pkl", ""),
                source_file=str(file_path),
                extra={"original_keys": list(data.keys())}
            )
            
            episodes.append(Episode(
                metadata=meta,
                timestamps=timestamps,
                observations=observations,
                actions=np.array(actions, dtype=np.float32),
            ))
            
        if not episodes:
            raise ValueError(f"No valid GRAIL episodes with 'dof_pos' or 'root_state' found in {path}")
            
        return EpisodeBatch(
            episodes=episodes,
            dataset_name=dataset_name,
            format=self.format_name,
            source_path=str(p),
        )

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        if path.suffix == ".gz" or path.name.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        with open(path, "rb") as f:
            return pickle.load(f)
