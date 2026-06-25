"""
Isaac Lab HDF5 adapter.

Reads datasets in the Isaac Lab / robomimic HDF5 convention:

    /data/
        demo_0/
            obs/
                robot0_joint_pos       (T, n_joints)
                robot0_joint_vel       (T, n_joints)
                robot0_eef_pos         (T, 3)
                robot0_eef_quat        (T, 4)
                robot0_gripper_qpos    (T, n_gripper)
                agentview_image        (T, H, W, C)   uint8, optional
                robot0_eye_in_hand_image (T, H, W, C) uint8, optional
            actions                    (T, action_dim)
            dones                      (T,)
            rewards                    (T,)
            states                     (T, state_dim) optional
        demo_1/ ...
    /mask/
        train  (n_train,)  optional
        valid  (n_valid,)  optional
        test   (n_test,)   optional

Root-level attributes:
    total     : int   — total number of demos
    env       : str   — environment name
    env_args  : str   — JSON-encoded environment arguments (optional)

This is the canonical output format of Isaac Lab's TeleopDataCollector and
the robomimic data pipeline. It is also used by NVIDIA's GR00T training
infrastructure.

Detection: auto-selected when an HDF5 file contains a top-level "data/"
group whose children start with "demo". Takes priority over the generic
HDF5Reader because it is registered first.

Dependency: pip install 'calibra[hdf5]'  (h5py)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from calibra.ingestion.base import DatasetReader
from calibra.ingestion.registry import register
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

if TYPE_CHECKING:
    import h5py as _h5py


def _require_h5py() -> "_h5py":
    try:
        import h5py

        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for the Isaac Lab adapter.\n"
            "Install it with: pip install 'calibra[hdf5]'"
        ) from None


@register
class IsaacLabReader(DatasetReader):
    """Reads Isaac Lab / robomimic HDF5 datasets."""

    @property
    def format_name(self) -> str:
        return "isaac_lab"

    @classmethod
    def can_read(cls, path: str) -> bool:
        p = Path(path)
        if p.is_file() and p.suffix in (".hdf5", ".h5"):
            return cls._is_isaac_lab_file(str(p))
        if p.is_dir():
            for f in list(p.glob("*.hdf5")) + list(p.glob("*.h5")):
                if cls._is_isaac_lab_file(str(f)):
                    return True
        return False

    @staticmethod
    def _is_isaac_lab_file(path: str) -> bool:
        try:
            import h5py

            with h5py.File(path, "r") as f:
                if "data" not in f:
                    return False
                data_grp = f["data"]
                return any(k.startswith("demo") for k in data_grp.keys())
        except Exception:
            return False

    def read(self, path: str) -> EpisodeBatch:
        h5py = _require_h5py()
        p = Path(path)

        if p.is_dir():
            files = sorted(p.glob("*.hdf5")) + sorted(p.glob("*.h5"))
            files = [f for f in files if self._is_isaac_lab_file(str(f))]
            if not files:
                raise ValueError(f"No Isaac Lab HDF5 files found in {path}")
            episodes: list[Episode] = []
            env_name = ""
            for f in files:
                eps, name = self._read_file(h5py, str(f))
                episodes.extend(eps)
                env_name = env_name or name
            dataset_name = env_name or p.name
        else:
            episodes, env_name = self._read_file(h5py, str(p))
            dataset_name = env_name or p.stem

        return EpisodeBatch(
            episodes=episodes,
            dataset_name=dataset_name,
            format=self.format_name,
            source_path=str(p),
        )

    # ── internal ────────────────────────────────────────────────────────────

    def _read_file(self, h5py: "_h5py", path: str) -> tuple[list[Episode], str]:
        with h5py.File(path, "r") as f:
            env_name = str(f.attrs.get("env", "") or f.attrs.get("env_name", ""))
            success_mask = self._load_success_mask(f)

            data_grp = f["data"]
            demo_keys = sorted(
                (k for k in data_grp.keys() if k.startswith("demo")),
                key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0,
            )

            episodes: list[Episode] = []
            for key in demo_keys:
                demo = data_grp[key]
                ep = self._read_demo(demo, key, path, success_mask)
                episodes.append(ep)

        return episodes, env_name

    def _read_demo(
        self,
        demo: "_h5py.Group",
        key: str,
        source: str,
        success_mask: set[str],
    ) -> Episode:
        actions = np.array(demo["actions"], dtype=np.float32)
        n_steps = len(actions)

        timestamps = self._synthesize_timestamps(demo, n_steps)
        observations = self._load_obs(demo)

        # If a composite proprio isn't already present, build one from
        # kinematic obs so the smoothness analyzer can run action-state checks.
        if "proprio" not in observations:
            proprio = self._build_proprio(observations, n_steps)
            if proprio is not None:
                observations["proprio"] = proprio

        success: bool | None = None
        if "dones" in demo:
            dones = np.array(demo["dones"])
            success = bool(dones[-1]) if len(dones) > 0 else None
        if key in success_mask:
            success = True

        task = None
        if "model_file" in demo.attrs:
            task = str(demo.attrs["model_file"])

        meta = EpisodeMetadata(
            episode_id=key,
            task_description=task,
            success=success,
            source_file=source,
        )
        return Episode(
            metadata=meta,
            timestamps=timestamps,
            observations=observations,
            actions=actions,
        )

    @staticmethod
    def _load_obs(demo: "_h5py.Group") -> dict[str, np.ndarray]:
        import h5py

        obs: dict[str, np.ndarray] = {}
        if "obs" not in demo:
            return obs
        obs_grp = demo["obs"]
        for k in obs_grp.keys():
            if not isinstance(obs_grp[k], h5py.Dataset):
                continue
            arr = np.array(obs_grp[k])
            obs[k] = arr
            # Normalise camera/image keys so callers can find them reliably.
            if "image" in k.lower() or "rgb" in k.lower():
                canonical = f"camera_{k}"
                if canonical not in obs:
                    obs[canonical] = arr
        return obs

    @staticmethod
    def _build_proprio(obs: dict[str, np.ndarray], n_steps: int) -> np.ndarray | None:
        """Concatenate kinematic arrays into a single proprio vector."""
        _KINEMATIC_KEYS = [
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        parts = []
        for k in _KINEMATIC_KEYS:
            if k in obs:
                arr = obs[k]
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                if len(arr) == n_steps:
                    parts.append(arr.astype(np.float32))
        if not parts:
            return None
        return np.concatenate(parts, axis=1)

    @staticmethod
    def _synthesize_timestamps(demo: "_h5py.Group", n_steps: int) -> np.ndarray:
        """Build a timestamp array. Isaac Lab rarely embeds explicit clocks."""
        for key in ("timestamps", "timestamp", "t"):
            if key in demo:
                return np.array(demo[key], dtype=np.float64)
        # Default: 50 Hz (Isaac Lab default simulation step for manipulation).
        return np.arange(n_steps, dtype=np.float64) * 0.02

    @staticmethod
    def _load_success_mask(f: "_h5py.File") -> set[str]:
        """Return set of demo keys marked as train/valid (implicitly successful)."""
        mask: set[str] = set()
        if "mask" not in f:
            return mask
        mask_grp = f["mask"]
        for split in ("train", "valid"):
            if split in mask_grp:
                for key in np.array(mask_grp[split]).astype(str):
                    mask.add(str(key))
        return mask
