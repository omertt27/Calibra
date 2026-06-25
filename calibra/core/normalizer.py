"""
calibra.core.normalizer — configurable schema normalizer.

Maps heterogeneous joint/sensor naming conventions to canonical Calibra channel
names. Works from a YAML config file so teams can add robot-specific mappings
without touching Python code.

The built-in defaults cover LeRobot, Robomimic, Isaac Lab, and RLDS conventions.
Custom entries in a mappings YAML are merged on top and take precedence.

Usage
-----
    # Use built-in defaults only
    n = SchemaNormalizer()
    canonical = n.normalize({"observation.state": arr, "action": arr2})

    # Override with a custom YAML config
    n = SchemaNormalizer(config_path="calibra/core/mappings.yaml")
    translation = n.map_columns(["observation.state", "joint_cmd"])
    # → {"observation.state": "observation_joints", "joint_cmd": "action_joints"}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_DEFAULT_MAPPINGS: dict[str, list[str]] = {
    "action_joints": ["action", "joint_cmd", "joint_command", "motors/target"],
    "observation_joints": ["observation.state", "joint_state", "joint_states", "motors/position"],
    "eef_position": ["eef_pos", "ee_pos", "end_effector_position", "robot0_eef_pos"],
    "eef_orientation": ["eef_quat", "ee_quat", "end_effector_quat", "robot0_eef_quat"],
    "gripper_state": ["gripper", "gripper_pos", "gripper_state", "robot0_gripper_qpos"],
    "joint_position": [
        "joint_pos",
        "joint_position",
        "joint_positions",
        "robot0_joint_pos",
        "dof_pos",
    ],
    "joint_velocity": [
        "joint_vel",
        "joint_velocity",
        "joint_velocities",
        "robot0_joint_vel",
        "dof_vel",
    ],
    "camera_top": ["observation.images.top", "images.top", "cam_top"],
    "camera_wrist": ["observation.images.wrist", "images.wrist", "wrist_camera"],
    "proprio": ["observation.proprio", "proprio_state", "proprioception"],
}


class SchemaNormalizer:
    """
    Maps heterogeneous sensor/joint naming schemas to consistent Calibra channels.

    Parameters
    ----------
    config_path : optional path to a YAML mappings file. If provided, its
                  entries are merged over the built-in defaults.

    Example
    -------
    >>> n = SchemaNormalizer()
    >>> n.map_columns(["observation.state", "action", "timestamp"])
    {'observation.state': 'observation_joints', 'action': 'action_joints', 'timestamp': 'timestamp'}
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._mappings: dict[str, list[str]] = dict(_DEFAULT_MAPPINGS)

        if config_path is not None:
            self._load_yaml(config_path)

    def _load_yaml(self, config_path: str) -> None:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load a mappings config file.\n"
                "Install it with: pip install pyyaml"
            ) from None

        with open(config_path) as f:
            data = yaml.safe_load(f)

        extra = data.get("mappings", {})
        for canonical, patterns in extra.items():
            if isinstance(patterns, list):
                self._mappings[canonical] = patterns

    def map_columns(self, raw_columns: List[str]) -> Dict[str, str]:
        """
        Translate raw sensor/joint strings to canonical Calibra channel names.

        Parameters
        ----------
        raw_columns : list of column names from a raw dataset.

        Returns
        -------
        dict mapping each raw column to its canonical name (pass-through if
        no pattern matches).
        """
        translation: dict[str, str] = {}
        for col in raw_columns:
            translation[col] = self._resolve(col)
        return translation

    def normalize(
        self,
        observations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply the mapping to an observations dict, returning canonical keys.

        Arrays are not copied — the same objects are referenced under new keys.
        If two raw keys map to the same canonical key, the last value wins.
        """
        result: dict[str, Any] = {}
        for raw_key, value in observations.items():
            canonical = self._resolve(raw_key)
            result[canonical] = value
        return result

    def _resolve(self, raw_key: str) -> str:
        for canonical, patterns in self._mappings.items():
            if any(pattern in raw_key for pattern in patterns):
                return canonical
        return raw_key  # pass-through if no match

    # ── integration with the built-in Python normalizer ───────────────────────

    def as_extra_mapping(self) -> Dict[str, str]:
        """
        Return a flat {raw_key: canonical} dict suitable for passing to
        calibra.schema.normalization.normalize_obs_keys(extra_mapping=...).

        Only exact-pattern entries (single-token patterns) are included.
        """
        flat: dict[str, str] = {}
        for canonical, patterns in self._mappings.items():
            for pattern in patterns:
                flat[pattern] = canonical
        return flat
