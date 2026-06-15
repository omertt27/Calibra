"""
Schema Normalization Layer.

Different robotics dataset formats use incompatible naming conventions for the
same logical signals, e.g.:

  LeRobot v2  : observation.images.top, observation.images.wrist_cam
  Isaac Lab   : obs/camera_rgb, obs/proprio_state
  Robomimic   : robot0_eef_pos, agentview_image
  Custom HDF5 : observation.image.camera1, observation.state.joint_positions

This module provides a configuration-driven mapping that translates all
incoming observation keys into a unified internal representation BEFORE any
analyzer sees the data.

Usage
-----
    from calibra.schema.normalization import normalize_obs_keys

    raw = {"images.top": arr1, "state": arr2}
    normalized = normalize_obs_keys(raw)   # uses built-in defaults
    # → {"camera_top": arr1, "state": arr2}

    # Override with a custom mapping (additive — merged over defaults):
    normalized = normalize_obs_keys(raw, extra_mapping={"images.side": "camera_side"})

Mapping rules
-------------
Keys are matched in order:

1. Exact match against the combined mapping dict.
2. Prefix-strip aliases (e.g. "images." → "camera_").
3. If no rule matches, the key is passed through unchanged.

Only observation keys are normalized here.  Action column names are treated as
opaque vectors and are not renamed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# ── built-in default mapping ─────────────────────────────────────────────────
# Maps raw observation key (after format-specific prefix stripping) → canonical key.
# Canonical names follow the pattern: <modality>_<descriptor>.
# Conventions:
#   camera_*   : visual observations (RGB, depth, wrist, overhead, …)
#   state      : full proprioceptive state vector (if undifferentiated)
#   joint_*    : per-joint quantities
#   eef_*      : end-effector quantities
#   proprio    : any lumped proprioceptive vector that doesn't fit above

_DEFAULT_MAPPING: dict[str, str] = {
    # ── LeRobot image column names ────────────────────────────────────────────
    "images.top":           "camera_top",
    "images.wrist":         "camera_wrist",
    "images.wrist_cam":     "camera_wrist",
    "images.overhead":      "camera_overhead",
    "images.side":          "camera_side",
    "images.front":         "camera_front",
    "images.camera0":       "camera_0",
    "images.camera1":       "camera_1",
    "images.camera2":       "camera_2",
    "images.camera3":       "camera_3",
    # Generic camera aliases (observation.image.* prefix already stripped)
    "image":                "camera_main",
    "image.camera0":        "camera_0",
    "image.camera1":        "camera_1",
    "image.top":            "camera_top",
    "image.wrist":          "camera_wrist",
    # ── Proprioception / state ────────────────────────────────────────────────
    "state":                "state",
    "proprio":              "state",
    "proprio_state":        "state",
    "proprioception":       "state",
    "obs/proprio":          "state",
    # ── Joint-level quantities ────────────────────────────────────────────────
    "joint_pos":            "joint_position",
    "joint_position":       "joint_position",
    "joint_positions":      "joint_position",
    "joint_vel":            "joint_velocity",
    "joint_velocity":       "joint_velocity",
    "joint_velocities":     "joint_velocity",
    "joint_torque":         "joint_torque",
    "joint_torques":        "joint_torque",
    # ── End-effector ─────────────────────────────────────────────────────────
    "eef_pos":              "eef_position",
    "eef_position":         "eef_position",
    "eef_vel":              "eef_velocity",
    "eef_velocity":         "eef_velocity",
    "eef_quat":             "eef_orientation",
    "eef_orientation":      "eef_orientation",
    # ── Robomimic / Isaac Lab conventions ────────────────────────────────────
    "robot0_eef_pos":       "eef_position",
    "robot0_eef_quat":      "eef_orientation",
    "robot0_gripper_qpos":  "gripper_state",
    "robot0_joint_pos":     "joint_position",
    "robot0_joint_vel":     "joint_velocity",
    "obs/camera_rgb":       "camera_main",
    "obs/proprio_state":    "state",
    # ── Depth ────────────────────────────────────────────────────────────────
    "depth":                "camera_depth",
    "depth_image":          "camera_depth",
    "images.depth":         "camera_depth",
    # ── RLDS / Bridge conventions ────────────────────────────────────────────
    "image_primary":        "camera_main",
    "image_wrist":          "camera_wrist",
    # ── Gripper ──────────────────────────────────────────────────────────────
    "gripper":              "gripper_state",
    "gripper_state":        "gripper_state",
    "gripper_pos":          "gripper_state",
}

# ── prefix-strip rules ────────────────────────────────────────────────────────
# Applied when exact-match lookup fails. Pattern → replacement prefix.
# The replacement is prepended to the remainder of the key (after stripping prefix).
_PREFIX_RULES: list[tuple[str, str]] = [
    ("images.",   "camera_"),
    ("image.",    "camera_"),
    ("obs/",      ""),
    ("robot0_",   ""),
]


def normalize_obs_keys(
    observations: dict[str, np.ndarray],
    extra_mapping: Optional[dict[str, str]] = None,
) -> dict[str, np.ndarray]:
    """
    Translate raw observation keys to canonical names using the default mapping
    plus any caller-supplied overrides.

    Parameters
    ----------
    observations : raw observations dict from the format adapter.
    extra_mapping : optional dict of additional/override key mappings, merged
                    over the built-in defaults (extra_mapping wins on conflict).

    Returns
    -------
    A new dict with canonical keys.  Arrays are not copied — the same numpy
    objects are referenced.  If two raw keys map to the same canonical key,
    the last one (in iteration order) wins with a warning.
    """
    mapping = {**_DEFAULT_MAPPING}
    if extra_mapping:
        mapping.update(extra_mapping)

    result: dict[str, np.ndarray] = {}
    for raw_key, arr in observations.items():
        canonical = _resolve(raw_key, mapping)
        if canonical in result:
            import warnings
            warnings.warn(
                f"normalize_obs_keys: two raw keys map to '{canonical}' "
                f"— keeping last value. Provide extra_mapping to resolve ambiguity.",
                stacklevel=3,
            )
        result[canonical] = arr
    return result


def _resolve(raw_key: str, mapping: dict[str, str]) -> str:
    """Return the canonical name for a raw observation key."""
    # 1. Exact match
    if raw_key in mapping:
        return mapping[raw_key]

    # 2. Prefix-strip rules
    for prefix, replacement in _PREFIX_RULES:
        if raw_key.startswith(prefix):
            remainder = raw_key[len(prefix):]
            candidate = replacement + remainder
            # Check the remapped candidate against the mapping too
            if candidate in mapping:
                return mapping[candidate]
            return candidate

    # 3. Pass-through
    return raw_key
