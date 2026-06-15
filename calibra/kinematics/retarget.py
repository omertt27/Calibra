"""
calibra.kinematics.retarget — Absolute-to-relative end-effector action converter.

NVIDIA GR00T N1.7+ uses a Relative End-Effector (EEF) action space: each
action is a 6-DoF delta transformation (Δposition, Δrotation) expressed in
the robot's *current* end-effector coordinate frame, not in world space.

Isaac Lab / robomimic HDF5 files record actions and observations in absolute
world-frame coordinates (Cartesian position + quaternion orientation). Passing
these directly to a relative-EEF policy breaks training: the policy learns to
predict world-frame displacements from a random origin, which do not generalise
across episodes or embodiments.

This module converts absolute 7-DoF Cartesian poses (x, y, z, qx, qy, qz, qw)
into 6-DoF relative action deltas (dx, dy, dz, droll, dpitch, dyaw) in the
current end-effector frame.

Mathematical formulation
------------------------
Given absolute EEF translation p_t ∈ ℝ³ and rotation quaternion q_t at step t,
with corresponding rotation matrix R_t ∈ SO(3):

  Position delta (local frame):
    Δp_t = R_t^T (p_{t+1} − p_t)

  Rotation delta (local frame):
    ΔR_t = R_t^T R_{t+1}
    converted to Euler angles (roll, pitch, yaw) in radians.

Output shape is (T−1, 6): one fewer step than the input because the last step
has no "next" state to compute a delta against.

Dependency: pip install 'calibra[kinematics]'  (scipy>=1.10)
"""
from __future__ import annotations

import numpy as np


def _require_scipy():
    try:
        from scipy.spatial.transform import Rotation
        return Rotation
    except ImportError:
        raise ImportError(
            "scipy is required for EEF retargeting.\n"
            "Install it with: pip install 'calibra[kinematics]'"
        ) from None


def absolute_to_relative_eef(poses: np.ndarray) -> np.ndarray:
    """
    Convert absolute 7D Cartesian poses to relative 6D action deltas.

    Parameters
    ----------
    poses : np.ndarray, shape (T, 7)
        Each row is [x, y, z, qx, qy, qz, qw] — world-frame position and
        quaternion orientation (scipy / ROS convention: scalar-last).

    Returns
    -------
    np.ndarray, shape (T−1, 6)
        Each row is [dx, dy, dz, droll, dpitch, dyaw] in the current
        end-effector frame (radians for rotation components).

    Raises
    ------
    ValueError
        If poses has fewer than 2 rows or wrong column count.
    ImportError
        If scipy is not installed.

    Notes
    -----
    The output has T−1 rows because computing a delta requires a "next" step.
    Append a zero row if your policy requires fixed-length action sequences.

    Rotation convention: Euler angles are intrinsic XYZ (roll → pitch → yaw)
    in radians, matching GR00T N1's 6-DoF controller input format.
    """
    Rotation = _require_scipy()

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(
            f"poses must have shape (T, 7), got {poses.shape}. "
            "Expected columns: [x, y, z, qx, qy, qz, qw]."
        )
    if len(poses) < 2:
        raise ValueError(
            f"Need at least 2 poses to compute a delta, got {len(poses)}."
        )

    positions = poses[:, :3]       # (T, 3)
    quaternions = poses[:, 3:]     # (T, 4)  scalar-last

    # Build Rotation objects for all steps at once (scipy handles the array).
    all_rots = Rotation.from_quat(quaternions)  # (T,)
    r_curr = all_rots[:-1]   # (T-1,)
    r_next = all_rots[1:]    # (T-1,)

    # Position delta expressed in the current EEF frame.
    dp_world = np.diff(positions, axis=0)       # (T-1, 3)
    dp_local = r_curr.inv().apply(dp_world)     # (T-1, 3)

    # Rotation delta expressed in the current EEF frame, as Euler angles.
    dr_local = r_curr.inv() * r_next            # (T-1,) composed Rotation
    euler_local = dr_local.as_euler("xyz", degrees=False)  # (T-1, 3) radians

    return np.concatenate([dp_local, euler_local], axis=1).astype(np.float32)


def retarget_episode_eef(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
) -> np.ndarray:
    """
    Convenience wrapper for Isaac Lab obs keys.

    Parameters
    ----------
    eef_pos  : (T, 3) world-frame end-effector positions.
    eef_quat : (T, 4) quaternions [qx, qy, qz, qw] (scalar-last).

    Returns
    -------
    np.ndarray, shape (T−1, 6) — relative EEF deltas.
    """
    eef_pos  = np.asarray(eef_pos,  dtype=np.float64)
    eef_quat = np.asarray(eef_quat, dtype=np.float64)
    if eef_pos.shape[0] != eef_quat.shape[0]:
        raise ValueError(
            f"eef_pos length {eef_pos.shape[0]} != eef_quat length {eef_quat.shape[0]}."
        )
    poses = np.concatenate([eef_pos, eef_quat], axis=1)  # (T, 7)
    return absolute_to_relative_eef(poses)
