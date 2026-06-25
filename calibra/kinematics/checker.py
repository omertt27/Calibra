"""
calibra.kinematics.checker — URDF joint limit and velocity verification.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional, Dict, List, Tuple

import numpy as np

from calibra.schema.episode import Episode, EpisodeBatch


class KinematicURDFChecker:
    """
    Parses a robot's URDF model to audit joint limit violations,
    over-speed joint velocities, and anomalous physical states.
    """

    def __init__(self, urdf_path: str) -> None:
        self.urdf_path = urdf_path
        self.joint_limits = self._parse_urdf(urdf_path)

    def _parse_urdf(self, path: str) -> Dict[str, Dict[str, float]]:
        limits = {}
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            # Find all joints with a limit sub-element
            for joint in root.findall(".//joint"):
                name = joint.get("name")
                limit = joint.find("limit")
                if name and limit is not None:
                    limits[name] = {
                        "lower": float(limit.get("lower", -np.inf)),
                        "upper": float(limit.get("upper", np.inf)),
                        "velocity": float(limit.get("velocity", np.inf)),
                        "effort": float(limit.get("effort", np.inf)),
                    }
        except Exception as e:
            # Graceful warning if parsing fails or file is not found
            import sys

            print(f"Warning: Failed to parse URDF {path}: {e}", file=sys.stderr)
        return limits

    def check_episode(
        self, episode: Episode, joint_key: Optional[str] = None
    ) -> Dict[str, List[Tuple[int, float, str]]]:
        """
        Audit joint positions and velocities within a single episode.

        Returns:
            Dict mapping joint_name -> list of tuples (step_index, value, violation_type)
        """
        violations = {}
        obs = episode.observations

        # Candidates for joint observations
        keys_to_try = [joint_key] if joint_key else ["joint_positions", "joint_pos", "q", "proprio"]
        q_pos = None
        for k in keys_to_try:
            if k and k in obs:
                q_pos = obs[k]
                break

        if q_pos is None or q_pos.ndim != 2:
            return violations

        # Filter to active joints (joints with limits defined in URDF)
        active_joints = [
            name
            for name, lim in self.joint_limits.items()
            if lim["lower"] != -np.inf or lim["upper"] != np.inf or lim["velocity"] != np.inf
        ]

        n_cols = q_pos.shape[1]
        if n_cols == len(active_joints):
            joint_names = active_joints
        else:
            joint_names = list(self.joint_limits.keys())[:n_cols]

        dt = np.diff(episode.timestamps)
        dt = np.clip(dt, 1e-5, None)  # Avoid division by zero

        for col_idx, joint_name in enumerate(joint_names):
            if joint_name not in self.joint_limits:
                continue
            lim = self.joint_limits[joint_name]
            pos_traj = pos_traj = q_pos[:, col_idx]

            # Position limits
            for step_idx, pos in enumerate(pos_traj):
                if pos < lim["lower"]:
                    violations.setdefault(joint_name, []).append(
                        (step_idx, float(pos), f"position_underflow (limit: {lim['lower']})")
                    )
                elif pos > lim["upper"]:
                    violations.setdefault(joint_name, []).append(
                        (step_idx, float(pos), f"position_overflow (limit: {lim['upper']})")
                    )

            # Velocity limits
            if len(pos_traj) > 1:
                vel_traj = np.abs(np.diff(pos_traj) / dt)
                for step_idx, vel in enumerate(vel_traj):
                    if vel > lim["velocity"]:
                        violations.setdefault(joint_name, []).append(
                            (step_idx, float(vel), f"velocity_exceeded (limit: {lim['velocity']})")
                        )

        return violations

    def check_batch(self, batch: EpisodeBatch, joint_key: Optional[str] = None) -> Dict[str, Dict]:
        """
        Audit joint limits across all episodes in an EpisodeBatch.
        """
        batch_violations = {}
        for ep in batch.episodes:
            violations = self.check_episode(ep, joint_key=joint_key)
            if violations:
                batch_violations[ep.metadata.episode_id] = violations
        return batch_violations
