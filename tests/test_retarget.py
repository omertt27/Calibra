"""Tests for calibra.kinematics.retarget — EEF action space converter."""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy", reason="scipy required for kinematics")

from calibra.kinematics.retarget import absolute_to_relative_eef, retarget_episode_eef  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────


def _identity_pose(n: int) -> np.ndarray:
    """Return n poses all at origin with identity quaternion [0,0,0,1]."""
    poses = np.zeros((n, 7))
    poses[:, 6] = 1.0  # qw = 1
    return poses


def _pure_translation(dx: float, dy: float, dz: float, n: int = 5) -> np.ndarray:
    """Linear translation along (dx, dy, dz) per step, no rotation."""
    poses = np.zeros((n, 7))
    poses[:, 6] = 1.0
    poses[:, 0] = np.arange(n) * dx
    poses[:, 1] = np.arange(n) * dy
    poses[:, 2] = np.arange(n) * dz
    return poses


# ── shape and type ────────────────────────────────────────────────────────────


class TestRetargetShape:
    def test_output_shape(self):
        poses = _identity_pose(10)
        out = absolute_to_relative_eef(poses)
        assert out.shape == (9, 6)

    def test_output_dtype_float32(self):
        poses = _identity_pose(5)
        out = absolute_to_relative_eef(poses)
        assert out.dtype == np.float32

    def test_two_steps_gives_one_delta(self):
        poses = _pure_translation(0.1, 0.0, 0.0, n=2)
        out = absolute_to_relative_eef(poses)
        assert out.shape == (1, 6)

    def test_raises_on_single_step(self):
        with pytest.raises(ValueError, match="at least 2"):
            absolute_to_relative_eef(_identity_pose(1))

    def test_raises_on_wrong_columns(self):
        with pytest.raises(ValueError, match="shape"):
            absolute_to_relative_eef(np.zeros((5, 6)))


# ── pure translation (no rotation) ───────────────────────────────────────────


class TestPureTranslation:
    def test_x_translation_maps_to_dx(self):
        poses = _pure_translation(0.1, 0.0, 0.0, n=5)
        out = absolute_to_relative_eef(poses)
        # At identity rotation, local frame == world frame
        np.testing.assert_allclose(out[:, 0], 0.1, atol=1e-5)  # dx
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-5)  # dy
        np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-5)  # dz

    def test_y_translation_maps_to_dy(self):
        poses = _pure_translation(0.0, 0.2, 0.0, n=5)
        out = absolute_to_relative_eef(poses)
        np.testing.assert_allclose(out[:, 1], 0.2, atol=1e-5)

    def test_no_rotation_in_pure_translation(self):
        poses = _pure_translation(0.1, 0.2, 0.3, n=5)
        out = absolute_to_relative_eef(poses)
        np.testing.assert_allclose(out[:, 3:], 0.0, atol=1e-5)  # droll, dpitch, dyaw

    def test_stationary_gives_zero_delta(self):
        poses = _identity_pose(5)
        out = absolute_to_relative_eef(poses)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)


# ── rotation expressed in local frame ────────────────────────────────────────


class TestRotationInLocalFrame:
    def test_90deg_rotation_followed_by_x_translation_becomes_rotated(self):
        """
        After a 90° yaw (rotation about Z), a world-frame +X step should appear
        as a +Y step in the rotated local EEF frame.
        """
        from scipy.spatial.transform import Rotation as R

        n = 3
        poses = np.zeros((n, 7))

        # Step 0: origin, no rotation
        poses[0, 3:] = R.from_euler("z", 0, degrees=True).as_quat()

        # Step 1: origin, 90° yaw
        poses[1, 3:] = R.from_euler("z", 90, degrees=True).as_quat()

        # Step 2: move +0.1 in world X, keep 90° yaw
        poses[2, 0] = 0.1
        poses[2, 3:] = R.from_euler("z", 90, degrees=True).as_quat()

        out = absolute_to_relative_eef(poses)

        # Step 1 delta (orientation change, no translation)
        np.testing.assert_allclose(out[0, :3], 0.0, atol=1e-5)

        # Step 2 delta: +0.1 world X in 90°-yawed frame.
        # After 90° CCW yaw, robot's local Y axis points in world -X direction,
        # so world +X = local -Y.
        np.testing.assert_allclose(out[1, 0], 0.0, atol=1e-5)  # local dx ≈ 0
        np.testing.assert_allclose(out[1, 1], -0.1, atol=1e-5)  # local dy ≈ -0.1


# ── convenience wrapper ───────────────────────────────────────────────────────


class TestRetargetEpisodeEEF:
    def test_matches_absolute_to_relative(self):
        rng = np.random.default_rng(0)
        from scipy.spatial.transform import Rotation as R

        n = 20
        positions = rng.random((n, 3)).astype(np.float64) * 0.5
        quats = R.random(n, random_state=0).as_quat()

        via_wrapper = retarget_episode_eef(positions, quats)
        poses = np.concatenate([positions, quats], axis=1)
        via_direct = absolute_to_relative_eef(poses)

        np.testing.assert_allclose(via_wrapper, via_direct, atol=1e-6)

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            retarget_episode_eef(np.zeros((5, 3)), np.zeros((4, 4)))
