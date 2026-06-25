"""
calibra.temporal.drift — Camera-to-proprioception temporal drift detection.

In Isaac Sim 5.x/6.x the RGB/depth render pipeline can drift behind the
physics solver: the camera frame logged at step t may reflect the physical
state at step t−k, introducing a silent latency of k × dt that causes
policies trained on this data to make decisions based on stale visual input.
The result is sim-to-real transfer failure and policy flailing near contacts.

This module detects the drift by cross-correlating the *magnitude* of the
physical signal (joint velocity L2 norm) against the *magnitude* of the
visual signal (mean absolute frame difference, a proxy for optical flow).
Both signals should be time-aligned; a non-zero peak lag indicates drift.

Cross-correlation convention
----------------------------
`estimate_sensor_command_latency(physical, visual)` returns the lag k such
that visual[t] ≈ physical[t − k]:

  k > 0  : camera frames are BEHIND physics by k steps (camera lags).
  k = 0  : signals are aligned (no drift).
  k < 0  : camera appears ahead of physics (unexpected; check timestamps).

Threshold: GR00T N1 trains with camera + proprioception assumed synchronous.
A lag of more than 2 frames at 50 Hz (= 40 ms) is significant enough to
degrade multi-step action prediction.
"""

from __future__ import annotations

import numpy as np


def compute_visual_activity(images: np.ndarray) -> np.ndarray:
    """
    Compute per-frame visual activity magnitude as a proxy for optical flow.

    Uses mean absolute difference between consecutive frames — fast, dependency-
    free, and sufficient for detecting gross temporal misalignment.

    Parameters
    ----------
    images : np.ndarray, shape (T, H, W, C) or (T, H, W)
        Raw image frames. dtype may be uint8 or float.

    Returns
    -------
    np.ndarray, shape (T−1,)
        Mean absolute pixel difference between consecutive frames, one value
        per frame transition. Larger values indicate more visual motion.

    Raises
    ------
    ValueError  If images has fewer than 2 frames or wrong number of dims.
    """
    imgs = np.asarray(images, dtype=np.float32)
    if imgs.ndim not in (3, 4):
        raise ValueError(f"images must be (T, H, W) or (T, H, W, C), got shape {imgs.shape}.")
    if len(imgs) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(imgs)}.")
    diffs = np.abs(np.diff(imgs, axis=0))  # (T-1, H, W[, C])
    spatial_axes = tuple(range(1, diffs.ndim))
    return diffs.mean(axis=spatial_axes)  # (T-1,)


def estimate_sensor_command_latency(
    physical_signal: np.ndarray,
    visual_signal: np.ndarray,
) -> int:
    """
    Estimate the frame-level lag between a physical and a visual signal
    via 1-D cross-correlation of their normalized magnitudes.

    Parameters
    ----------
    physical_signal : np.ndarray, shape (N,)
        Physical activity magnitudes over time — typically the L2 norm of
        joint velocities per step.
    visual_signal : np.ndarray, shape (M,)
        Visual activity magnitudes over time — typically the output of
        `compute_visual_activity`. Length M may differ from N when the
        visual signal is derived from frame differences (M = N − 1).
        The two signals are truncated to the same length before correlation.

    Returns
    -------
    int
        Estimated lag k derived from the cross-correlation peak.
        Convention (from numpy cross-correlation of a against b):
          k < 0  : visual_signal lags physical_signal by |k| frames
                   (camera is BEHIND physics — the common Isaac Sim failure mode).
          k = 0  : signals are aligned.
          k > 0  : visual_signal leads physical_signal (unusual; check clock).
        Use abs(k) to compare against a frame-count threshold.

    Notes
    -----
    The cross-correlation is computed as:
        xcorr = np.correlate(p_norm, v_norm, mode='full')
    With lag k = lags[argmax(xcorr)].  If v[t] ≈ p[t - L] (visual lags by L),
    the peak is at k = -L (negative).  The GR00T drift check uses abs(k).
    """
    p = np.asarray(physical_signal, dtype=np.float64).ravel()
    v = np.asarray(visual_signal, dtype=np.float64).ravel()

    # Align lengths — visual signal from frame diffs is one step shorter.
    n = min(len(p), len(v))
    p, v = p[:n], v[:n]

    p_std = p.std()
    v_std = v.std()

    if p_std < 1e-12 or v_std < 1e-12:
        return 0  # constant signal — cannot estimate lag

    p_norm = (p - p.mean()) / p_std
    v_norm = (v - v.mean()) / v_std

    xcorr = np.correlate(p_norm, v_norm, mode="full")
    lags = np.arange(-(n - 1), n)
    return int(lags[int(np.argmax(xcorr))])


def estimate_visual_physics_lag(
    images: np.ndarray,
    joint_velocities: np.ndarray,
) -> int:
    """
    Convenience wrapper: compute visual activity from raw images and then
    estimate the lag against joint velocities.

    Parameters
    ----------
    images           : (T, H, W, C) image stack.
    joint_velocities : (T, D) joint velocity array (e.g. robot0_joint_vel).

    Returns
    -------
    int — estimated lag in frames (positive = camera lags physics).
    """
    visual_activity = compute_visual_activity(images)  # (T-1,)
    physical_activity = np.linalg.norm(joint_velocities, axis=1)  # (T,)
    return estimate_sensor_command_latency(physical_activity, visual_activity)
