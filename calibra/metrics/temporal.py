"""
calibra.metrics.temporal — standalone frame-rate stability estimators.

Pure-numpy functions for temporal quality checks. Suitable for notebooks and
custom scripts without the full Pipeline/EpisodeBatch machinery.

For pipeline-integrated versions with bootstrap CI and AnalyzerResult output,
use calibra.analyzers.temporal.TemporalAnalyzer.
"""

from __future__ import annotations

import numpy as np


def compute_jitter_cv(timestamps: np.ndarray) -> float:
    """
    Coefficient of variation (std / mean) of inter-frame time deltas.

    Measures how consistently the control loop fires. Near-zero for simulation
    (machine-precision clocks). Typically 5–25% for real hardware (USB camera
    + ROS recording stacks).

    Parameters
    ----------
    timestamps : 1-D array of timestamps in seconds, monotonically increasing.

    Returns
    -------
    float — CV of inter-frame deltas. 0 = perfectly regular; 0.15+ = noisy.

    Confidence: LOW for hardware — see claims.yaml TEMP-002 (not yet validated).
    """
    if len(timestamps) < 2:
        return 0.0
    deltas = np.diff(timestamps.astype(np.float64))
    mean_dt = float(np.mean(deltas))
    if mean_dt <= 0:
        return 0.0
    return float(np.std(deltas) / mean_dt)


def compute_dropout_rate(timestamps: np.ndarray, k: float = 3.0) -> float:
    """
    Fraction of inter-frame gaps that exceed k × median(delta).

    A gap larger than k × median indicates a dropped frame or missed control
    tick. Thresholds: warning at 1%, critical at 5%.

    Parameters
    ----------
    timestamps : 1-D array of timestamps in seconds, monotonically increasing.
    k          : gap threshold multiplier on the median delta. Default 3.0.

    Returns
    -------
    float — fraction of steps with a gap exceeding k × median. Range [0, 1].

    Confidence: NOT VALIDATED on real hardware — see claims.yaml TEMP-003.
    """
    if len(timestamps) < 2:
        return 0.0
    deltas = np.diff(timestamps.astype(np.float64))
    median_dt = float(np.median(deltas))
    if median_dt <= 0:
        return 0.0
    return float(np.sum(deltas > k * median_dt) / len(deltas))


def compute_frame_rate_stability(
    timestamps: np.ndarray,
) -> dict[str, float]:
    """
    Summary statistics for the frame rate across an episode or dataset.

    Parameters
    ----------
    timestamps : 1-D array of timestamps in seconds, monotonically increasing.

    Returns
    -------
    dict with keys:
      mean_fps    : mean frames per second
      std_fps     : standard deviation of instantaneous fps
      cv          : coefficient of variation (std/mean of deltas)
      dropout_rate: fraction of steps with a gap > 3× median delta
      n_frames    : number of frames
    """
    if len(timestamps) < 2:
        return {
            "mean_fps": 0.0,
            "std_fps": 0.0,
            "cv": 0.0,
            "dropout_rate": 0.0,
            "n_frames": len(timestamps),
        }

    deltas = np.diff(timestamps.astype(np.float64))
    fps_inst = 1.0 / np.where(deltas > 0, deltas, np.nan)

    mean_fps = float(np.nanmean(fps_inst))
    std_fps = float(np.nanstd(fps_inst))
    cv = compute_jitter_cv(timestamps)
    dropout = compute_dropout_rate(timestamps)

    return {
        "mean_fps": mean_fps,
        "std_fps": std_fps,
        "cv": cv,
        "dropout_rate": dropout,
        "n_frames": len(timestamps),
    }


def compute_multimodal_lag(
    primary_timestamps: np.ndarray,
    secondary_timestamps: np.ndarray,
) -> dict[str, float]:
    """
    Measure the alignment between two modality timestamp streams (e.g. action
    timestamps vs. camera observation timestamps).

    Parameters
    ----------
    primary_timestamps   : reference clock (e.g. action timestamps), shape (T,).
    secondary_timestamps : secondary modality (e.g. camera timestamps), shape (T,).
                           Must be same length as primary.

    Returns
    -------
    dict with keys:
      mean_lag_ms  : mean signed lag in milliseconds (positive = secondary lags behind)
      std_lag_ms   : standard deviation of lag in milliseconds
      max_lag_ms   : maximum absolute lag in milliseconds
      misaligned_fraction : fraction of steps where |lag| > 5 ms
    """
    if len(primary_timestamps) != len(secondary_timestamps):
        raise ValueError("Timestamp arrays must have the same length.")

    lags_s = secondary_timestamps.astype(np.float64) - primary_timestamps.astype(np.float64)
    lags_ms = lags_s * 1000.0

    return {
        "mean_lag_ms": float(np.mean(lags_ms)),
        "std_lag_ms": float(np.std(lags_ms)),
        "max_lag_ms": float(np.max(np.abs(lags_ms))),
        "misaligned_fraction": float(np.mean(np.abs(lags_ms) > 5.0)),
    }
