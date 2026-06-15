"""
calibra.metrics.kinematics — standalone vectorized kinematic estimators.

These are pure-numpy functions that operate directly on arrays. They are
suitable for notebooks, custom scripts, and profiling passes where you don't
want the full Pipeline/EpisodeBatch machinery.

For pipeline-integrated versions with bootstrap CI, per-episode tracking, and
AnalyzerResult output, use calibra.analyzers.smoothness.ControlSmoothnessAnalyzer.

All functions expect:
  states / actions : np.ndarray of shape (T, D) — T timesteps, D joint dimensions.
  dt               : float — time delta between consecutive frames (seconds).

Kinematics convention: "position control" (actions are positions).
For velocity or acceleration control, differentiate/integrate before calling.
"""
from __future__ import annotations

import numpy as np


def compute_velocity_discontinuity_rate(
    actions: np.ndarray,
    states: np.ndarray,
    dt: float,
    threshold: float = 0.05,
) -> float:
    """
    Fraction of frames where the commanded action diverges abruptly from the
    observed physical state (action-state tracking error).

    A large divergence indicates communication latency, packet drops, or a
    control mode mismatch (e.g. velocity commands compared against position
    observations produce meaninglessly large values).

    Parameters
    ----------
    actions   : commanded actions, shape (T, D).
    states    : observed joint states, shape (T, D). Must match actions shape.
    dt        : timestep in seconds (unused in this formulation but kept for
                API consistency with the σ-based jerk functions).
    threshold : L2 divergence above which a frame is flagged as discontinuous.
                Default 0.05 calibrated from ALOHA position-control hardware:
                clean teleop → divergence ≈ 0.01–0.05 (radians / normalised units).

    Returns
    -------
    float — fraction of frames exceeding the threshold. Range [0, 1].

    Confidence
    ----------
    NOT VALIDATED for threshold — calibrated from ALOHA hardware only.
    See calibra/knowledge_base/claims.yaml claim VD-001.
    """
    if actions.shape != states.shape:
        raise ValueError(
            f"actions shape {actions.shape} must match states shape {states.shape}."
        )
    l2_gaps = np.linalg.norm(actions - states, axis=1)
    return float(np.sum(l2_gaps > threshold) / len(actions))


def compute_jerk_spike_rate(
    states: np.ndarray,
    dt: float,
    sigma_limit: float = 5.0,
) -> float:
    """
    Fraction of frames where the third derivative of position (jerk) exceeds
    an n-sigma anomaly limit.

    Anomalous jerk spikes are a strong signal of:
      - Physical collisions or mechanical gear play
      - Sensor quantization errors or dropped frames
      - Incorrect episode boundary labels

    Parameters
    ----------
    states      : joint positions, shape (T, D).
    dt          : timestep in seconds.
    sigma_limit : jerk frames more than sigma_limit standard deviations above the
                  mean are flagged. Default 5.0 (roughly 1 in 3.5 million under
                  Gaussian assumptions; in practice catches genuine outliers).

    Returns
    -------
    float — fraction of jerk frames above the sigma limit. Range [0, 1].
            Returns 0.0 if jerk has zero variance (perfectly regular motion).

    Note
    ----
    The pipeline uses a k×median threshold instead (calibra.analyzers.smoothness).
    The σ-based threshold is more robust when the distribution has heavy tails
    but less so for bimodal speed profiles (fast approach + slow manipulation).
    See calibra/knowledge_base/claims.yaml claim JS-001.
    """
    if len(states) < 4:
        return 0.0

    velocities     = np.diff(states, axis=0) / dt
    accelerations  = np.diff(velocities, axis=0) / dt
    jerk           = np.diff(accelerations, axis=0) / dt

    jerk_norms = np.linalg.norm(jerk, axis=1)
    mean_jerk  = float(np.mean(jerk_norms))
    std_jerk   = float(np.std(jerk_norms))

    if std_jerk == 0.0:
        return 0.0

    spikes = np.sum(jerk_norms > (mean_jerk + sigma_limit * std_jerk))
    return float(spikes / len(jerk_norms))


def compute_ldlj(
    trajectory: np.ndarray,
    dt: float,
) -> float | None:
    """
    Logarithmic Dimensionless Jerk (LDLJ) — a smoothness metric from motor control.

    LDLJ = −log((T³ / v_max²) × ∫‖jerk‖² dt)

    The formula is dimensionless by construction. Less negative = smoother.
    Typical clean arm trajectories: −3 to −7.
    Values below −10 indicate concerning discontinuities.

    Parameters
    ----------
    trajectory : position trajectory, shape (T, D).
    dt         : timestep in seconds.

    Returns
    -------
    float — LDLJ value (negative; less negative is smoother).
    None  — if the trajectory is too short or has zero velocity.

    See also
    --------
    calibra.analyzers.smoothness._episode_ldlj — pipeline version with bootstrap CI.
    """
    if len(trajectory) < 5:
        return None

    T = len(trajectory) * dt
    vel   = np.diff(trajectory, axis=0) / dt
    if len(vel) < 3:
        return None
    acc   = np.diff(vel, axis=0) / dt
    if len(acc) < 2:
        return None
    jerk  = np.diff(acc, axis=0) / dt

    v_max = float(np.max(np.linalg.norm(vel, axis=-1)))
    if v_max <= 1e-12:
        return None

    jerk_sq_integral = float(np.sum(np.sum(jerk ** 2, axis=-1)) * dt)
    if jerk_sq_integral <= 0:
        return None

    inner = (T ** 3 / v_max ** 2) * jerk_sq_integral
    if inner <= 0:
        return None

    return float(-np.log(inner))


def compute_action_entropy(
    actions: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    Mean per-dimension Shannon entropy of the action distribution (bits/dim).

    Higher entropy → more diverse trajectories → better OOD generalisation.
    Below 3.0 bits/dim is a risk signal for trajectory redundancy (claim ENT-001).

    Parameters
    ----------
    actions : action array, shape (T, D).
    n_bins  : histogram bins per dimension. 50 works well for most datasets.

    Returns
    -------
    float — mean entropy in bits per action dimension.

    Confidence: PROVISIONAL — see claims.yaml ENT-001.
    """
    if actions.ndim == 1:
        actions = actions[:, np.newaxis]

    T, D = actions.shape
    entropy_per_dim: list[float] = []

    for d in range(D):
        col = actions[:, d]
        counts, _ = np.histogram(col, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy_per_dim.append(float(-np.sum(probs * np.log2(probs))))

    return float(np.mean(entropy_per_dim))
