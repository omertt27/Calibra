"""
calibra.metrics — standalone pure-numpy metric estimators.

For quick scripting and notebooks. Does not depend on the pipeline schema.

For pipeline use (bootstrap CI, AnalyzerResult, per-episode tracking), use:
  calibra.analyzers.smoothness.ControlSmoothnessAnalyzer
  calibra.analyzers.temporal.TemporalAnalyzer
"""
from calibra.metrics.kinematics import (
    compute_action_entropy,
    compute_jerk_spike_rate,
    compute_ldlj,
    compute_velocity_discontinuity_rate,
)
from calibra.metrics.temporal import (
    compute_dropout_rate,
    compute_frame_rate_stability,
    compute_jitter_cv,
    compute_multimodal_lag,
)

__all__ = [
    "compute_velocity_discontinuity_rate",
    "compute_jerk_spike_rate",
    "compute_ldlj",
    "compute_action_entropy",
    "compute_jitter_cv",
    "compute_dropout_rate",
    "compute_frame_rate_stability",
    "compute_multimodal_lag",
]
