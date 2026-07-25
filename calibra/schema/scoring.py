"""
Scoring rubric for Calibra public reports.

Rubric identifier: robot-dataset-quality-v1.0

Methodology identifiers are versioned independently of the Calibra package
version. A Calibra release may fix the CLI without changing any rubric;
the rubric may change without any CLI changes.

v1 uses flag-severity-based normalization: each flag's level maps directly
to a metric score. Per-metric scoring functions (threshold interpolation)
will replace this in v2 once rubric calibration data is available from the
public audit corpus.
"""

from __future__ import annotations

from calibra.schema.report import RiskLevel

CURRENT_RUBRIC = "robot-dataset-quality-v1.0"

# ── dimension weights (must sum to 1.0) ───────────────────────────────────────

DIMENSION_WEIGHTS: dict[str, float] = {
    "temporal_integrity": 0.30,
    "motion_quality": 0.30,
    "behavioral_coverage": 0.20,
    "task_integrity": 0.20,
}

# ── metric → dimension routing ────────────────────────────────────────────────
# Substring match against lowercased metric name; first match wins.
# task_integrity is the catch-all for anything that doesn't match below.

_TEMPORAL_TERMS = frozenset(
    {
        "timestamp",
        "jitter",
        "dropout",
        "lag",
        "sync",
        "align",
        "temporal",
    }
)
_MOTION_TERMS = frozenset(
    {
        "ldlj",
        "jerk",
        "velocity_discontinuity",
        "vel_disc",
        "smoothness",
        "divergence",
    }
)
_COVERAGE_TERMS = frozenset(
    {
        "entropy",
        "ssl",
        "novelty",
        "coverage",
        "diversity",
        "embed",
    }
)


def route_metric_to_dimension(metric_name: str) -> str:
    key = metric_name.lower()
    if any(t in key for t in _TEMPORAL_TERMS):
        return "temporal_integrity"
    if any(t in key for t in _MOTION_TERMS):
        return "motion_quality"
    if any(t in key for t in _COVERAGE_TERMS):
        return "behavioral_coverage"
    return "task_integrity"


# ── flag level → metric score (v1 rubric) ─────────────────────────────────────

_LEVEL_SCORE: dict[RiskLevel, float] = {
    RiskLevel.OK: 100.0,
    RiskLevel.INFO: 90.0,
    RiskLevel.WARNING: 65.0,
    RiskLevel.CRITICAL: 30.0,
}


def flag_level_to_score(level: RiskLevel) -> float:
    return _LEVEL_SCORE[level]


# ── aggregation helpers ───────────────────────────────────────────────────────


def dimension_score(metric_scores: list[float]) -> float:
    """Mean of metric scores; 100.0 when there are no flags in the dimension."""
    return sum(metric_scores) / len(metric_scores) if metric_scores else 100.0


def overall_score(
    dimension_scores: dict[str, float],
    weights: dict[str, float] = DIMENSION_WEIGHTS,
) -> float:
    weighted = sum(dimension_scores.get(d, 100.0) * w for d, w in weights.items())
    return weighted / sum(weights.values())


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ── versioned methodology identifiers ─────────────────────────────────────────
# Stable IDs — change only when the underlying computation changes, not when
# the package version changes.

_METHODOLOGY: dict[str, str] = {
    "temporal_jitter_cv": "temporal.jitter_cv.v1",
    "dropout_rate": "temporal.dropout_rate.v1",
    "camera_lag": "temporal.camera_lag.v1",
    "action_obs_misalignment": "temporal.action_obs_align.v1",
    "ldlj": "motion.ldlj.v1",
    "spike_rate": "motion.jerk_spike_rate.v1",
    "vel_disc_rate": "motion.velocity_discontinuity.v1",
    "action_state_divergence": "motion.action_state_divergence.v1",
    "action_entropy": "coverage.action_entropy.v1",
    "state_entropy": "coverage.state_entropy.v1",
    "trajectory_diversity": "coverage.trajectory_diversity.v1",
    "contact_density": "task.contact_density.v1",
    "grasp_events": "task.grasp_events.v1",
}


def get_methodology(metric_name: str) -> str:
    return _METHODOLOGY.get(metric_name, f"calibra.{metric_name}.v1")
