"""
Control Smoothness Analyzer.

Computes three metrics over the action trajectory of each episode:

  1. LDLJ (Logarithmic Dimensionless Jerk)
     A standard smoothness metric from motor control literature.
     LDLJ = -log((T^3 / v_max^2) * integral(||jerk||^2 dt))
     The formula is dimensionless by construction (verified in docstring below).
     Less negative = smoother. Typical smooth arm trajectories: -3 to -7.
     Values below -10 are a strong signal of training-corrupting discontinuities.

  2. Jerk spike rate
     Fraction of steps where ||jerk_t|| > k × median(||jerk||).
     Spikes indicate sharp, non-smooth transitions — often a sign of
     bad episode boundaries, dropped frames, or incorrectly labelled contacts.

  3. Velocity discontinuity rate
     Fraction of steps where ||Δv_t|| > threshold × ||v_max||.
     Catches sudden velocity reversals that would not be captured by
     per-step jerk (because jerk is already three derivatives deep).

Notes on action_type:
  "position"     (default): actions are positions → differentiate 3×.
  "velocity":               actions are velocities → differentiate 2×.
  "acceleration":           actions are accelerations → differentiate 1×.

The gripper dimension (typically the last column) is excluded from all
smoothness computations because it is binary/discrete and would
artificially inflate jerk. Use `gripper_dims` to override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import (
    AnalyzerResult,
    CompatibilityHint,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)
from calibra.analyzers.temporal import _bootstrap_ci   # reuse bootstrap helper

# ── thresholds ───────────────────────────────────────────────────────────────

_LDLJ_WARNING  = -10.0   # below this → concerning
_LDLJ_CRITICAL = -15.0

_JERK_SPIKE_WARNING  = 0.02   # 2% of steps
_JERK_SPIKE_CRITICAL = 0.05

_VEL_DISC_THRESHOLD = 0.20   # Δv > 20% of v_max counts as a discontinuity
_VEL_DISC_WARNING   = 0.02
_VEL_DISC_CRITICAL  = 0.05


@dataclass
class ControlSmoothnessAnalyzer(Analyzer):
    """
    Action trajectory smoothness diagnostics.

    Parameters
    ----------
    action_type : "position" | "velocity" | "acceleration"
        Semantic type of the action signal. Controls how many derivatives
        are computed before reaching jerk.
    gripper_dims : list of column indices to exclude from smoothness metrics.
        Default is [-1] (last column). Pass [] to include all columns.
    jerk_spike_k : multiplier on median jerk magnitude to define a spike.
    ldlj_warning, ldlj_critical : LDLJ thresholds (both negative — more
        negative is worse).
    vel_disc_threshold : fraction of v_max above which a Δv step is a
        discontinuity.
    n_bootstrap, ci_level : bootstrap CI parameters.
    """

    action_type:      str         = "position"
    gripper_dims:     list[int]   = field(default_factory=lambda: [-1])
    jerk_spike_k:     float       = 5.0
    ldlj_warning:     float       = _LDLJ_WARNING
    ldlj_critical:    float       = _LDLJ_CRITICAL
    jerk_spike_warning:  float    = _JERK_SPIKE_WARNING
    jerk_spike_critical: float    = _JERK_SPIKE_CRITICAL
    vel_disc_threshold:  float    = _VEL_DISC_THRESHOLD
    vel_disc_warning:    float    = _VEL_DISC_WARNING
    vel_disc_critical:   float    = _VEL_DISC_CRITICAL
    n_bootstrap:      int         = 1000
    ci_level:         float       = 0.95

    @property
    def name(self) -> str:
        return "control_smoothness"

    # ── public entry point ───────────────────────────────────────────────────

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        flags: list[RiskFlag] = []
        raw: dict = {}

        ldlj_flag, ldlj_raw = self._check_ldlj(batch)
        flags.append(ldlj_flag)
        raw["ldlj"] = ldlj_raw

        spike_flag, spike_raw = self._check_jerk_spikes(batch)
        flags.append(spike_flag)
        raw["jerk_spikes"] = spike_raw

        disc_flag, disc_raw = self._check_vel_discontinuities(batch)
        flags.append(disc_flag)
        raw["vel_discontinuities"] = disc_raw

        hints = self._policy_hints(flags, policy_family, raw)

        # Per-episode arrays for Phase 2 comparison/curation (convention: "per_episode_<key>").
        raw["per_episode_ldlj"]           = ldlj_raw.get("episode_values", [])
        raw["per_episode_spike_rate"]     = spike_raw.get("episode_values", [])
        raw["per_episode_vel_disc_rate"]  = disc_raw.get("episode_values", [])

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )

    # ── metric: LDLJ ────────────────────────────────────────────────────────

    def _check_ldlj(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        ep_values = [_episode_ldlj(ep, self.action_type, self._active_dims(ep))
                     for ep in batch.episodes]
        values = [v for v in ep_values if v is not None]

        if not values:
            return self._skip_flag("ldlj", "insufficient episode length for LDLJ"), {
                "episode_values": ep_values
            }

        arr = np.array(values)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        raw = {"mean_ldlj": float(stat), "ci_lower": float(lo), "ci_upper": float(hi),
               "n_episodes": len(values), "episode_values": ep_values}

        level = _threshold_level_lower(stat, self.ldlj_warning, self.ldlj_critical)

        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="ldlj",
                observed=ObservedValue(value=stat, ci_lower=lo, ci_upper=hi,
                                       ci_level=self.ci_level, ci_method="bootstrap"),
                threshold=self.ldlj_warning,
                interpretation="Action trajectories are smooth (LDLJ within threshold).",
                implication="No smoothness risk detected.",
            ), raw

        return RiskFlag(
            level=level,
            metric="ldlj",
            observed=ObservedValue(value=stat, ci_lower=lo, ci_upper=hi,
                                   ci_level=self.ci_level, ci_method="bootstrap"),
            threshold=self.ldlj_warning,
            interpretation=(
                f"Mean LDLJ = {stat:.2f} across {len(values)} episodes "
                f"(threshold: >{self.ldlj_warning:.0f}). "
                "Action trajectories contain significant jerk."
            ),
            implication=(
                "High jerk in demonstration data forces the policy to learn "
                "discontinuous action transitions. BC policies trained on jerky "
                "data produce jerky rollouts that stress hardware and reduce "
                "task success on contact-rich tasks. Consider applying action "
                "smoothing (e.g. Savitzky-Golay) before training."
            ),
        ), raw

    # ── metric: jerk spikes ──────────────────────────────────────────────────

    def _check_jerk_spikes(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        ep_values = [_episode_jerk_spike_fraction(
            ep, self.action_type, self._active_dims(ep), self.jerk_spike_k
        ) for ep in batch.episodes]
        fracs = [f for f in ep_values if f is not None]

        if not fracs:
            return self._skip_flag("jerk_spike_rate", "insufficient episode length"), {
                "episode_values": ep_values
            }

        arr = np.array(fracs)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        raw = {"mean_spike_fraction": float(stat), "ci_lower": float(lo),
               "ci_upper": float(hi), "spike_k": self.jerk_spike_k,
               "episode_values": ep_values}

        level = _threshold_level_upper(stat, self.jerk_spike_warning, self.jerk_spike_critical)

        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="jerk_spike_rate",
                observed=ObservedValue(value=stat, unit="fraction",
                                       ci_lower=lo, ci_upper=hi,
                                       ci_level=self.ci_level, ci_method="bootstrap"),
                threshold=self.jerk_spike_warning,
                interpretation="Jerk spike rate is within acceptable range.",
                implication="No jerk spike risk detected.",
                affected_fraction=float(stat),
            ), raw

        return RiskFlag(
            level=level,
            metric="jerk_spike_rate",
            observed=ObservedValue(value=stat, unit="fraction",
                                   ci_lower=lo, ci_upper=hi,
                                   ci_level=self.ci_level, ci_method="bootstrap"),
            threshold=self.jerk_spike_warning,
            interpretation=(
                f"{stat:.1%} of steps have jerk > {self.jerk_spike_k}× "
                "median jerk — anomalous discontinuities in action sequence."
            ),
            implication=(
                "Jerk spikes are typically caused by dropped frames, bad episode "
                "boundaries, or incorrect time alignment. They create spurious "
                "high-gradient targets that BC policies overfit to, producing "
                "unstable rollouts."
            ),
            affected_fraction=float(stat),
        ), raw

    # ── metric: velocity discontinuities ─────────────────────────────────────

    def _check_vel_discontinuities(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        ep_values = [_episode_vel_disc_fraction(
            ep, self.action_type, self._active_dims(ep), self.vel_disc_threshold
        ) for ep in batch.episodes]
        fracs = [f for f in ep_values if f is not None]

        if not fracs:
            return self._skip_flag("velocity_discontinuity_rate",
                                   "insufficient episode length"), {
                "episode_values": ep_values
            }

        arr = np.array(fracs)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        raw = {"mean_disc_fraction": float(stat), "ci_lower": float(lo),
               "ci_upper": float(hi), "threshold": self.vel_disc_threshold,
               "episode_values": ep_values}

        level = _threshold_level_upper(stat, self.vel_disc_warning, self.vel_disc_critical)

        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="velocity_discontinuity_rate",
                observed=ObservedValue(value=stat, unit="fraction",
                                       ci_lower=lo, ci_upper=hi,
                                       ci_level=self.ci_level, ci_method="bootstrap"),
                threshold=self.vel_disc_warning,
                interpretation="Velocity profile is continuous — no sudden reversals.",
                implication="No velocity discontinuity risk detected.",
                affected_fraction=float(stat),
            ), raw

        return RiskFlag(
            level=level,
            metric="velocity_discontinuity_rate",
            observed=ObservedValue(value=stat, unit="fraction",
                                   ci_lower=lo, ci_upper=hi,
                                   ci_level=self.ci_level, ci_method="bootstrap"),
            threshold=self.vel_disc_warning,
            interpretation=(
                f"{stat:.1%} of steps show velocity change > "
                f"{self.vel_disc_threshold:.0%} of peak velocity."
            ),
            implication=(
                "Sudden velocity reversals are physically impossible under normal "
                "actuation constraints. These are either annotation errors or "
                "real but undesired operator behaviour. Policies trained on them "
                "will attempt to reproduce them, stressing hardware actuators."
            ),
            affected_fraction=float(stat),
        ), raw

    # ── policy hints ─────────────────────────────────────────────────────────

    def _policy_hints(
        self,
        flags: list[RiskFlag],
        policy_family: Optional[str],
        raw: dict,
    ) -> list[CompatibilityHint]:
        if not policy_family:
            return []

        pf = policy_family.lower()
        hints: list[CompatibilityHint] = []
        ldlj = raw.get("ldlj", {}).get("mean_ldlj")
        spike_rate = raw.get("jerk_spikes", {}).get("mean_spike_fraction")

        if "diffusion" in pf:
            caveats: list[str] = []
            compatible: Optional[bool] = True
            if ldlj is not None and ldlj < self.ldlj_warning:
                caveats.append(
                    "Diffusion Policy score-matching is sensitive to multimodal "
                    "action distributions; jerk spikes can manifest as spurious "
                    "high-energy modes in the learned score function."
                )
                compatible = None
            hints.append(CompatibilityHint(
                policy_family="Diffusion Policy",
                compatible=compatible,
                explanation="Smooth demonstrations are important for diffusion score quality.",
                caveats=caveats,
            ))

        if pf in ("act", "action chunking"):
            caveats = []
            compatible = True
            if spike_rate is not None and spike_rate > self.jerk_spike_warning:
                caveats.append(
                    "ACT predicts action chunks; if jerk spikes cluster at "
                    "episode boundaries, chunks will span discontinuities and "
                    "the cross-attention mechanism will struggle."
                )
                compatible = None
            hints.append(CompatibilityHint(
                policy_family="ACT",
                compatible=compatible,
                explanation="ACT action chunks are sensitive to intra-chunk smoothness.",
                caveats=caveats,
            ))

        return hints

    # ── helpers ──────────────────────────────────────────────────────────────

    def _active_dims(self, ep: Episode) -> list[int]:
        """Return action column indices, excluding gripper_dims."""
        n_dims = ep.actions.shape[1] if ep.actions.ndim > 1 else 1
        all_dims = list(range(n_dims))
        exclude = {d % n_dims for d in self.gripper_dims}
        return [d for d in all_dims if d not in exclude]

    @staticmethod
    def _skip_flag(metric: str, reason: str) -> RiskFlag:
        return RiskFlag(
            level=RiskLevel.INFO,
            metric=metric,
            observed=ObservedValue(value=None),
            interpretation=f"Metric skipped: {reason}.",
            implication="Cannot assess this metric with available data.",
        )


# ── per-episode metric helpers ───────────────────────────────────────────────

def _get_velocity(
    ep: Episode, action_type: str, active_dims: list[int]
) -> Optional[np.ndarray]:
    """
    Return velocity array (T-k, D) for the active action dimensions.
    k depends on how many derivatives are needed to reach velocity.
    Returns None if the episode is too short or actions are static.
    """
    acts = ep.actions
    if acts.ndim == 1:
        acts = acts[:, np.newaxis]

    if active_dims:
        acts = acts[:, active_dims]

    dt = float(np.median(np.diff(ep.timestamps))) if ep.n_steps > 1 else None
    if dt is None or dt <= 0:
        return None

    if action_type == "position":
        if len(acts) < 2:
            return None
        return np.diff(acts, axis=0) / dt
    elif action_type == "velocity":
        return acts
    elif action_type == "acceleration":
        if len(acts) < 2:
            return None
        return np.cumsum(acts, axis=0) * dt   # integrate to get velocity
    else:
        raise ValueError(f"Unknown action_type: {action_type!r}. "
                         "Expected 'position', 'velocity', or 'acceleration'.")


def _get_jerk(
    vel: np.ndarray, dt: float, action_type: str
) -> Optional[np.ndarray]:
    """Differentiate velocity to reach jerk (2 more derivatives)."""
    if len(vel) < 3:
        return None
    acc = np.diff(vel, axis=0) / dt
    if len(acc) < 2:
        return None
    # For action_type=="acceleration", vel was integrated so one more diff suffices.
    # For all types, we need jerk from velocity (2 diffs).
    jerk = np.diff(acc, axis=0) / dt
    return jerk


def _episode_ldlj(
    ep: Episode,
    action_type: str = "position",
    active_dims: Optional[list[int]] = None,
) -> Optional[float]:
    """
    LDLJ = -log((T^3 / v_max^2) * integral(||jerk||^2 dt))

    Dimensionality check (all in SI units, position in arbitrary scale):
      jerk: [u/s^3]  → ||jerk||^2: [u^2/s^6]
      integral(||jerk||^2 dt): [u^2/s^5]
      T^3 / v_max^2: [s^3 / (u/s)^2] = [s^3 * s^2 / u^2] = [s^5/u^2]
      product: [s^5/u^2 * u^2/s^5] = dimensionless ✓

    Returns None if the episode is too short or has zero-velocity content.
    """
    if ep.n_steps < 5:
        return None

    dt = float(np.median(np.diff(ep.timestamps)))
    if dt <= 0:
        return None

    T = ep.duration_s
    if T <= 0:
        return None

    vel = _get_velocity(ep, action_type, active_dims or [])
    if vel is None or len(vel) < 3:
        return None

    jerk = _get_jerk(vel, dt, action_type)
    if jerk is None:
        return None

    speed = np.linalg.norm(vel, axis=-1)
    v_max = float(np.max(speed))
    if v_max <= 1e-12:
        return None

    jerk_sq_integral = float(np.sum(np.sum(jerk ** 2, axis=-1)) * dt)
    if jerk_sq_integral <= 0:
        return None

    inner = (T ** 3 / v_max ** 2) * jerk_sq_integral
    if inner <= 0:
        return None

    return float(-np.log(inner))


def _episode_jerk_spike_fraction(
    ep: Episode,
    action_type: str = "position",
    active_dims: Optional[list[int]] = None,
    k: float = 5.0,
) -> Optional[float]:
    """Fraction of steps with ||jerk|| > k × median(||jerk||)."""
    if ep.n_steps < 5:
        return None

    dt = float(np.median(np.diff(ep.timestamps)))
    if dt <= 0:
        return None

    vel = _get_velocity(ep, action_type, active_dims or [])
    if vel is None:
        return None

    jerk = _get_jerk(vel, dt, action_type)
    if jerk is None:
        return None

    jerk_mag = np.linalg.norm(jerk, axis=-1)
    median_jerk = float(np.median(jerk_mag))
    if median_jerk <= 0:
        return 0.0

    return float(np.mean(jerk_mag > k * median_jerk))


def _episode_vel_disc_fraction(
    ep: Episode,
    action_type: str = "position",
    active_dims: Optional[list[int]] = None,
    threshold: float = 0.20,
) -> Optional[float]:
    """Fraction of steps with ||Δv|| > threshold × max(||v||)."""
    if ep.n_steps < 3:
        return None

    dt = float(np.median(np.diff(ep.timestamps)))
    if dt <= 0:
        return None

    vel = _get_velocity(ep, action_type, active_dims or [])
    if vel is None or len(vel) < 2:
        return None

    speed = np.linalg.norm(vel, axis=-1)
    v_max = float(np.max(speed))
    if v_max <= 1e-12:
        return 0.0

    delta_v = np.linalg.norm(np.diff(vel, axis=0), axis=-1)
    return float(np.mean(delta_v > threshold * v_max))


# ── shared threshold helpers ─────────────────────────────────────────────────

def _threshold_level_upper(
    value: float, warning: float, critical: float
) -> RiskLevel:
    """Higher value is worse (e.g. spike rate, dropout rate)."""
    if value >= critical:
        return RiskLevel.CRITICAL
    if value >= warning:
        return RiskLevel.WARNING
    return RiskLevel.OK


def _threshold_level_lower(
    value: float, warning: float, critical: float
) -> RiskLevel:
    """Lower value is worse (e.g. LDLJ — more negative is worse)."""
    if value <= critical:
        return RiskLevel.CRITICAL
    if value <= warning:
        return RiskLevel.WARNING
    return RiskLevel.OK
