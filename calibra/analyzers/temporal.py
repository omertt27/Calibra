"""
Temporal Stability Analyzer.

Computes:
  1. Timestamp jitter (coefficient of variation of step deltas) — measures
     how consistent the control loop frequency is across the dataset.
  2. Timestamp dropout — fraction of steps where the inter-step gap exceeds
     `dropout_k × median(delta)`, indicating missed ticks or dropped frames.
  3. Camera lag std — if per-modality obs_timestamps are available for camera
     modalities, measures the std-dev of the lag relative to the master clock.
  4. Action-observation misalignment — if action_timestamps differ from
     obs_timestamps, measures the fraction of steps exceeding `align_tol_ms`.

Confidence intervals are computed via percentile bootstrap over episodes
(i.e., the unit of resampling is the episode, not the step). This avoids
artificially narrow intervals from treating correlated within-episode steps
as i.i.d. samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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

# ── thresholds (all times in seconds unless noted) ──────────────────────────

_JITTER_CV_WARNING  = 0.15   # 15 % coefficient of variation
_JITTER_CV_CRITICAL = 0.30

_DROPOUT_K          = 3.0    # gap must exceed k × median_delta to count

_DROPOUT_WARNING    = 0.01   # 1 % of steps
_DROPOUT_CRITICAL   = 0.05   # 5 %

_CAM_LAG_WARNING_S  = 0.010  # 10 ms
_CAM_LAG_CRITICAL_S = 0.020  # 20 ms — from Calibra spec example

_ALIGN_TOL_S        = 0.005  # 5 ms action-obs alignment tolerance
_ALIGN_WARNING      = 0.01   # 1 % misaligned steps
_ALIGN_CRITICAL     = 0.05

_CAMERA_PREFIXES    = ("camera", "cam", "rgb", "depth", "wrist", "overhead")

# ── camera-physics drift thresholds (frames) ─────────────────────────────────

_DRIFT_WARNING_FRAMES:  int = 2   # 40 ms at 50 Hz
_DRIFT_CRITICAL_FRAMES: int = 5   # 100 ms at 50 Hz

# Observation key fragments used to detect joint-velocity arrays.
_JOINT_VEL_KEYS = frozenset(["joint_vel", "robot0_joint_vel", "velocity"])
# Observation key fragments used to detect image arrays.
_VISUAL_KEYS    = frozenset(["camera", "image", "rgb", "depth", "visual"])


@dataclass
class TemporalAnalyzer(Analyzer):
    """
    Temporal stability diagnostics.

    Parameters
    ----------
    jitter_cv_warning, jitter_cv_critical : CV thresholds for jitter risk levels.
    dropout_k           : multiplier on median_delta to declare a step a dropout.
    dropout_warning, dropout_critical : dropout fraction thresholds.
    cam_lag_warning_s, cam_lag_critical_s : camera lag std thresholds (seconds).
    align_tol_s         : per-step tolerance for action-obs alignment (seconds).
    align_warning, align_critical : misalignment fraction thresholds.
    drift_warning_frames, drift_critical_frames : camera-physics lag thresholds
                          (frames). Checked when both image and joint-velocity
                          observations are present. 2 frames ≈ 40 ms at 50 Hz.
    n_bootstrap         : number of bootstrap resamples for CIs.
    ci_level            : confidence level for all CIs.
    camera_keys         : explicit list of obs keys to treat as camera modalities.
                          If None, auto-detected by prefix matching.
    """

    jitter_cv_warning:    float = _JITTER_CV_WARNING
    jitter_cv_critical:   float = _JITTER_CV_CRITICAL
    dropout_k:            float = _DROPOUT_K
    dropout_warning:      float = _DROPOUT_WARNING
    dropout_critical:     float = _DROPOUT_CRITICAL
    cam_lag_warning_s:    float = _CAM_LAG_WARNING_S
    cam_lag_critical_s:   float = _CAM_LAG_CRITICAL_S
    align_tol_s:          float = _ALIGN_TOL_S
    align_warning:        float = _ALIGN_WARNING
    align_critical:       float = _ALIGN_CRITICAL
    drift_warning_frames: int   = _DRIFT_WARNING_FRAMES
    drift_critical_frames: int  = _DRIFT_CRITICAL_FRAMES
    n_bootstrap:          int   = 1000
    ci_level:             float = 0.95
    camera_keys:          Optional[list[str]] = None

    @property
    def name(self) -> str:
        return "temporal_stability"

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

        jitter_flag, jitter_raw = self._check_jitter(batch)
        if jitter_flag:
            flags.append(jitter_flag)
        raw["jitter"] = jitter_raw

        dropout_flag, dropout_raw = self._check_dropout(batch)
        if dropout_flag:
            flags.append(dropout_flag)
        raw["dropout"] = dropout_raw

        cam_keys = self._resolve_camera_keys(batch)
        for cam_key in cam_keys:
            lag_flag, lag_raw = self._check_camera_lag(batch, cam_key)
            if lag_flag:
                flags.append(lag_flag)
            raw[f"cam_lag_{cam_key}"] = lag_raw

        align_flag, align_raw = self._check_alignment(batch)
        if align_flag:
            flags.append(align_flag)
        raw["alignment"] = align_raw

        drift_flag, drift_raw = self._check_visual_physics_drift(batch)
        if drift_flag is not None:
            flags.append(drift_flag)
            raw["camera_physics_drift"] = drift_raw

        hints = self._policy_hints(flags, policy_family, raw)

        # Per-episode arrays for Phase 2 comparison/curation (convention: "per_episode_<key>").
        raw["per_episode_jitter_cv"]        = jitter_raw.get("episode_values", [])
        raw["per_episode_dropout_fraction"] = dropout_raw.get("episode_values", [])
        raw["per_episode_drift_lag_frames"] = drift_raw.get("episode_lags", []) if drift_raw else []

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )

    # ── metric: timestamp jitter ─────────────────────────────────────────────

    def _check_jitter(
        self, batch: EpisodeBatch
    ) -> tuple[Optional[RiskFlag], dict]:
        ep_values = [_episode_jitter_cv(ep) for ep in batch.episodes]
        cvs = [v for v in ep_values if v is not None]

        if not cvs:
            return None, {"skipped": "insufficient steps", "episode_values": ep_values}

        arr = np.array(cvs)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        raw = {"mean_cv": float(stat), "ci_lower": float(lo), "ci_upper": float(hi),
               "n_episodes": len(cvs), "episode_values": ep_values}

        level = self._threshold_level(
            stat, self.jitter_cv_warning, self.jitter_cv_critical
        )
        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="timestamp_jitter_cv",
                observed=ObservedValue(value=stat, ci_lower=lo, ci_upper=hi,
                                       ci_level=self.ci_level, ci_method="bootstrap"),
                threshold=self.jitter_cv_warning,
                interpretation="Step-to-step timing is consistent.",
                implication="No jitter risk detected.",
            ), raw

        return RiskFlag(
            level=level,
            metric="timestamp_jitter_cv",
            observed=ObservedValue(value=stat, ci_lower=lo, ci_upper=hi,
                                   ci_level=self.ci_level, ci_method="bootstrap"),
            threshold=self.jitter_cv_warning,
            interpretation=(
                f"High coefficient of variation in inter-step timing "
                f"({stat:.2%} mean CV across {len(cvs)} episodes)."
            ),
            implication=(
                "Irregular control-loop timing degrades time-series policies "
                "(transformers, diffusion) that assume fixed-frequency data. "
                "Consider resampling to a uniform frequency before training."
            ),
        ), raw

    # ── metric: timestamp dropout ────────────────────────────────────────────

    def _check_dropout(
        self, batch: EpisodeBatch
    ) -> tuple[Optional[RiskFlag], dict]:
        ep_values = [
            _episode_dropout_fraction(ep, self.dropout_k)
            for ep in batch.episodes
        ]
        arr = np.array(ep_values)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        raw = {"mean_dropout_fraction": float(stat), "ci_lower": float(lo),
               "ci_upper": float(hi), "dropout_k": self.dropout_k,
               "episode_values": ep_values}

        level = self._threshold_level(
            stat, self.dropout_warning, self.dropout_critical
        )
        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="timestamp_dropout_rate",
                observed=ObservedValue(value=stat, unit="fraction",
                                       ci_lower=lo, ci_upper=hi,
                                       ci_level=self.ci_level, ci_method="bootstrap"),
                threshold=self.dropout_warning,
                interpretation="Timestamp dropout rate is within acceptable range.",
                implication="No dropout risk detected.",
                affected_fraction=float(stat),
            ), raw

        return RiskFlag(
            level=level,
            metric="timestamp_dropout_rate",
            observed=ObservedValue(value=stat, unit="fraction",
                                   ci_lower=lo, ci_upper=hi,
                                   ci_level=self.ci_level, ci_method="bootstrap"),
            threshold=self.dropout_warning,
            interpretation=(
                f"{stat:.1%} of steps have inter-step gaps > {self.dropout_k}× "
                "the median step duration (dropped ticks or frame skips)."
            ),
            implication=(
                "Dropout creates artificial velocity discontinuities. "
                "BC policies will learn spurious high-jerk transitions at dropout "
                "sites. Filter or interpolate affected episodes before training."
            ),
            affected_fraction=float(stat),
        ), raw

    # ── metric: camera lag ───────────────────────────────────────────────────

    def _check_camera_lag(
        self, batch: EpisodeBatch, cam_key: str
    ) -> tuple[Optional[RiskFlag], dict]:
        lag_stds: list[float] = []
        for ep in batch.episodes:
            std = _episode_camera_lag_std(ep, cam_key)
            if std is not None:
                lag_stds.append(std)

        if not lag_stds:
            return None, {"skipped": f"no per-modality timestamps for '{cam_key}'"}

        arr = np.array(lag_stds)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        stat_ms, lo_ms, hi_ms = stat * 1000, lo * 1000, hi * 1000
        raw = {"mean_lag_std_ms": float(stat_ms), "ci_lower_ms": float(lo_ms),
               "ci_upper_ms": float(hi_ms), "n_episodes": len(lag_stds)}

        level = self._threshold_level(
            stat, self.cam_lag_warning_s, self.cam_lag_critical_s
        )
        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric=f"camera_lag_std[{cam_key}]",
                observed=ObservedValue(value=stat_ms, unit="ms",
                                       ci_lower=lo_ms, ci_upper=hi_ms,
                                       ci_level=self.ci_level, ci_method="bootstrap"),
                threshold=self.cam_lag_warning_s * 1000,
                interpretation=f"Camera '{cam_key}' lag variance is within threshold.",
                implication="No camera synchronisation risk detected.",
            ), raw

        return RiskFlag(
            level=level,
            metric=f"camera_lag_std[{cam_key}]",
            observed=ObservedValue(value=stat_ms, unit="ms",
                                   ci_lower=lo_ms, ci_upper=hi_ms,
                                   ci_level=self.ci_level, ci_method="bootstrap"),
            threshold=self.cam_lag_critical_s * 1000,
            interpretation=(
                f"Camera '{cam_key}' timestamp std-dev relative to master clock "
                f"is {stat_ms:.1f} ms (threshold: <{self.cam_lag_critical_s * 1000:.0f} ms)."
            ),
            implication=(
                "Closed-loop policies that fuse camera and proprioception will "
                "experience systematic observation–action desync, especially on "
                "contact transitions where precise timing matters."
            ),
        ), raw

    # ── metric: action-observation alignment ─────────────────────────────────

    def _check_alignment(
        self, batch: EpisodeBatch
    ) -> tuple[Optional[RiskFlag], dict]:
        fracs: list[float] = []
        for ep in batch.episodes:
            if ep.action_timestamps is None:
                continue
            frac = _episode_misalignment_fraction(ep, self.align_tol_s)
            fracs.append(frac)

        if not fracs:
            return None, {"skipped": "no separate action_timestamps in dataset"}

        arr = np.array(fracs)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        raw = {"mean_misalign_fraction": float(stat), "ci_lower": float(lo),
               "ci_upper": float(hi), "tolerance_ms": self.align_tol_s * 1000}

        level = self._threshold_level(
            stat, self.align_warning, self.align_critical
        )
        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="action_obs_misalignment",
                observed=ObservedValue(value=stat, unit="fraction",
                                       ci_lower=lo, ci_upper=hi,
                                       ci_level=self.ci_level, ci_method="bootstrap"),
                threshold=self.align_warning,
                interpretation="Action and observation timestamps are well-aligned.",
                implication="No alignment risk detected.",
                affected_fraction=float(stat),
            ), raw

        return RiskFlag(
            level=level,
            metric="action_obs_misalignment",
            observed=ObservedValue(value=stat, unit="fraction",
                                   ci_lower=lo, ci_upper=hi,
                                   ci_level=self.ci_level, ci_method="bootstrap"),
            threshold=self.align_warning,
            interpretation=(
                f"{stat:.1%} of steps have action-observation timestamp offsets "
                f"> {self.align_tol_s * 1000:.0f} ms."
            ),
            implication=(
                "Imitation learning assumes each (obs, action) pair is causally "
                "aligned. Misaligned samples corrupt the BC gradient signal — "
                "the policy learns to predict actions for the wrong observation."
            ),
            affected_fraction=float(stat),
        ), raw

    # ── policy-conditioned hints ──────────────────────────────────────────────

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

        jitter_cv = raw.get("jitter", {}).get("mean_cv")
        dropout   = raw.get("dropout", {}).get("mean_dropout_fraction")
        drift_lag = raw.get("camera_physics_drift", {}).get("median_lag_frames") if raw.get("camera_physics_drift") else None
        has_cam_lag_crit = any(
            f.level == RiskLevel.CRITICAL and "camera_lag" in f.metric
            for f in flags
        )
        has_drift_issue = (
            drift_lag is not None and abs(drift_lag) > self.drift_warning_frames
        )

        if "diffusion" in pf:
            caveats: list[str] = []
            compatible: Optional[bool] = True
            if jitter_cv is not None and jitter_cv > self.jitter_cv_warning:
                caveats.append(
                    "Diffusion Policy conditions on observation history; "
                    "high jitter makes the implicit time-axis noisy."
                )
                compatible = None
            if has_cam_lag_crit:
                caveats.append(
                    "Camera lag will cause visual observations to lag behind "
                    "proprioception, corrupting multi-modal conditioning."
                )
                compatible = False
            if has_drift_issue:
                caveats.append(
                    f"Camera-physics render lag of {abs(drift_lag)} frames detected. "
                    "Visual observations reflect a stale physical state, which "
                    "corrupts diffusion conditioning on multi-modal history."
                )
                compatible = False
            hints.append(CompatibilityHint(
                policy_family="Diffusion Policy",
                compatible=compatible,
                explanation="Temporal stability is a key prerequisite for "
                            "diffusion-based BC policies.",
                caveats=caveats,
            ))

        if pf in ("act", "action chunking"):
            caveats = []
            compatible = True
            if dropout is not None and dropout > self.dropout_warning:
                caveats.append(
                    "ACT uses fixed-length action chunks — dropout creates "
                    "variable-length gaps that misalign chunk boundaries."
                )
                compatible = None
            hints.append(CompatibilityHint(
                policy_family="ACT",
                compatible=compatible,
                explanation="ACT is sensitive to consistent episode step counts "
                            "and action chunk alignment.",
                caveats=caveats,
            ))

        if "transformer" in pf:
            caveats = []
            compatible = True
            if jitter_cv is not None and jitter_cv > self.jitter_cv_critical:
                caveats.append(
                    "Transformer policies with positional encoding assume "
                    "approximately uniform step intervals."
                )
                compatible = None
            if has_drift_issue:
                caveats.append(
                    f"Camera-physics render lag of {abs(drift_lag)} frames detected. "
                    "Transformer token sequences will mix observations from "
                    "different physical states, degrading temporal attention."
                )
                compatible = None
            hints.append(CompatibilityHint(
                policy_family="Transformer",
                compatible=compatible,
                explanation="Transformer BC policies rely on positional encoding "
                            "that assumes fixed-rate sequences.",
                caveats=caveats,
            ))

        return hints

    # ── metric: camera-physics temporal drift ────────────────────────────────

    def _check_visual_physics_drift(
        self, batch: EpisodeBatch
    ) -> tuple[Optional[RiskFlag], Optional[dict]]:
        """
        Detect render-pipeline lag by cross-correlating joint-velocity magnitude
        against visual activity magnitude (mean absolute frame difference).

        Only runs when the batch contains both:
          - at least one image observation (ndim 3 or 4, spatial dims ≥ 8 px), AND
          - at least one joint-velocity observation (key matches _JOINT_VEL_KEYS).

        Silently returns (None, None) when prerequisites are not met — i.e. for
        datasets without image data (e.g. proprioception-only LeRobot datasets).

        Returns
        -------
        (RiskFlag | None, dict | None)
            RiskFlag with metric "camera_physics_drift" and lag in frames, or
            None if prerequisites are not met.
        """
        from calibra.temporal.drift import compute_visual_activity, estimate_sensor_command_latency

        lag_samples: list[int] = []

        for ep in batch.episodes:
            jv_arr: Optional[np.ndarray] = None
            for key in _JOINT_VEL_KEYS:
                if key in ep.observations:
                    jv_arr = ep.observations[key]
                    break

            img_arr: Optional[np.ndarray] = None
            for key in ep.observations:
                if any(kw in key.lower() for kw in _VISUAL_KEYS):
                    candidate = ep.observations[key]
                    if (
                        candidate.ndim in (3, 4)
                        and candidate.shape[1] >= 8
                        and candidate.shape[2] >= 8
                    ):
                        img_arr = candidate
                        break

            if jv_arr is None or img_arr is None:
                continue
            if len(img_arr) < 4 or len(jv_arr) < 4:
                continue

            try:
                visual_activity = compute_visual_activity(img_arr)
                physical_activity = (
                    np.linalg.norm(jv_arr.astype(np.float32), axis=1)
                    if jv_arr.ndim > 1
                    else np.abs(jv_arr.astype(np.float32))
                )
                lag = estimate_sensor_command_latency(physical_activity, visual_activity)
                lag_samples.append(lag)
            except Exception:
                continue

        if not lag_samples:
            return None, None

        median_lag = int(np.median(lag_samples))
        raw: dict = {
            "median_lag_frames": median_lag,
            "n_episodes_checked": len(lag_samples),
            "episode_lags": lag_samples,
        }

        abs_lag = abs(median_lag)
        if abs_lag <= self.drift_warning_frames:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="camera_physics_drift",
                observed=ObservedValue(value=float(median_lag), unit="frames"),
                threshold=float(self.drift_warning_frames),
                interpretation=(
                    f"Camera-proprioception temporal alignment: {median_lag:+d} frames "
                    f"(within ±{self.drift_warning_frames} frame tolerance)."
                ),
                implication="No significant render-pipeline lag detected.",
            ), raw

        level = RiskLevel.CRITICAL if abs_lag > self.drift_critical_frames else RiskLevel.WARNING
        direction = "behind" if median_lag < 0 else "ahead of"
        return RiskFlag(
            level=level,
            metric="camera_physics_drift",
            observed=ObservedValue(value=float(median_lag), unit="frames"),
            threshold=float(self.drift_warning_frames),
            interpretation=(
                f"Camera frames are {abs_lag} frames {direction} physics "
                f"(median over {len(lag_samples)} episodes). "
                f"Threshold: ±{self.drift_warning_frames} frames."
            ),
            implication=(
                "Temporal misalignment between visual and proprioceptive observations "
                "causes policies to make decisions from stale visual input. "
                "This is a known Isaac Sim 5.x/6.x render-pipeline issue. "
                "Apply timestamp correction or use `calibra retarget` to re-align "
                "before training."
            ),
            affected_fraction=float(sum(1 for l in lag_samples if abs(l) > self.drift_warning_frames) / len(lag_samples)),
        ), raw

    # ── helpers ──────────────────────────────────────────────────────────────

    def _resolve_camera_keys(self, batch: EpisodeBatch) -> list[str]:
        if self.camera_keys is not None:
            return self.camera_keys
        return [
            k for k in batch.modalities
            if any(k.lower().startswith(p) for p in _CAMERA_PREFIXES)
        ]

    @staticmethod
    def _threshold_level(
        value: float, warning: float, critical: float
    ) -> RiskLevel:
        if value >= critical:
            return RiskLevel.CRITICAL
        if value >= warning:
            return RiskLevel.WARNING
        return RiskLevel.OK


# ── per-episode metric helpers (pure functions, testable in isolation) ───────

def _episode_jitter_cv(ep: Episode) -> Optional[float]:
    """Coefficient of variation of inter-step deltas. None if < 3 steps."""
    if ep.n_steps < 3:
        return None
    deltas = np.diff(ep.timestamps)
    mean_dt = float(np.mean(deltas))
    if mean_dt <= 0:
        return None
    return float(np.std(deltas) / mean_dt)


def _episode_dropout_fraction(ep: Episode, k: float = 3.0) -> float:
    """Fraction of steps with inter-step gap > k × median gap."""
    if ep.n_steps < 3:
        return 0.0
    deltas = np.diff(ep.timestamps)
    median_dt = float(np.median(deltas))
    if median_dt <= 0:
        return 0.0
    return float(np.mean(deltas > k * median_dt))


def _episode_camera_lag_std(ep: Episode, cam_key: str) -> Optional[float]:
    """
    Std-dev of (obs_timestamps[cam_key] - master_clock), in seconds.
    Returns None if per-modality timestamps are not available for this key.
    """
    if cam_key not in ep.obs_timestamps:
        return None
    ref = ep.action_timestamps if ep.action_timestamps is not None else ep.timestamps
    cam = ep.obs_timestamps[cam_key]
    n = min(len(ref), len(cam))
    if n < 2:
        return None
    lags = cam[:n] - ref[:n]
    return float(np.std(lags))


def _episode_misalignment_fraction(ep: Episode, tol_s: float) -> float:
    """Fraction of steps where |action_ts - obs_ts| > tol_s."""
    if ep.action_timestamps is None:
        return 0.0
    n = min(len(ep.action_timestamps), len(ep.timestamps))
    if n == 0:
        return 0.0
    diffs = np.abs(ep.action_timestamps[:n] - ep.timestamps[:n])
    return float(np.mean(diffs > tol_s))


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI. Returns (point_estimate, lower, upper).
    Resampling unit is the element of `values` (i.e. episodes, not steps).
    """
    rng = np.random.default_rng(seed)
    if len(values) == 1:
        v = float(stat_fn(values))
        return v, v, v

    boot_stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci_level) / 2.0
    return (
        float(stat_fn(values)),
        float(np.quantile(boot_stats, alpha)),
        float(np.quantile(boot_stats, 1.0 - alpha)),
    )
