"""
Calibration Drift Analyzer.

Detects a systematic per-motor offset between the commanded action and the
observed joint state during stationary ("stable") frames — the signature of
a stale leader/follower calibration. See LeRobot issue #3758: a stable ~17°
offset on one joint produced a ~6cm Cartesian gripper error at deployment,
while training loss looked normal because the policy simply learned the
biased mapping.

Deliberately restricted to stable frames (state barely moving for a
sustained run, i.e. a hold/pause): during active motion, the action-state
gap is dominated by real tracking lag/latency, a different and much larger
phenomenon already covered by ControlSmoothnessAnalyzer's
action_state_divergence. A hold is the only place a *pure* calibration
offset is visible uncontaminated by dynamics.

Post-hoc and read-only — uses `action` and `observations["state"|...]`
already present in most teleoperation datasets, no new data collection.

Thresholds are relative to each motor's own observed range (state units
are dataset-specific — radians, normalized, ticks — so there is no
universal absolute cutoff, the same reasoning BlurAnalyzer applies to
Laplacian variance). Not yet calibrated against a reference dataset with a
known injected offset (unlike action_state_divergence's 12-profile
calibration in smoothness.py) — shipped conservatively, capped at WARNING
and never CRITICAL, until validated against real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import AnalyzerResult, ObservedValue, RiskFlag, RiskLevel

_STATE_KEYS = ("state", "joint_position", "proprio")

_MIN_STABLE_RUN = 5  # consecutive steps required to count as a genuine hold
_STABLE_VEL_FRAC = 0.05  # per-step state velocity below this fraction of the
# motor's own std (times sqrt(active dims), same idiom as
# temporal.py's _check_action_dropout) counts as "stationary"

_MIN_STABLE_FRAMES = 20  # pooled stable frames needed before reporting anything
_MIN_CONSISTENCY = 2.0  # |mean offset| / std(offset) must exceed this to call
# it a systematic bias rather than noise around zero

_OFFSET_WARNING_FRAC = 0.03  # |mean offset| / motor's own dynamic range


def _find_state_array(ep: Episode) -> Optional[np.ndarray]:
    for key in _STATE_KEYS:
        if key in ep.observations:
            candidate = ep.observations[key]
            if candidate.ndim == 2 and candidate.shape == ep.actions.shape:
                return candidate
    return None


def _stable_mask(mask: np.ndarray, min_run: int) -> np.ndarray:
    """True only for positions belonging to a run of >= min_run consecutive
    True values in `mask` — an isolated low-velocity sample (e.g. a
    zero-crossing mid-motion) doesn't count as a hold."""
    out = np.zeros_like(mask, dtype=bool)
    run_start = 0
    run = 0
    for i, v in enumerate(mask):
        if v:
            run += 1
        else:
            if run >= min_run:
                out[run_start:i] = True
            run = 0
            run_start = i + 1
    if run >= min_run:
        out[run_start : run_start + run] = True
    return out


def _episode_stable_offsets(
    ep: Episode, active: list[int], vel_frac: float, min_run: int
) -> Optional[np.ndarray]:
    """(n_stable, len(active)) array of (action - state) at stable frames,
    or None if this episode has no usable state pairing / no stable run."""
    state = _find_state_array(ep)
    if state is None or ep.n_steps < min_run + 1:
        return None

    act = ep.actions[:, active]
    obs = state[:, active]

    state_std = float(np.mean(np.std(obs, axis=0))) or 1.0
    moving_thresh = vel_frac * state_std * (len(active) ** 0.5)
    step_delta = np.linalg.norm(np.diff(obs, axis=0), axis=1)
    # step_delta[t] is the transition state[t] -> state[t+1]; treat state[t+1]
    # as stable when the transition into it was small.
    stationary = np.concatenate([[False], step_delta < moving_thresh])
    stable = _stable_mask(stationary, min_run)
    if not np.any(stable):
        return None
    return (act - obs)[stable]


@dataclass
class CalibrationDriftAnalyzer(Analyzer):
    """
    Flags a systematic per-motor (action - state) offset observed during
    stationary/hold frames, pooled across the dataset.

    Parameters
    ----------
    gripper_dims : column indices excluded (binary/discrete, not a
                   calibration-relevant joint). Default is [-1] (last column).
    stable_vel_frac : fraction of a motor's own std that counts as
                       "not moving" for a single step.
    min_stable_run  : consecutive stationary steps required to count as a hold.
    min_stable_frames : pooled stable-frame count required before reporting.
    consistency_min : |mean offset| / std(offset) threshold for calling an
                       offset systematic rather than noise around zero.
    offset_warning_frac : |mean offset| / motor's own (max-min) range,
                           above which a motor is flagged.
    """

    requires = frozenset({"proprio"})

    gripper_dims: list[int] = field(default_factory=lambda: [-1])
    stable_vel_frac: float = _STABLE_VEL_FRAC
    min_stable_run: int = _MIN_STABLE_RUN
    min_stable_frames: int = _MIN_STABLE_FRAMES
    consistency_min: float = _MIN_CONSISTENCY
    offset_warning_frac: float = _OFFSET_WARNING_FRAC

    @property
    def name(self) -> str:
        return "calibration_drift"

    def _active_dims(self, ep: Episode) -> list[int]:
        dim = ep.action_dim
        excluded = {d % dim for d in self.gripper_dims if dim}
        return [d for d in range(dim) if d not in excluded]

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        per_ep_offsets = []
        per_ep_states = []
        active_dims: Optional[list[int]] = None
        for ep in batch.episodes:
            active = self._active_dims(ep)
            if active_dims is None:
                active_dims = active
            elif active != active_dims:
                continue  # inconsistent action_dim across episodes, skip pairing
            offsets = _episode_stable_offsets(ep, active, self.stable_vel_frac, self.min_stable_run)
            if offsets is not None:
                per_ep_offsets.append(offsets)
                state = _find_state_array(ep)
                if state is not None:
                    per_ep_states.append(state[:, active])

        if not per_ep_offsets:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="joint_offset_max_abs",
                        observed=ObservedValue(value=None),
                        interpretation=(
                            "No paired action/state observations with a sustained "
                            "stationary hold were found — calibration drift check skipped."
                        ),
                        implication="Calibration drift detection was skipped.",
                    )
                ],
                raw_metrics={"skipped": "no stable-frame action/state pairs"},
            )

        all_offsets = np.concatenate(per_ep_offsets, axis=0)
        n_stable = len(all_offsets)
        if n_stable < self.min_stable_frames:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="joint_offset_max_abs",
                        observed=ObservedValue(value=None),
                        interpretation=(
                            f"Only {n_stable} stable frames found across the dataset "
                            f"(need >= {self.min_stable_frames}) — too few for a reliable "
                            "per-motor offset estimate."
                        ),
                        implication="Calibration drift detection was skipped.",
                    )
                ],
                raw_metrics={"skipped": "too few stable frames", "n_stable_frames": n_stable},
            )

        mean_offset = np.mean(all_offsets, axis=0)
        std_offset = np.std(all_offsets, axis=0)
        all_states = np.concatenate(per_ep_states, axis=0)
        state_range = np.ptp(all_states, axis=0)
        state_range = np.where(state_range < 1e-9, 1.0, state_range)  # guard divide-by-zero

        consistency = np.abs(mean_offset) / np.where(std_offset < 1e-9, 1e-9, std_offset)
        offset_frac = np.abs(mean_offset) / state_range
        is_systematic = consistency >= self.consistency_min

        per_motor = [
            {
                "dim": int(active_dims[j]),
                "mean_offset": float(mean_offset[j]),
                "std_offset": float(std_offset[j]),
                "offset_frac_of_range": float(offset_frac[j]),
                "consistency": float(consistency[j]),
                "systematic": bool(is_systematic[j]),
            }
            for j in range(len(active_dims))
        ]
        raw = {"per_motor_offset": per_motor, "n_stable_frames": n_stable}

        candidate_fracs = offset_frac[is_systematic]
        worst_frac = float(np.max(candidate_fracs)) if candidate_fracs.size else 0.0
        worst_idx = (
            int(np.argmax(np.where(is_systematic, offset_frac, -np.inf)))
            if candidate_fracs.size
            else None
        )

        if worst_idx is None or worst_frac < self.offset_warning_frac:
            flag = RiskFlag(
                level=RiskLevel.OK,
                metric="joint_offset_max_abs",
                observed=ObservedValue(value=worst_frac, unit="fraction of joint range"),
                threshold=self.offset_warning_frac,
                interpretation=(
                    f"No systematic action/state offset detected across {n_stable} "
                    "pooled stationary frames."
                ),
                implication="No calibration drift risk detected.",
            )
        else:
            motor = per_motor[worst_idx]
            flag = RiskFlag(
                level=RiskLevel.WARNING,  # capped: thresholds not yet validated, see module docstring
                metric="joint_offset_max_abs",
                observed=ObservedValue(value=worst_frac, unit="fraction of joint range"),
                threshold=self.offset_warning_frac,
                interpretation=(
                    f"Motor/dim {motor['dim']} shows a consistent action-state offset of "
                    f"{motor['mean_offset']:.4g} ({worst_frac:.1%} of its observed range) "
                    f"across {n_stable} pooled stationary frames, stable across holds "
                    f"(|mean|/std = {motor['consistency']:.1f})."
                ),
                implication=(
                    "A stable per-motor action/state offset during holds — not during "
                    "motion — is the signature of a leader/follower calibration drift "
                    "(see LeRobot issue #3758: an unnoticed joint offset trained fine "
                    "but caused consistent under/overshoot at deployment). This is a "
                    "review signal, not proof of a broken dataset: verify the leader/"
                    "follower zero-offset calibration before relying on this data for "
                    "precision manipulation."
                ),
            )

        return AnalyzerResult(analyzer_name=self.name, flags=[flag], raw_metrics=raw)
