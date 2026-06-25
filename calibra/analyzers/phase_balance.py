"""
Phase Balance Analyzer.

Manipulation datasets are systematically biased toward approach trajectories.
Most robot demonstrations spend 80-90% of steps moving to the object and
only 5-15% in the contact/grasp/insert phase — yet the contact phase is where
policies fail most in deployment.

This analyzer quantifies the per-phase time allocation across the dataset:
  - Approach:  steps before the first sustained contact event
  - Contact:   steps during the primary contact/manipulation phase
  - Retract:   steps after the last sustained contact event

Low contact-phase fraction (<10%) is flagged because:
  1. The policy sees far fewer examples of the critical manipulation phase.
  2. BC loss is dominated by approach-phase gradients; contact-phase
     gradients are diluted and the policy learns approach well but
     contact/insert poorly.
  3. This is a *structural* bias in the dataset, not a quality defect —
     it cannot be fixed by collecting more episodes of the same type.
     It requires deliberate augmentation with contact-phase demonstrations.

Claim backed: PB-001 — "Manipulation datasets typically allocate <15% of
steps to the contact phase, regardless of gripper hardware or task type."

Metric
------
Per episode, the contact mask (from gripper state + velocity envelope) is
smoothed and split into phase regions using the first and last sustained
contact block. Dataset-level phase fractions are bootstrapped over episodes.

Minimum sustained contact: a contact block must span >= min_contact_run steps
to count as the start/end of the contact zone. This prevents brief spurious
gripper activations from distorting phase boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.analyzers.task_structure import (
    _detect_gripper_dims,
)
from calibra.analyzers.temporal import _bootstrap_ci
from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import (
    AnalyzerResult,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)

# ── thresholds ────────────────────────────────────────────────────────────────

_CONTACT_WARNING = 0.10  # <10% of steps in contact → WARNING
_CONTACT_CRITICAL = 0.05  # <5%  → CRITICAL
_APPROACH_HEAVY = 0.80  # >80% approach → INFO (structural note)

_MIN_CONTACT_RUN = 3  # minimum consecutive contact steps to mark phase boundary


@dataclass
class PhaseBalanceAnalyzer(Analyzer):
    """
    Quantifies approach / contact / retract phase balance across a dataset.

    Parameters
    ----------
    gripper_dims : explicit gripper action column indices. None = auto-detect.
    vel_slow_threshold : speed fraction below which a step is counted as contact.
    action_type : "position" | "velocity".
    min_contact_run : minimum consecutive contact steps to define a phase boundary.
    contact_warning, contact_critical : fraction thresholds for contact phase.
    n_bootstrap, ci_level : bootstrap CI parameters.
    """

    gripper_dims: Optional[list[int]] = None
    vel_slow_threshold: float = 0.08
    action_type: str = "position"
    min_contact_run: int = _MIN_CONTACT_RUN
    contact_warning: float = _CONTACT_WARNING
    contact_critical: float = _CONTACT_CRITICAL
    n_bootstrap: int = 500
    ci_level: float = 0.95

    @property
    def name(self) -> str:
        return "phase_balance"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        from calibra.analyzers.task_structure import _collect_actions

        actions_all = _collect_actions(batch)
        g_dims = (
            self.gripper_dims
            if self.gripper_dims is not None
            else _detect_gripper_dims(actions_all)
        )

        episode_phases = [
            _episode_phase_fractions(
                ep, g_dims, self.vel_slow_threshold, self.action_type, self.min_contact_run
            )
            for ep in batch.episodes
        ]

        approach_vals = np.array([p["approach"] for p in episode_phases])
        contact_vals = np.array([p["contact"] for p in episode_phases])
        retract_vals = np.array([p["retract"] for p in episode_phases])

        contact_mean, c_lo, c_hi = _bootstrap_ci(
            contact_vals, np.mean, self.n_bootstrap, self.ci_level
        )
        approach_mean = float(np.mean(approach_vals))
        retract_mean = float(np.mean(retract_vals))

        raw: dict = {
            "mean_approach_fraction": float(approach_mean),
            "mean_contact_fraction": float(contact_mean),
            "mean_retract_fraction": float(retract_mean),
            "contact_ci_lower": float(c_lo),
            "contact_ci_upper": float(c_hi),
            "n_episodes": batch.n_episodes,
            "gripper_dims_used": g_dims,
            "per_episode_approach": approach_vals.tolist(),
            "per_episode_contact": contact_vals.tolist(),
            "per_episode_retract": retract_vals.tolist(),
        }

        contact_flag = self._check_contact_phase(contact_mean, c_lo, c_hi, approach_mean)
        approach_flag = self._check_approach_dominance(approach_mean)

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=[contact_flag, approach_flag],
            raw_metrics=raw,
        )

    # ── flags ─────────────────────────────────────────────────────────────────

    def _check_contact_phase(
        self,
        contact_mean: float,
        c_lo: float,
        c_hi: float,
        approach_mean: float,
    ) -> RiskFlag:
        level = _level_lower(contact_mean, self.contact_warning, self.contact_critical)

        observed = ObservedValue(
            value=contact_mean,
            unit="fraction",
            ci_lower=c_lo,
            ci_upper=c_hi,
            ci_level=self.ci_level,
            ci_method="bootstrap",
        )

        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="contact_phase_fraction",
                observed=observed,
                threshold=self.contact_warning,
                interpretation=(
                    f"Contact phase is {contact_mean:.1%} of total steps — "
                    "adequate representation of the manipulation phase."
                ),
                implication="Policy will see sufficient contact-phase examples for generalisation.",
            )

        return RiskFlag(
            level=level,
            metric="contact_phase_fraction",
            observed=observed,
            threshold=self.contact_warning,
            interpretation=(
                f"Contact phase is only {contact_mean:.1%} of total steps "
                f"(threshold: >{self.contact_warning:.0%}). "
                f"Approach dominates at {approach_mean:.1%}."
            ),
            implication=(
                "The policy's BC loss is dominated by approach-phase gradients. "
                "Contact and grasp steps are underrepresented — the policy will "
                "learn to approach reliably but fail at the critical manipulation "
                "phase. Fix: augment with contact-phase-only demonstrations, or "
                "apply per-phase loss weighting (upweight steps where gripper is "
                "closed or velocity is near zero)."
            ),
        )

    def _check_approach_dominance(self, approach_mean: float) -> RiskFlag:
        if approach_mean > _APPROACH_HEAVY:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="approach_phase_dominance",
                observed=ObservedValue(value=approach_mean, unit="fraction"),
                threshold=_APPROACH_HEAVY,
                interpretation=(
                    f"Approach phase accounts for {approach_mean:.1%} of all steps. "
                    "Dataset is approach-heavy."
                ),
                implication=(
                    "Approach-heavy datasets are common in table-top manipulation "
                    "with sparse task-completion criteria. If task success depends "
                    "on contact precision (insertion, in-hand manipulation), the "
                    "approach dominance creates a distribution mismatch at deployment. "
                    "Claim PB-001: this pattern appears in ≥80% of public manipulation datasets."
                ),
            )

        return RiskFlag(
            level=RiskLevel.OK,
            metric="approach_phase_dominance",
            observed=ObservedValue(value=approach_mean, unit="fraction"),
            interpretation=(f"Phase distribution is balanced: approach {approach_mean:.1%}."),
            implication="No approach-dominance bias detected.",
        )


# ── per-episode phase segmentation ───────────────────────────────────────────


def _episode_phase_fractions(
    ep: Episode,
    gripper_dims: list[int],
    vel_slow_threshold: float,
    action_type: str,
    min_contact_run: int,
) -> dict[str, float]:
    """
    Segment a single episode into approach / contact / retract phases.

    Returns dict with keys "approach", "contact", "retract" (fractions summing to 1).
    """
    n = ep.n_steps
    if n < 2:
        return {"approach": 1.0, "contact": 0.0, "retract": 0.0}

    contact_mask = _compute_contact_mask(ep, gripper_dims, vel_slow_threshold, action_type)
    smoothed = _smooth_mask(contact_mask, window=min_contact_run)

    first_contact = _first_sustained_run(smoothed, min_contact_run)
    last_contact = _last_sustained_run(smoothed, min_contact_run)

    if first_contact is None or last_contact is None:
        # No sustained contact — all approach
        return {"approach": 1.0, "contact": 0.0, "retract": 0.0}

    approach_steps = first_contact
    contact_steps = last_contact - first_contact + 1
    retract_steps = n - last_contact - 1

    total = float(n)
    return {
        "approach": approach_steps / total,
        "contact": contact_steps / total,
        "retract": retract_steps / total,
    }


def _compute_contact_mask(
    ep: Episode,
    gripper_dims: list[int],
    vel_slow_threshold: float,
    action_type: str,
) -> np.ndarray:
    """Return boolean array (n_steps,) where True = contact/slow step."""
    n = ep.n_steps
    acts = ep.actions if ep.actions.ndim > 1 else ep.actions[:, np.newaxis]
    contact = np.zeros(n, dtype=bool)

    for gd in gripper_dims:
        if gd >= acts.shape[1]:
            continue
        col = acts[:, gd].astype(np.float64)
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-8:
            continue
        col_norm = (col - lo) / (hi - lo)
        frac_low = float(np.mean(col_norm < 0.3))
        frac_high = float(np.mean(col_norm > 0.7))
        contact |= col_norm > 0.5 if frac_high >= frac_low else col_norm < 0.5

    active = [d for d in range(acts.shape[1]) if d not in gripper_dims]
    if active and action_type == "position" and len(ep.timestamps) >= 2:
        sub = acts[:, active].astype(np.float64)
        dt = float(np.median(np.diff(ep.timestamps)))
        if dt > 0 and len(sub) >= 2:
            vel = np.diff(sub, axis=0) / dt
            speed = np.linalg.norm(vel, axis=-1)
            v_max = float(np.max(speed))
            if v_max > 0:
                slow = speed < vel_slow_threshold * v_max
                contact[1:] |= slow

    return contact


def _smooth_mask(mask: np.ndarray, window: int) -> np.ndarray:
    """Majority-vote smoothing over a sliding window."""
    if window <= 1 or len(mask) < window:
        return mask.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(mask.astype(np.float64), kernel, mode="same")
    return smoothed >= 0.5


def _first_sustained_run(mask: np.ndarray, min_run: int) -> Optional[int]:
    """Return index of first True value that starts a run of >= min_run."""
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            run_end = i
            while run_end < n and mask[run_end]:
                run_end += 1
            if run_end - i >= min_run:
                return i
            i = run_end
        else:
            i += 1
    return None


def _last_sustained_run(mask: np.ndarray, min_run: int) -> Optional[int]:
    """Return index of the last True value in the last run of >= min_run."""
    n = len(mask)
    i = n - 1
    while i >= 0:
        if mask[i]:
            run_start = i
            while run_start >= 0 and mask[run_start]:
                run_start -= 1
            run_start += 1
            if i - run_start + 1 >= min_run:
                return i
            i = run_start - 1
        else:
            i -= 1
    return None


# ── helpers ───────────────────────────────────────────────────────────────────


def _level_lower(value: float, warning: float, critical: float) -> RiskLevel:
    if value <= critical:
        return RiskLevel.CRITICAL
    if value <= warning:
        return RiskLevel.WARNING
    return RiskLevel.OK
