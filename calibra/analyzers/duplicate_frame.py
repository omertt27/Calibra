"""
Duplicate Frame Analyzer.

Detects camera frames that are near-identical to the frame immediately
preceding them — a signal that the capture pipeline logged the same image
twice (dropped grab, buffered frame re-emitted, or a stalled sensor driver)
rather than a genuinely new observation.

Reuses `calibra.temporal.drift.compute_visual_activity` (mean absolute
pixel difference between consecutive frames) as the detection primitive;
a transition with near-zero activity is a duplicate. This is a single-frame
signal — a *sustained run* of duplicates is the separate, more severe
`CameraFreezeAnalyzer` finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.analyzers.task_structure import _threshold_level_upper
from calibra.analyzers.temporal import _bootstrap_ci
from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import AnalyzerResult, ObservedValue, RiskFlag, RiskLevel

_VISUAL_KEYS = frozenset(["camera", "image", "rgb", "depth", "visual"])

_DUPLICATE_ACTIVITY_THRESHOLD = 0.5  # mean abs pixel diff below this = duplicate transition
_DUPLICATE_WARNING = 0.05  # 5% of transitions duplicated
_DUPLICATE_CRITICAL = 0.15  # 15%


def _find_image_obs(ep: Episode) -> Optional[np.ndarray]:
    for key in ep.observations:
        if any(kw in key.lower() for kw in _VISUAL_KEYS):
            candidate = ep.observations[key]
            if candidate.ndim in (3, 4) and len(candidate) >= 2:  # (T, H, W) or (T, H, W, C)
                return candidate
    return None


def _episode_duplicate_fraction(ep: Episode, activity_threshold: float) -> Optional[float]:
    images = _find_image_obs(ep)
    if images is None:
        return None
    from calibra.temporal.drift import compute_visual_activity

    activity = compute_visual_activity(images)
    return float(np.mean(activity < activity_threshold))


@dataclass
class DuplicateFrameAnalyzer(Analyzer):
    """
    Detects camera frames that are near-identical repeats of the previous frame.

    Parameters
    ----------
    activity_threshold : mean abs pixel difference below which a frame
                          transition counts as a duplicate. Provisional
                          default — tune against your own camera/exposure
                          settings if this over- or under-fires.
    warning, critical   : duplicate-frame-rate thresholds for risk level.
    n_bootstrap, ci_level : bootstrap CI parameters, matching TemporalAnalyzer.
    """

    requires = frozenset({"images"})

    activity_threshold: float = _DUPLICATE_ACTIVITY_THRESHOLD
    warning: float = _DUPLICATE_WARNING
    critical: float = _DUPLICATE_CRITICAL
    n_bootstrap: int = 1000
    ci_level: float = 0.95

    @property
    def name(self) -> str:
        return "duplicate_frame"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        ep_values: list[Optional[float]] = [
            _episode_duplicate_fraction(ep, self.activity_threshold) for ep in batch.episodes
        ]
        checked = [(ep, v) for ep, v in zip(batch.episodes, ep_values) if v is not None]

        if not checked:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="duplicate_frame_rate",
                        observed=ObservedValue(value=None),
                        interpretation="No decodable camera frames found for this dataset.",
                        implication="Duplicate-frame detection was skipped.",
                    )
                ],
                raw_metrics={"skipped": "no image observations", "episode_values": ep_values},
            )

        arr = np.array([v for _, v in checked])
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        outlier_ids = [
            ep.metadata.episode_id for ep, v in checked if v is not None and v >= self.warning
        ]
        raw = {
            "duplicate_frame_rate": float(stat),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "n_episodes_checked": len(checked),
            "episode_values": ep_values,
            "outlier_episode_ids": outlier_ids[:20],
        }

        level = _threshold_level_upper(stat, self.warning, self.critical)
        if level == RiskLevel.OK:
            flag = RiskFlag(
                level=RiskLevel.OK,
                metric="duplicate_frame_rate",
                observed=ObservedValue(
                    value=stat,
                    unit="fraction",
                    ci_lower=lo,
                    ci_upper=hi,
                    ci_level=self.ci_level,
                    ci_method="bootstrap",
                ),
                threshold=self.warning,
                interpretation="Camera frames show expected frame-to-frame variation.",
                implication="No duplicate-frame risk detected.",
                affected_fraction=float(stat),
            )
        else:
            flag = RiskFlag(
                level=level,
                metric="duplicate_frame_rate",
                observed=ObservedValue(
                    value=stat,
                    unit="fraction",
                    ci_lower=lo,
                    ci_upper=hi,
                    ci_level=self.ci_level,
                    ci_method="bootstrap",
                ),
                threshold=self.warning,
                interpretation=(
                    f"{stat:.1%} of camera frame transitions are near-identical to the "
                    f"previous frame, across {len(checked)} episodes with image data."
                ),
                implication=(
                    "Duplicate frames mean the camera pipeline is not capturing a new "
                    "image every control step. Policies trained on repeated frames may "
                    "learn to associate a stale visual observation with the wrong action, "
                    "or waste model capacity encoding redundant frames."
                ),
                affected_fraction=float(stat),
            )

        return AnalyzerResult(analyzer_name=self.name, flags=[flag], raw_metrics=raw)
