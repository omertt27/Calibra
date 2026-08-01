"""
Camera Freeze Analyzer.

Detects sustained runs of near-identical consecutive camera frames within an
episode — the signature of a camera that stopped updating entirely (dropped
connection, stalled driver, buffer re-emission) rather than a single dropped
grab. This is deliberately a *separate* check from `DuplicateFrameAnalyzer`:
an isolated duplicate frame is common and often benign (a paused robot, a
static scene); a long run of duplicates means the visual stream went stale
for a real stretch of the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.analyzers.duplicate_frame import _find_image_obs
from calibra.analyzers.task_structure import _threshold_level_upper
from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import AnalyzerResult, ObservedValue, RiskFlag, RiskLevel

_FREEZE_ACTIVITY_THRESHOLD = 0.5  # mean abs pixel diff below this = frozen transition
_MIN_FREEZE_RUN = 5  # consecutive frozen transitions to count as a freeze event
_FREEZE_WARNING_FRACTION = 1e-9  # any episode with a freeze event triggers at least WARNING
_FREEZE_CRITICAL_FRACTION = 0.10  # 10% of checked episodes affected


def _max_run_length(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values in `mask`."""
    if mask.size == 0:
        return 0
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _episode_freeze_run(ep: Episode, activity_threshold: float) -> Optional[int]:
    images = _find_image_obs(ep)
    if images is None:
        return None
    from calibra.temporal.drift import compute_visual_activity

    activity = compute_visual_activity(images)
    return _max_run_length(activity < activity_threshold)


@dataclass
class CameraFreezeAnalyzer(Analyzer):
    """
    Detects episodes containing a sustained run of frozen (near-identical)
    consecutive camera frames.

    Parameters
    ----------
    activity_threshold : mean abs pixel difference below which a frame
                          transition counts as frozen. Provisional default —
                          tune against your own camera/exposure settings.
    min_freeze_run      : consecutive frozen transitions required to count
                           as a freeze event (isolated duplicates don't count;
                           see DuplicateFrameAnalyzer for that).
    warning_fraction, critical_fraction : fraction-of-episodes-affected
                          thresholds for risk level.
    """

    requires = frozenset({"images"})

    activity_threshold: float = _FREEZE_ACTIVITY_THRESHOLD
    min_freeze_run: int = _MIN_FREEZE_RUN
    warning_fraction: float = _FREEZE_WARNING_FRACTION
    critical_fraction: float = _FREEZE_CRITICAL_FRACTION

    @property
    def name(self) -> str:
        return "camera_freeze"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        run_lengths: list[Optional[int]] = [
            _episode_freeze_run(ep, self.activity_threshold) for ep in batch.episodes
        ]
        checked = [(ep, r) for ep, r in zip(batch.episodes, run_lengths) if r is not None]

        if not checked:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="camera_freeze_events",
                        observed=ObservedValue(value=None),
                        interpretation="No decodable camera frames found for this dataset.",
                        implication="Camera-freeze detection was skipped.",
                    )
                ],
                raw_metrics={"skipped": "no image observations"},
            )

        frozen = [
            {"episode_id": ep.metadata.episode_id, "run_length": r}
            for ep, r in checked
            if r >= self.min_freeze_run
        ]
        frozen.sort(key=lambda d: d["run_length"], reverse=True)
        affected_fraction = len(frozen) / len(checked)

        raw = {
            "affected_fraction": affected_fraction,
            "n_episodes_checked": len(checked),
            "freeze_episodes": frozen[:20],
        }

        level = _threshold_level_upper(
            affected_fraction, self.warning_fraction, self.critical_fraction
        )
        if level == RiskLevel.OK:
            flag = RiskFlag(
                level=RiskLevel.OK,
                metric="camera_freeze_events",
                observed=ObservedValue(value=affected_fraction, unit="fraction"),
                threshold=self.warning_fraction,
                interpretation="No sustained camera-freeze runs detected.",
                implication="No camera-freeze risk detected.",
                affected_fraction=affected_fraction,
            )
        else:
            worst = frozen[0]
            flag = RiskFlag(
                level=level,
                metric="camera_freeze_events",
                observed=ObservedValue(value=affected_fraction, unit="fraction"),
                threshold=self.warning_fraction,
                interpretation=(
                    f"{len(frozen)} of {len(checked)} episodes ({affected_fraction:.1%}) "
                    f"contain a run of ≥{self.min_freeze_run} consecutive near-identical "
                    f"camera frames (longest: {worst['run_length']} frames in episode "
                    f"{worst['episode_id']}), suggesting the camera stopped updating."
                ),
                implication=(
                    "A frozen camera segment means the policy would be trained on stale "
                    "visual input during that window — likely to cause visually-triggered "
                    "failures when deployed against a live, moving scene."
                ),
                affected_fraction=affected_fraction,
            )

        return AnalyzerResult(analyzer_name=self.name, flags=[flag], raw_metrics=raw)
