"""
Blur Analyzer.

Detects episodes whose camera frames are anomalously blurry relative to the
rest of the dataset. Uses Laplacian variance — the standard, dependency-free
sharpness proxy (a low-variance Laplacian response means little high-frequency
detail, i.e. a blurry or out-of-focus frame; a high-variance response means
sharp edges).

Deliberately an IQR-outlier check, not a fixed absolute threshold: Laplacian
variance's scale depends on camera resolution, exposure, and scene content,
so there's no universal "blurry" cutoff that generalizes across datasets —
the same pitfall `TaskStructureAnalyzer._check_short_episodes` avoids for
episode length by comparing episodes against the dataset's own distribution
rather than a magic number.
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

_BLUR_WARNING = 0.05  # 5% of episodes are blur outliers
_BLUR_CRITICAL = 0.15  # 15%


def compute_laplacian_variance(images: np.ndarray) -> np.ndarray:
    """
    Per-frame Laplacian variance — a standard, dependency-free blur metric.

    Parameters
    ----------
    images : np.ndarray, shape (T, H, W, C) or (T, H, W)

    Returns
    -------
    np.ndarray, shape (T,)
        Variance of the discrete Laplacian response per frame. Lower values
        indicate less high-frequency detail (blurrier frames).
    """
    imgs = np.asarray(images, dtype=np.float32)
    if imgs.ndim not in (3, 4):
        raise ValueError(f"images must be (T, H, W) or (T, H, W, C), got shape {imgs.shape}.")
    gray = imgs.mean(axis=-1) if imgs.ndim == 4 else imgs

    # Discrete Laplacian (kernel [[0,1,0],[1,-4,1],[0,1,0]]) via shifted sums —
    # avoids a scipy/cv2 dependency, matching the rest of the ingestion-agnostic
    # numpy-only convention (see calibra.temporal.drift.compute_visual_activity).
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
        + np.roll(gray, 1, axis=2)
        + np.roll(gray, -1, axis=2)
    )
    lap = lap[:, 1:-1, 1:-1]  # drop the 1px border contaminated by np.roll wraparound
    return lap.reshape(len(gray), -1).var(axis=1)


def _episode_mean_sharpness(ep: Episode) -> Optional[float]:
    images = _find_image_obs(ep)
    if images is None:
        return None
    return float(np.mean(compute_laplacian_variance(images)))


@dataclass
class BlurAnalyzer(Analyzer):
    """
    Flags episodes whose camera frames are blur outliers relative to the
    rest of the dataset (IQR outlier on mean per-episode Laplacian variance).

    Parameters
    ----------
    warning, critical : blurry-episode-fraction thresholds for risk level.
    """

    requires = frozenset({"images"})

    warning: float = _BLUR_WARNING
    critical: float = _BLUR_CRITICAL

    @property
    def name(self) -> str:
        return "blur"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        ep_values: list[Optional[float]] = [_episode_mean_sharpness(ep) for ep in batch.episodes]
        checked = [(ep, v) for ep, v in zip(batch.episodes, ep_values) if v is not None]

        if not checked:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="blurry_episode_fraction",
                        observed=ObservedValue(value=None),
                        interpretation="No decodable camera frames found for this dataset.",
                        implication="Blur detection was skipped.",
                    )
                ],
                raw_metrics={"skipped": "no image observations"},
            )

        if len(checked) < 4:
            return AnalyzerResult(
                analyzer_name=self.name,
                flags=[
                    RiskFlag(
                        level=RiskLevel.INFO,
                        metric="blurry_episode_fraction",
                        observed=ObservedValue(value=None),
                        interpretation="Too few episodes with image data for IQR outlier detection.",
                        implication="Need at least 4 episodes with camera frames.",
                    )
                ],
                raw_metrics={"skipped": "too few episodes with image data"},
            )

        sharpness = np.array([v for _, v in checked])
        q1, q3 = np.percentile(sharpness, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr

        outlier_mask = sharpness < lower_fence
        frac = float(np.mean(outlier_mask))
        outlier_ids = [
            ep.metadata.episode_id
            for (ep, _), is_outlier in zip(checked, outlier_mask)
            if is_outlier
        ]
        raw = {
            "blurry_episode_fraction": frac,
            "lower_fence_laplacian_var": float(lower_fence),
            "q1": float(q1),
            "q3": float(q3),
            "n_episodes_checked": len(checked),
            "outlier_episode_ids": outlier_ids[:20],
        }

        level = _threshold_level_upper(frac, self.warning, self.critical)
        if level == RiskLevel.OK or frac == 0.0:
            flag = RiskFlag(
                level=RiskLevel.OK,
                metric="blurry_episode_fraction",
                observed=ObservedValue(value=frac, unit="fraction"),
                threshold=self.warning,
                interpretation="No episodes are anomalously blurry relative to the rest of the dataset.",
                implication="No blur risk detected.",
            )
        else:
            flag = RiskFlag(
                level=level,
                metric="blurry_episode_fraction",
                observed=ObservedValue(value=frac, unit="fraction"),
                threshold=self.warning,
                interpretation=(
                    f"{frac:.1%} of episodes ({len(outlier_ids)}/{len(checked)}) have "
                    f"camera frames markedly blurrier than the rest of the dataset "
                    f"(mean Laplacian variance below the IQR lower fence). "
                    f"IDs: {outlier_ids[:5]}{'...' if len(outlier_ids) > 5 else ''}"
                ),
                implication=(
                    "Blurry frames (motion blur, defocus, or a dirty/misconfigured lens) "
                    "give the policy a degraded or misleading visual observation for that "
                    "episode. Inspect the flagged episodes before training — this is a "
                    "relative comparison within this dataset, not an absolute sharpness "
                    "standard, so a genuinely blurry whole dataset won't self-flag."
                ),
                affected_fraction=frac,
            )

        return AnalyzerResult(analyzer_name=self.name, flags=[flag], raw_metrics=raw)
