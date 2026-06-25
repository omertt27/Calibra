"""
π0 Compatibility Analyzer.

Checks whether a dataset meets the structural requirements for fine-tuning
Physical Intelligence's π0 — a flow-matching Vision-Language-Action model.

π0 requirements (https://www.physicalintelligence.company/blog/pi0):

  1. Visual observations      — at least one RGB stream per episode.
  2. Language annotations     — task description string in episode metadata.
  3. Action chunking          — episodes must be long enough for one chunk
                                (default CHUNK_SIZE = 50 steps).
  4. Control frequency        — π0 is trained at 50 Hz. Data at 10–100 Hz is
                                usable; outside this range tokenisation may degrade.
  5. Action dimensionality    — typical π0 configs: 7D (single arm),
                                14D (bimanual). Unusual dims are flagged as WARNING
                                so the user can verify the action-space head.
  6. Trajectory smoothness    — π0 uses flow matching; very high jerk (LDLJ < -15)
                                can cause the flow to degenerate.

Only runs when policy_family contains "pi0". All other policy families
receive an empty result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import (
    AnalyzerResult,
    CompatibilityHint,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)

# ── constants ────────────────────────────────────────────────────────────────

_CHUNK_SIZE: int = 50  # π0 default action chunk

_FREQ_LOW_WARNING: float = 10.0  # Hz — below this, temporal tokenisation degrades
_FREQ_HIGH_WARNING: float = 100.0  # Hz — above this, chunks cover < 0.5 s of motion

_KNOWN_ACTION_DIMS: set[int] = {7, 14}

_VISUAL_KEYS = frozenset(["camera", "image", "rgb", "depth", "visual"])

_LDLJ_WARNING: float = -15.0  # π0 flow matching is sensitive to discontinuities


@dataclass
class Pi0CompatibilityAnalyzer(Analyzer):
    """
    Structural compatibility checks for Physical Intelligence π0 fine-tuning.

    Parameters
    ----------
    chunk_size : int
        π0 action chunk size. Default is 50.
    freq_low_warning : float
        Hz below which control frequency is flagged as too slow.
    freq_high_warning : float
        Hz above which control frequency is flagged as too fast.
    known_action_dims : set[int]
        Action dimensions that match standard π0 robot configurations.
    """

    chunk_size: int = _CHUNK_SIZE
    freq_low_warning: float = _FREQ_LOW_WARNING
    freq_high_warning: float = _FREQ_HIGH_WARNING
    known_action_dims: set[int] = field(default_factory=lambda: set(_KNOWN_ACTION_DIMS))

    @property
    def name(self) -> str:
        return "pi0_compatibility"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if not policy_family or "pi0" not in policy_family.lower():
            return AnalyzerResult(analyzer_name=self.name)

        flags: list[RiskFlag] = []
        hints: list[CompatibilityHint] = []
        raw: dict = {}

        # ── 1. Visual observations ────────────────────────────────────────────
        has_visual = any(
            any(vk in obs_key.lower() for vk in _VISUAL_KEYS)
            for ep in batch.episodes
            for obs_key in ep.observations
        )
        raw["has_visual_obs"] = has_visual
        if not has_visual:
            flags.append(
                RiskFlag(
                    level=RiskLevel.CRITICAL,
                    metric="pi0_visual_observations",
                    observed=ObservedValue(value=0.0),
                    interpretation=(
                        "No RGB/visual observation stream detected. "
                        "π0 is a vision-language-action model and requires camera input."
                    ),
                    implication=(
                        "Fine-tuning π0 without visual input is not supported. "
                        "Ensure at least one camera observation key is present per episode."
                    ),
                )
            )

        # ── 2. Language annotations ───────────────────────────────────────────
        n_with_lang = sum(
            1
            for ep in batch.episodes
            if ep.metadata.task_description and ep.metadata.task_description.strip()
        )
        lang_fraction = n_with_lang / max(len(batch.episodes), 1)
        raw["language_annotation_fraction"] = lang_fraction
        if lang_fraction < 0.5:
            flags.append(
                RiskFlag(
                    level=RiskLevel.CRITICAL,
                    metric="pi0_language_annotations",
                    observed=ObservedValue(value=lang_fraction, unit="fraction"),
                    threshold=0.5,
                    interpretation=(
                        f"Only {lang_fraction:.0%} of episodes have language task descriptions. "
                        "π0 requires language conditioning for each demonstration."
                    ),
                    implication=(
                        "Missing language annotations will prevent π0 from learning the "
                        "action→language grounding that enables generalization. "
                        "Add task_description to EpisodeMetadata."
                    ),
                )
            )
        elif lang_fraction < 1.0:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="pi0_language_annotations",
                    observed=ObservedValue(value=lang_fraction, unit="fraction"),
                    threshold=1.0,
                    interpretation=(
                        f"{lang_fraction:.0%} of episodes have language annotations "
                        "(expected 100%)."
                    ),
                    implication=(
                        "Episodes without language descriptions will fall back to an empty "
                        "prompt, weakening the policy's language conditioning."
                    ),
                )
            )

        # ── 3. Episode length vs. chunk size ──────────────────────────────────
        short_eps = [ep for ep in batch.episodes if ep.n_steps < self.chunk_size]
        short_frac = len(short_eps) / max(len(batch.episodes), 1)
        raw["short_episode_fraction_pi0"] = short_frac
        if short_frac > 0.10:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="pi0_episode_length",
                    observed=ObservedValue(value=short_frac, unit="fraction"),
                    threshold=0.10,
                    interpretation=(
                        f"{short_frac:.0%} of episodes are shorter than π0 chunk size "
                        f"({self.chunk_size} steps)."
                    ),
                    implication=(
                        "Short episodes produce incomplete final chunks, forcing padding. "
                        "This may bias the flow-matching trajectory distribution."
                    ),
                    affected_fraction=short_frac,
                )
            )

        # ── 4. Control frequency ──────────────────────────────────────────────
        freqs: list[float] = []
        for ep in batch.episodes:
            if ep.n_steps >= 2:
                diffs = np.diff(ep.timestamps)
                median_dt = float(np.median(diffs[diffs > 0])) if diffs.size else 0.0
                if median_dt > 0:
                    freqs.append(1.0 / median_dt)

        if freqs:
            mean_freq = float(np.mean(freqs))
            raw["mean_control_hz"] = mean_freq
            if mean_freq < self.freq_low_warning:
                flags.append(
                    RiskFlag(
                        level=RiskLevel.WARNING,
                        metric="pi0_control_frequency",
                        observed=ObservedValue(value=mean_freq, unit="Hz"),
                        threshold=self.freq_low_warning,
                        interpretation=(
                            f"Mean control frequency {mean_freq:.1f} Hz is below the "
                            f"recommended {self.freq_low_warning} Hz for π0."
                        ),
                        implication=(
                            "Low-frequency data undersamples fast motions; flow-matching "
                            "may learn coarse trajectories that fail at execution speed."
                        ),
                    )
                )
            elif mean_freq > self.freq_high_warning:
                flags.append(
                    RiskFlag(
                        level=RiskLevel.WARNING,
                        metric="pi0_control_frequency",
                        observed=ObservedValue(value=mean_freq, unit="Hz"),
                        threshold=self.freq_high_warning,
                        interpretation=(
                            f"Mean control frequency {mean_freq:.1f} Hz exceeds "
                            f"{self.freq_high_warning} Hz — chunks cover < 0.5 s of motion."
                        ),
                        implication=(
                            "Very high-frequency data means π0 chunks represent short "
                            "time horizons; consider sub-sampling or reducing chunk size."
                        ),
                    )
                )

        # ── 5. Action dimensionality ──────────────────────────────────────────
        if batch.episodes:
            action_dim = batch.episodes[0].action_dim
            raw["action_dim"] = action_dim
            if action_dim not in self.known_action_dims:
                flags.append(
                    RiskFlag(
                        level=RiskLevel.WARNING,
                        metric="pi0_action_dim",
                        observed=ObservedValue(value=float(action_dim)),
                        interpretation=(
                            f"Action dim {action_dim} is not a standard π0 configuration "
                            f"(known: {sorted(self.known_action_dims)})."
                        ),
                        implication=(
                            "Verify the action-space head in your π0 config matches this dim. "
                            "Mismatched dims cause silent projection errors."
                        ),
                    )
                )

        # ── 6. Trajectory smoothness ──────────────────────────────────────────
        from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer

        smooth_result = ControlSmoothnessAnalyzer().analyze(batch)
        ldlj_raw = smooth_result.raw_metrics.get("ldlj", {}).get("mean_ldlj")
        raw["mean_ldlj"] = ldlj_raw
        if ldlj_raw is not None and ldlj_raw < _LDLJ_WARNING:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="pi0_trajectory_smoothness",
                    observed=ObservedValue(value=ldlj_raw),
                    threshold=_LDLJ_WARNING,
                    interpretation=(
                        f"Mean LDLJ = {ldlj_raw:.2f} (threshold: >{_LDLJ_WARNING}). "
                        "High jerk may cause π0 flow-matching to learn degenerate paths."
                    ),
                    implication=(
                        "Consider applying Savitzky-Golay smoothing or running "
                        "`calibra prune` to remove jerk-spike episodes before fine-tuning."
                    ),
                )
            )

        # ── compatibility hint ────────────────────────────────────────────────
        n_critical = sum(1 for f in flags if f.level == RiskLevel.CRITICAL)
        n_warning = sum(1 for f in flags if f.level == RiskLevel.WARNING)
        if n_critical == 0 and n_warning == 0:
            compatible: Optional[bool] = True
            explanation = "Dataset meets all π0 structural requirements."
        elif n_critical == 0:
            compatible = None
            explanation = (
                f"Dataset is likely compatible with π0 but has {n_warning} warning(s). "
                "Review flagged metrics before fine-tuning."
            )
        else:
            compatible = False
            explanation = f"Dataset has {n_critical} critical issue(s) that block π0 fine-tuning."

        caveats = [
            "π0 fine-tuning is only officially supported via the π0 SDK.",
            "Verify your action head config matches the dataset action dim.",
        ]
        hints.append(
            CompatibilityHint(
                policy_family="pi0",
                compatible=compatible,
                explanation=explanation,
                caveats=caveats,
            )
        )

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )
