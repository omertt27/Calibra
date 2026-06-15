"""
GR00T Compatibility Analyzer.

Checks whether a dataset meets the structural requirements for fine-tuning
NVIDIA GR00T N1 — a vision-language-action foundation model for humanoid
and manipulation robots.

GR00T N1 requirements (https://developer.nvidia.com/isaac/gr00t):

  1. Visual observations        — at least one RGB camera stream per episode.
  2. Language annotations       — task instruction strings in episode metadata.
  3. Action chunk support       — episodes must be long enough to fill one chunk
                                  (default: CHUNK_SIZE = 16 steps). Very short
                                  episodes produce incomplete trailing chunks.
  4. Control frequency          — GR00T expects 15–120 Hz. Very slow or very
                                  fast data may not match its tokenisation regime.
  5. Action dimensionality      — common GR00T configs are 7D (single arm),
                                  8D (arm + gripper), 14D (bimanual), 16D
                                  (bimanual + gripper). Unusual dims are flagged
                                  as a warning so the user can verify the action
                                  space matches their robot's GR00T config.

Only runs when policy_family contains "gr00t". All other policy families
receive an empty result so there is no performance overhead.
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

_CHUNK_SIZE: int = 16          # GR00T N1 default action chunk size

_FREQ_LOW_WARNING:  float = 15.0   # Hz — below this, temporal tokenisation degrades
_FREQ_HIGH_WARNING: float = 120.0  # Hz — above this, chunks cover < 0.13 s of motion

_KNOWN_ACTION_DIMS: set[int] = {7, 8, 14, 16}  # documented GR00T robot configs

_VISUAL_KEYS = frozenset(["camera", "image", "rgb", "depth", "visual"])


@dataclass
class GR00TCompatibilityAnalyzer(Analyzer):
    """
    Structural compatibility checks for NVIDIA GR00T N1 fine-tuning.

    Parameters
    ----------
    chunk_size : int
        GR00T action chunk size. Default is 16 (GR00T N1 shipped default).
    freq_low_warning : float
        Hz below which control frequency is flagged as too slow.
    freq_high_warning : float
        Hz above which control frequency is flagged as too fast.
    known_action_dims : set[int]
        Action dimensions that are known GR00T robot configurations.
    """

    chunk_size:          int       = _CHUNK_SIZE
    freq_low_warning:    float     = _FREQ_LOW_WARNING
    freq_high_warning:   float     = _FREQ_HIGH_WARNING
    known_action_dims:   set[int]  = field(default_factory=lambda: set(_KNOWN_ACTION_DIMS))

    @property
    def name(self) -> str:
        return "gr00t_compatibility"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if not policy_family or "gr00t" not in policy_family.lower():
            return AnalyzerResult(analyzer_name=self.name)

        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        flags: list[RiskFlag] = []
        hints: list[CompatibilityHint] = []
        raw: dict = {}

        flags.append(self._check_visual_modality(batch))
        flags.append(self._check_language_annotations(batch))
        flags.append(self._check_episode_length(batch))

        freq_flag, freq_raw = self._check_control_frequency(batch)
        flags.append(freq_flag)
        raw["control_frequency"] = freq_raw

        dim_flag, dim_raw = self._check_action_dim(batch)
        flags.append(dim_flag)
        raw["action_dim"] = dim_raw

        hint = self._overall_hint(flags)
        hints.append(hint)

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )

    # ── checks ───────────────────────────────────────────────────────────────

    def _check_visual_modality(self, batch: EpisodeBatch) -> RiskFlag:
        """GR00T requires at least one RGB camera stream."""
        has_visual = any(
            any(kw in mod_key.lower() for kw in _VISUAL_KEYS)
            for mod_key in batch.modalities
        )
        if has_visual:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="gr00t.visual_modality",
                observed=ObservedValue(value=1.0),
                interpretation="At least one visual modality detected.",
                implication="Dataset includes camera observations — GR00T visual encoder can be used.",
            )
        return RiskFlag(
            level=RiskLevel.CRITICAL,
            metric="gr00t.visual_modality",
            observed=ObservedValue(value=0.0),
            interpretation=(
                f"No visual observations found. Modalities present: "
                f"{sorted(batch.modalities) or ['none']}."
            ),
            implication=(
                "GR00T N1 is a vision-language-action model and requires at "
                "least one RGB camera stream (e.g. wrist cam or agentview). "
                "Add 'agentview_image' or 'robot0_eye_in_hand_image' to your "
                "data collection pipeline before fine-tuning."
            ),
        )

    def _check_language_annotations(self, batch: EpisodeBatch) -> RiskFlag:
        """GR00T is language-conditioned — all episodes should have a task string."""
        n_annotated = sum(
            1 for ep in batch.episodes
            if ep.metadata.task_description and ep.metadata.task_description.strip()
        )
        frac = n_annotated / batch.n_episodes

        if frac >= 0.99:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="gr00t.language_annotations",
                observed=ObservedValue(value=frac, unit="fraction annotated"),
                interpretation=f"{n_annotated}/{batch.n_episodes} episodes have task descriptions.",
                implication="Language conditioning input is available for all episodes.",
            )

        level = RiskLevel.CRITICAL if frac < 0.50 else RiskLevel.WARNING
        return RiskFlag(
            level=level,
            metric="gr00t.language_annotations",
            observed=ObservedValue(value=frac, unit="fraction annotated"),
            interpretation=(
                f"Only {n_annotated}/{batch.n_episodes} episodes "
                f"({frac:.0%}) have a task_description string."
            ),
            implication=(
                "GR00T N1 is conditioned on natural-language task instructions. "
                "Episodes without a task description will receive a null/empty "
                "language token, which degrades policy generalisation. "
                "Add a 'task' or 'language_instruction' field to every episode "
                "in your data collection pipeline."
            ),
        )

    def _check_episode_length(self, batch: EpisodeBatch) -> RiskFlag:
        """Episodes shorter than chunk_size produce incomplete trailing chunks."""
        lengths = [ep.n_steps for ep in batch.episodes]
        median_len = float(np.median(lengths))
        short_count = sum(1 for l in lengths if l < self.chunk_size)
        short_frac = short_count / batch.n_episodes

        if short_frac == 0.0:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="gr00t.episode_length",
                observed=ObservedValue(value=median_len, unit="steps"),
                interpretation=(
                    f"All episodes are ≥ {self.chunk_size} steps "
                    f"(median: {median_len:.0f} steps)."
                ),
                implication=f"Episodes fill at least one complete GR00T action chunk ({self.chunk_size} steps).",
            )

        level = RiskLevel.CRITICAL if short_frac > 0.20 else RiskLevel.WARNING
        return RiskFlag(
            level=level,
            metric="gr00t.episode_length",
            observed=ObservedValue(value=median_len, unit="steps"),
            threshold=float(self.chunk_size),
            interpretation=(
                f"{short_count}/{batch.n_episodes} episodes ({short_frac:.0%}) "
                f"are shorter than GR00T's chunk size ({self.chunk_size} steps). "
                f"Median episode length: {median_len:.0f} steps."
            ),
            implication=(
                f"Episodes shorter than chunk_size={self.chunk_size} produce "
                "incomplete trailing chunks during GR00T training. "
                "Either collect longer episodes or reduce chunk_size in the "
                "GR00T training config to match your episode length."
            ),
            affected_fraction=short_frac,
        )

    def _check_control_frequency(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        """Estimate per-episode control frequency from timestamps."""
        freqs: list[float] = []
        for ep in batch.episodes:
            if ep.n_steps < 2:
                continue
            dt = float(np.median(np.diff(ep.timestamps)))
            if dt > 0:
                freqs.append(1.0 / dt)

        if not freqs:
            flag = RiskFlag(
                level=RiskLevel.INFO,
                metric="gr00t.control_frequency",
                observed=ObservedValue(value=None),
                interpretation="Could not estimate control frequency (too few timesteps).",
                implication="Verify control frequency manually before GR00T fine-tuning.",
            )
            return flag, {}

        median_freq = float(np.median(freqs))
        raw = {"median_hz": median_freq, "n_episodes": len(freqs)}

        if self.freq_low_warning <= median_freq <= self.freq_high_warning:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="gr00t.control_frequency",
                observed=ObservedValue(value=median_freq, unit="Hz"),
                interpretation=f"Control frequency {median_freq:.1f} Hz is within GR00T's expected range.",
                implication=f"Temporal tokenisation should work correctly at {median_freq:.1f} Hz.",
            ), raw

        if median_freq < self.freq_low_warning:
            return RiskFlag(
                level=RiskLevel.WARNING,
                metric="gr00t.control_frequency",
                observed=ObservedValue(value=median_freq, unit="Hz"),
                threshold=self.freq_low_warning,
                interpretation=(
                    f"Control frequency {median_freq:.1f} Hz is below "
                    f"GR00T's expected minimum ({self.freq_low_warning:.0f} Hz)."
                ),
                implication=(
                    "GR00T N1 was pre-trained on data at 15–120 Hz. "
                    f"At {median_freq:.1f} Hz each action chunk spans "
                    f"{self.chunk_size / median_freq:.1f} s, which may exceed "
                    "GR00T's temporal receptive field. Consider resampling to "
                    f"≥ {self.freq_low_warning:.0f} Hz or reducing chunk_size."
                ),
            ), raw

        return RiskFlag(
            level=RiskLevel.WARNING,
            metric="gr00t.control_frequency",
            observed=ObservedValue(value=median_freq, unit="Hz"),
            threshold=self.freq_high_warning,
            interpretation=(
                f"Control frequency {median_freq:.1f} Hz exceeds "
                f"GR00T's expected maximum ({self.freq_high_warning:.0f} Hz)."
            ),
            implication=(
                f"At {median_freq:.1f} Hz each GR00T action chunk covers only "
                f"{self.chunk_size / median_freq * 1000:.0f} ms of motion. "
                "The model may struggle to learn meaningful multi-step plans. "
                "Consider downsampling or increasing chunk_size."
            ),
        ), raw

    def _check_action_dim(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        """Flag unusual action dimensions that don't match documented GR00T configs."""
        dims = [ep.action_dim for ep in batch.episodes if ep.actions.ndim > 1]
        if not dims:
            flag = RiskFlag(
                level=RiskLevel.INFO,
                metric="gr00t.action_dim",
                observed=ObservedValue(value=None),
                interpretation="Could not determine action dimensionality.",
                implication="Verify action space matches your GR00T robot config.",
            )
            return flag, {}

        unique_dims = set(dims)
        modal_dim = int(np.bincount(dims).argmax())
        raw = {"modal_dim": modal_dim, "unique_dims": sorted(unique_dims)}

        if modal_dim in self.known_action_dims:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="gr00t.action_dim",
                observed=ObservedValue(value=float(modal_dim), unit="dims"),
                interpretation=(
                    f"Action dimensionality {modal_dim}D matches a known "
                    "GR00T robot configuration."
                ),
                implication=(
                    f"{modal_dim}D matches: "
                    + {7: "single-arm (no gripper)", 8: "single-arm + gripper",
                       14: "bimanual (no gripper)", 16: "bimanual + gripper"}.get(modal_dim, "")
                ),
            ), raw

        known_str = ", ".join(f"{d}D" for d in sorted(self.known_action_dims))
        return RiskFlag(
            level=RiskLevel.WARNING,
            metric="gr00t.action_dim",
            observed=ObservedValue(value=float(modal_dim), unit="dims"),
            interpretation=(
                f"Action dimensionality {modal_dim}D does not match any standard "
                f"GR00T robot configuration (known: {known_str})."
            ),
            implication=(
                f"Verify that your GR00T robot config's action_head output_dim "
                f"is set to {modal_dim}. If you are using a custom embodiment, "
                "this is expected — dismiss this warning after confirming the config."
            ),
        ), raw

    # ── overall hint ─────────────────────────────────────────────────────────

    def _overall_hint(self, flags: list[RiskFlag]) -> CompatibilityHint:
        has_critical = any(f.level == RiskLevel.CRITICAL for f in flags)
        has_warning  = any(f.level == RiskLevel.WARNING  for f in flags)

        if has_critical:
            return CompatibilityHint(
                policy_family="GR00T N1",
                compatible=False,
                explanation=(
                    "Dataset has critical structural issues that must be resolved "
                    "before GR00T N1 fine-tuning."
                ),
                caveats=[
                    f.interpretation
                    for f in flags if f.level == RiskLevel.CRITICAL
                ],
            )
        if has_warning:
            return CompatibilityHint(
                policy_family="GR00T N1",
                compatible=None,
                explanation=(
                    "Dataset is structurally compatible with GR00T N1 but has "
                    "warnings that may reduce fine-tuning quality."
                ),
                caveats=[
                    f.interpretation
                    for f in flags if f.level == RiskLevel.WARNING
                ],
            )
        return CompatibilityHint(
            policy_family="GR00T N1",
            compatible=True,
            explanation=(
                "Dataset passes all GR00T N1 structural compatibility checks. "
                "Ready for fine-tuning."
            ),
        )