"""
Octo Compatibility Analyzer.

Checks whether a dataset meets the structural requirements for fine-tuning
UC Berkeley's Octo — a generalist transformer-based robot policy.

Octo requirements (https://octo-models.github.io):

  1. Visual observations      — at least one camera stream per episode
                                (Octo processes one or two image views).
  2. Language annotations     — task description or goal image per episode.
                                Language AND/OR goal image goal conditioning.
  3. Episode length           — Octo uses context windows; very short episodes
                                (< 4 steps, i.e. < 1 context window at window_size=4)
                                produce no valid training samples.
  4. Control frequency        — Octo was pre-trained at 10–50 Hz. Data below 5 Hz
                                may cause sparse coverage; above 100 Hz creates
                                context windows that span < 40 ms of motion.
  5. Action dimensionality    — Octo supports any continuous action dim, but
                                pre-trained checkpoints use 7D (single arm) or
                                14D (bimanual). Other dims require head re-init.
  6. Dataset size             — Octo fine-tuning converges reliably with ≥ 50 episodes.
                                Smaller datasets may overfit on the residual head.

Only runs when policy_family contains "octo". All other policy families
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

_WINDOW_SIZE:         int   = 4      # Octo default context window
_MIN_EPISODES:        int   = 50     # below this, fine-tuning may overfit
_FREQ_LOW_WARNING:    float = 5.0    # Hz
_FREQ_HIGH_WARNING:   float = 100.0  # Hz
_KNOWN_ACTION_DIMS:   set[int] = {7, 14}
_VISUAL_KEYS = frozenset(["camera", "image", "rgb", "depth", "visual"])


@dataclass
class OctoCompatibilityAnalyzer(Analyzer):
    """
    Structural compatibility checks for Octo fine-tuning.

    Parameters
    ----------
    window_size : int
        Octo observation context window. Default is 4.
    min_episodes : int
        Minimum episode count for reliable fine-tuning.
    freq_low_warning : float
        Hz below which control frequency is flagged as too slow.
    freq_high_warning : float
        Hz above which control frequency is flagged as too fast.
    known_action_dims : set[int]
        Action dimensions that match standard Octo pre-trained checkpoints.
    """

    window_size:        int       = _WINDOW_SIZE
    min_episodes:       int       = _MIN_EPISODES
    freq_low_warning:   float     = _FREQ_LOW_WARNING
    freq_high_warning:  float     = _FREQ_HIGH_WARNING
    known_action_dims:  set[int]  = field(default_factory=lambda: set(_KNOWN_ACTION_DIMS))

    @property
    def name(self) -> str:
        return "octo_compatibility"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if not policy_family or "octo" not in policy_family.lower():
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
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL,
                metric="octo_visual_observations",
                observed=ObservedValue(value=0.0),
                interpretation=(
                    "No RGB/visual observation stream detected. "
                    "Octo requires at least one camera input per step."
                ),
                implication=(
                    "Octo's image tokeniser has no input to process. "
                    "Add at least one camera observation key per episode."
                ),
            ))
        else:
            visual_keys: set[str] = set()
            for ep in batch.episodes[:5]:
                for k in ep.observations:
                    if any(vk in k.lower() for vk in _VISUAL_KEYS):
                        visual_keys.add(k)
            raw["n_visual_streams"] = len(visual_keys)
            if len(visual_keys) > 2:
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="octo_camera_count",
                    observed=ObservedValue(value=float(len(visual_keys))),
                    interpretation=(
                        f"{len(visual_keys)} visual streams detected. "
                        "Standard Octo checkpoints use 1–2 cameras."
                    ),
                    implication=(
                        "Using more than 2 cameras requires Octo architecture changes. "
                        "Extra camera streams will be ignored unless you configure "
                        "additional image tokenisers."
                    ),
                ))

        # ── 2. Language / goal conditioning ──────────────────────────────────
        n_with_lang = sum(
            1 for ep in batch.episodes
            if ep.metadata.task_description and ep.metadata.task_description.strip()
        )
        lang_fraction = n_with_lang / max(len(batch.episodes), 1)
        raw["language_annotation_fraction"] = lang_fraction
        if lang_fraction == 0.0:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                metric="octo_task_conditioning",
                observed=ObservedValue(value=lang_fraction, unit="fraction"),
                interpretation=(
                    "No language annotations found. Octo supports language OR "
                    "goal-image conditioning, but neither is detected."
                ),
                implication=(
                    "Without a task specification, Octo will use an empty language "
                    "token, which severely degrades generalisation. Provide language "
                    "descriptions or goal images."
                ),
            ))
        elif lang_fraction < 1.0:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                metric="octo_language_annotations",
                observed=ObservedValue(value=lang_fraction, unit="fraction"),
                threshold=1.0,
                interpretation=(
                    f"{lang_fraction:.0%} of episodes have language task descriptions."
                ),
                implication=(
                    "Inconsistent language conditioning can cause unstable fine-tuning. "
                    "Aim for 100% language coverage."
                ),
            ))

        # ── 3. Episode length vs. context window ──────────────────────────────
        min_steps = self.window_size + 1
        short_eps = [ep for ep in batch.episodes if ep.n_steps < min_steps]
        short_frac = len(short_eps) / max(len(batch.episodes), 1)
        raw["context_window_short_fraction"] = short_frac
        if short_frac > 0.0:
            flags.append(RiskFlag(
                level=RiskLevel.CRITICAL if short_frac > 0.5 else RiskLevel.WARNING,
                metric="octo_episode_length",
                observed=ObservedValue(value=short_frac, unit="fraction"),
                threshold=0.0,
                interpretation=(
                    f"{short_frac:.0%} of episodes have fewer than {min_steps} steps "
                    f"(Octo window_size={self.window_size} requires ≥{min_steps} steps)."
                ),
                implication=(
                    "Episodes shorter than window_size+1 produce zero valid training "
                    "samples and must be excluded."
                ),
                affected_fraction=short_frac,
            ))

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
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="octo_control_frequency",
                    observed=ObservedValue(value=mean_freq, unit="Hz"),
                    threshold=self.freq_low_warning,
                    interpretation=(
                        f"Mean control frequency {mean_freq:.1f} Hz is below the "
                        f"recommended {self.freq_low_warning} Hz for Octo."
                    ),
                    implication=(
                        "Sparse observations may cause Octo's temporal attention "
                        "to interpolate across large motion gaps."
                    ),
                ))
            elif mean_freq > self.freq_high_warning:
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="octo_control_frequency",
                    observed=ObservedValue(value=mean_freq, unit="Hz"),
                    threshold=self.freq_high_warning,
                    interpretation=(
                        f"Mean control frequency {mean_freq:.1f} Hz is high; "
                        "each context window covers < 40 ms of motion."
                    ),
                    implication=(
                        "Very high-frequency data shortens the temporal horizon "
                        "captured by each context window. Consider sub-sampling."
                    ),
                ))

        # ── 5. Action dimensionality ──────────────────────────────────────────
        if batch.episodes:
            action_dim = batch.episodes[0].action_dim
            raw["action_dim"] = action_dim
            if action_dim not in self.known_action_dims:
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="octo_action_dim",
                    observed=ObservedValue(value=float(action_dim)),
                    interpretation=(
                        f"Action dim {action_dim} differs from standard Octo "
                        f"pre-trained checkpoints (7D or 14D). "
                        "The action head will need to be re-initialised."
                    ),
                    implication=(
                        "Fine-tuning with a custom action dim requires reinitialising "
                        "the action-prediction head from scratch, which needs more data."
                    ),
                ))

        # ── 6. Dataset size ───────────────────────────────────────────────────
        n_eps = len(batch.episodes)
        raw["n_episodes"] = n_eps
        if n_eps < self.min_episodes:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                metric="octo_dataset_size",
                observed=ObservedValue(value=float(n_eps)),
                threshold=float(self.min_episodes),
                interpretation=(
                    f"Dataset has {n_eps} episodes (recommended ≥{self.min_episodes} "
                    "for Octo fine-tuning to avoid overfitting)."
                ),
                implication=(
                    "Small datasets may cause the residual action head to overfit. "
                    "Use aggressive augmentation or lower the learning rate."
                ),
            ))

        # ── compatibility hint ────────────────────────────────────────────────
        n_critical = sum(1 for f in flags if f.level == RiskLevel.CRITICAL)
        n_warning  = sum(1 for f in flags if f.level == RiskLevel.WARNING)
        if n_critical == 0 and n_warning == 0:
            compatible: Optional[bool] = True
            explanation = "Dataset meets all Octo structural requirements."
        elif n_critical == 0:
            compatible = None
            explanation = (
                f"Dataset is likely compatible with Octo but has {n_warning} "
                "warning(s). Review flagged metrics before fine-tuning."
            )
        else:
            compatible = False
            explanation = (
                f"Dataset has {n_critical} critical issue(s) that block "
                "Octo fine-tuning."
            )

        hints.append(CompatibilityHint(
            policy_family="octo",
            compatible=compatible,
            explanation=explanation,
            caveats=[
                "Octo fine-tuning requires the octo Python package from "
                "https://github.com/octo-models/octo.",
                "Non-standard action dims require re-initialising the action head "
                "from random weights — budget more compute for convergence.",
            ],
        ))

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )
