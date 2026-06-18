"""
OpenVLA Compatibility Analyzer.

Checks whether a dataset meets the structural requirements for fine-tuning
Stanford's OpenVLA — an open Vision-Language-Action model built on Prismatic VLM.

OpenVLA requirements (https://openvla.github.io):

  1. Visual observations      — exactly one primary RGB camera per episode
                                (OpenVLA's ViT backbone processes one view by default).
  2. Language annotations     — task instruction string per episode.
  3. Single-step prediction   — OpenVLA predicts one action per step (no chunking).
                                Episodes of any length are supported, but very short
                                episodes (< 10 steps) produce negligible training signal.
  4. Control frequency        — OpenVLA was pre-trained on BridgeData V2 at ~5 Hz
                                (action-tokenised). Works well at 5–30 Hz.
  5. Action dimensionality    — OpenVLA standard: 7D continuous (Δpos + Δrot + gripper).
                                Other dims are supported but require head reconfiguration.
  6. Action space             — OpenVLA discretises each action dim into 256 bins.
                                Very high-range or bimodal action distributions may
                                lose resolution. Entropy < 2 bits/dim is flagged.

Only runs when policy_family contains "openvla". All other policy families
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

_MIN_EPISODE_STEPS:  int   = 10
_FREQ_LOW_WARNING:   float = 3.0    # Hz
_FREQ_HIGH_WARNING:  float = 30.0   # Hz
_KNOWN_ACTION_DIMS:  set[int] = {7}
_ENTROPY_WARNING:    float = 2.0    # bits/dim — below this, discretisation loses info
_VISUAL_KEYS = frozenset(["camera", "image", "rgb", "depth", "visual"])


@dataclass
class OpenVLACompatibilityAnalyzer(Analyzer):
    """
    Structural compatibility checks for OpenVLA fine-tuning.

    Parameters
    ----------
    min_episode_steps : int
        Minimum steps per episode. Episodes shorter than this are flagged.
    freq_low_warning : float
        Hz below which control frequency is flagged as too slow for OpenVLA.
    freq_high_warning : float
        Hz above which control frequency suggests sub-sampling may help.
    known_action_dims : set[int]
        Action dimensions that match standard OpenVLA robot configurations.
    entropy_warning : float
        bits/dim below which action discretisation may lose resolution.
    """

    min_episode_steps:  int       = _MIN_EPISODE_STEPS
    freq_low_warning:   float     = _FREQ_LOW_WARNING
    freq_high_warning:  float     = _FREQ_HIGH_WARNING
    known_action_dims:  set[int]  = field(default_factory=lambda: set(_KNOWN_ACTION_DIMS))
    entropy_warning:    float     = _ENTROPY_WARNING

    @property
    def name(self) -> str:
        return "openvla_compatibility"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if not policy_family or "openvla" not in policy_family.lower():
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
                metric="openvla_visual_observations",
                observed=ObservedValue(value=0.0),
                interpretation=(
                    "No RGB/visual observation stream detected. "
                    "OpenVLA's ViT backbone requires at least one camera view per step."
                ),
                implication=(
                    "OpenVLA cannot be fine-tuned without image inputs. "
                    "Ensure at least one camera observation key is present."
                ),
            ))
        else:
            # Count distinct visual streams
            visual_keys: set[str] = set()
            for ep in batch.episodes[:5]:
                for k in ep.observations:
                    if any(vk in k.lower() for vk in _VISUAL_KEYS):
                        visual_keys.add(k)
            raw["n_visual_streams"] = len(visual_keys)
            if len(visual_keys) > 1:
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="openvla_multi_camera",
                    observed=ObservedValue(value=float(len(visual_keys))),
                    interpretation=(
                        f"{len(visual_keys)} visual streams detected "
                        f"({', '.join(sorted(visual_keys))}). "
                        "OpenVLA's default architecture processes a single camera view."
                    ),
                    implication=(
                        "Multi-camera data is supported via OpenVLA-OFT or custom "
                        "forks, but the base model uses one view. Confirm your config "
                        "before fine-tuning."
                    ),
                ))

        # ── 2. Language annotations ───────────────────────────────────────────
        n_with_lang = sum(
            1 for ep in batch.episodes
            if ep.metadata.task_description and ep.metadata.task_description.strip()
        )
        lang_fraction = n_with_lang / max(len(batch.episodes), 1)
        raw["language_annotation_fraction"] = lang_fraction
        if lang_fraction < 1.0:
            level = RiskLevel.CRITICAL if lang_fraction < 0.5 else RiskLevel.WARNING
            flags.append(RiskFlag(
                level=level,
                metric="openvla_language_annotations",
                observed=ObservedValue(value=lang_fraction, unit="fraction"),
                threshold=1.0,
                interpretation=(
                    f"{lang_fraction:.0%} of episodes have language task descriptions "
                    "(OpenVLA requires 100%)."
                ),
                implication=(
                    "OpenVLA tokenises language instructions as part of the input prompt. "
                    "Episodes without annotations will use an empty string, degrading "
                    "instruction-following during inference."
                ),
            ))

        # ── 3. Episode length ─────────────────────────────────────────────────
        short_eps = [ep for ep in batch.episodes if ep.n_steps < self.min_episode_steps]
        short_frac = len(short_eps) / max(len(batch.episodes), 1)
        raw["very_short_episode_fraction"] = short_frac
        if short_frac > 0.05:
            flags.append(RiskFlag(
                level=RiskLevel.WARNING,
                metric="openvla_episode_length",
                observed=ObservedValue(value=short_frac, unit="fraction"),
                threshold=0.05,
                interpretation=(
                    f"{short_frac:.0%} of episodes have fewer than "
                    f"{self.min_episode_steps} steps."
                ),
                implication=(
                    "Very short episodes produce negligible fine-tuning signal for "
                    "OpenVLA's autoregressive training objective."
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
                    metric="openvla_control_frequency",
                    observed=ObservedValue(value=mean_freq, unit="Hz"),
                    threshold=self.freq_low_warning,
                    interpretation=(
                        f"Mean control frequency {mean_freq:.1f} Hz is below "
                        f"OpenVLA's typical operating range ({self.freq_low_warning}+ Hz)."
                    ),
                    implication=(
                        "Very low-frequency data may cause the action tokeniser to "
                        "represent large per-step displacements, compressing range."
                    ),
                ))
            elif mean_freq > self.freq_high_warning:
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="openvla_control_frequency",
                    observed=ObservedValue(value=mean_freq, unit="Hz"),
                    threshold=self.freq_high_warning,
                    interpretation=(
                        f"Mean control frequency {mean_freq:.1f} Hz is above "
                        f"OpenVLA's recommended range (≤{self.freq_high_warning} Hz)."
                    ),
                    implication=(
                        "High-frequency data makes each action step very small; "
                        "the 256-bin discretisation may saturate at the centre bins. "
                        "Consider sub-sampling to 10–15 Hz."
                    ),
                ))

        # ── 5. Action dimensionality ──────────────────────────────────────────
        if batch.episodes:
            action_dim = batch.episodes[0].action_dim
            raw["action_dim"] = action_dim
            if action_dim not in self.known_action_dims:
                flags.append(RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="openvla_action_dim",
                    observed=ObservedValue(value=float(action_dim)),
                    interpretation=(
                        f"Action dim {action_dim} differs from standard OpenVLA (7D). "
                        "OpenVLA-OFT supports custom dims with head reconfiguration."
                    ),
                    implication=(
                        "Ensure the action tokeniser bins are re-fitted to your "
                        "action range when using non-standard action dims."
                    ),
                ))

        # ── 6. Action entropy / discretisation headroom ───────────────────────
        all_actions = np.concatenate(
            [ep.actions for ep in batch.episodes if ep.actions.ndim > 1], axis=0
        ) if batch.episodes else None

        if all_actions is not None and all_actions.size > 0:
            try:
                from calibra.metrics.kinematics import compute_action_entropy
                entropy = compute_action_entropy(all_actions)
                raw["action_entropy_bits_per_dim"] = entropy
                if entropy < self.entropy_warning:
                    flags.append(RiskFlag(
                        level=RiskLevel.WARNING,
                        metric="openvla_action_discretisation",
                        observed=ObservedValue(value=entropy, unit="bits/dim"),
                        threshold=self.entropy_warning,
                        interpretation=(
                            f"Action entropy {entropy:.2f} bits/dim < {self.entropy_warning}. "
                            "Low entropy means most actions cluster in a narrow range."
                        ),
                        implication=(
                            "When OpenVLA discretises actions into 256 bins, low-entropy "
                            "distributions waste most bins. Re-normalise action range or "
                            "collect more diverse demonstrations."
                        ),
                    ))
            except Exception:
                pass

        # ── compatibility hint ────────────────────────────────────────────────
        n_critical = sum(1 for f in flags if f.level == RiskLevel.CRITICAL)
        n_warning  = sum(1 for f in flags if f.level == RiskLevel.WARNING)
        if n_critical == 0 and n_warning == 0:
            compatible: Optional[bool] = True
            explanation = "Dataset meets all OpenVLA structural requirements."
        elif n_critical == 0:
            compatible = None
            explanation = (
                f"Dataset is likely compatible with OpenVLA but has {n_warning} "
                "warning(s). Review flagged metrics before fine-tuning."
            )
        else:
            compatible = False
            explanation = (
                f"Dataset has {n_critical} critical issue(s) that block "
                "OpenVLA fine-tuning."
            )

        hints.append(CompatibilityHint(
            policy_family="openvla",
            compatible=compatible,
            explanation=explanation,
            caveats=[
                "OpenVLA uses a fixed action tokeniser fitted to BridgeData V2. "
                "Re-fit bins to your dataset range via the OpenVLA fine-tuning script.",
                "Multi-camera setups require OpenVLA-OFT or a custom architecture.",
            ],
        ))

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )
