"""
Coverage Entropy Analyzer.

Measures how well the dataset covers the action and state spaces.
Low coverage → mode collapse risk → policies fail to generalise.

Metrics computed:

  1. Action-space entropy (bits/dim)
     Marginal Shannon entropy averaged over action dimensions.
     Computed via histogram with Freedman-Diaconis bin-width selection.
     Thresholds calibrated to single-task fine-tuning datasets (typical
     healthy value: 2-5 bits/dim for normalised continuous actions).

  2. State-space entropy (bits/dim)
     Same, computed over proprioceptive state if available.

  3. PCA variance concentration
     Fraction of total action variance explained by the top-2 principal
     components. If one or two directions dominate, the dataset's action
     distribution is effectively low-rank — a structural indicator of
     mode collapse or single-strategy demonstrations.

  4. Episode length distribution
     Reports mean, std, and coefficient of variation (CV) of episode
     lengths. High CV with a specific bimodal shape suggests the dataset
     mixes two demonstration strategies (e.g. fast vs slow, direct vs
     recovery paths). Flagged as INFO — requires human interpretation.

Calibration note:
  Entropy values depend heavily on the number of samples and bins. All
  entropy flags include the sample count in raw_metrics so the user can
  judge reliability. With < 200 total steps, entropy estimates are
  unreliable and we emit INFO rather than WARNING/CRITICAL.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# ── thresholds ───────────────────────────────────────────────────────────────

_ACTION_ENTROPY_WARNING = 3.5  # bits/dim — low diversity (< ~12 out of 50 bins occupied)
_ACTION_ENTROPY_CRITICAL = 2.0  # bits/dim — very low (< ~4 bins occupied)

_STATE_ENTROPY_WARNING = 3.5
_STATE_ENTROPY_CRITICAL = 2.0

_PCA2_WARNING = 0.90  # top-2 PCs explain > 90% → low-rank warning
_PCA2_CRITICAL = 0.97  # top-2 PCs explain > 97% → near-rank-2 collapse

_MIN_SAMPLES_FOR_ENTROPY = 200  # below this, flag INFO not WARNING/CRITICAL

_PROPRIO_KEYS = ("proprio", "state", "joint_state", "joint_pos", "robot_state", "qpos", "obs")


@dataclass
class CoverageEntropyAnalyzer(Analyzer):
    """
    State-space and action-space coverage diagnostics.

    Parameters
    ----------
    n_bins : number of histogram bins for entropy estimation (per dimension).
             Overridden by Freedman-Diaconis when n_samples is large enough.
    action_entropy_warning, action_entropy_critical : bits/dim thresholds.
    state_entropy_warning, state_entropy_critical : bits/dim thresholds.
    pca2_warning, pca2_critical : top-2-PC variance fraction thresholds.
    proprio_keys : observation keys to use as the state signal. The first
                   matching key found in a batch is used.
    n_bootstrap, ci_level : bootstrap CI parameters (over episodes).
    """

    n_bins: int = 50
    action_entropy_warning: float = _ACTION_ENTROPY_WARNING
    action_entropy_critical: float = _ACTION_ENTROPY_CRITICAL
    state_entropy_warning: float = _STATE_ENTROPY_WARNING
    state_entropy_critical: float = _STATE_ENTROPY_CRITICAL
    pca2_warning: float = _PCA2_WARNING
    pca2_critical: float = _PCA2_CRITICAL
    proprio_keys: tuple = _PROPRIO_KEYS
    n_bootstrap: int = 500
    ci_level: float = 0.95
    action_range: Optional[tuple[float, float]] = None
    state_range: Optional[tuple[float, float]] = None

    @property
    def name(self) -> str:
        return "coverage_entropy"

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

        actions = _collect_actions(batch)
        action_flag, action_raw = self._check_action_entropy(actions, batch.n_samples)
        flags.append(action_flag)
        raw["action_entropy"] = action_raw

        pca_flag, pca_raw = self._check_pca_concentration(actions)
        flags.append(pca_flag)
        raw["pca_variance"] = pca_raw

        state_key, state_data = _collect_state(batch, self.proprio_keys)
        if state_data is not None:
            state_flag, state_raw = self._check_state_entropy(
                state_data, batch.n_samples, state_key
            )
            flags.append(state_flag)
            raw["state_entropy"] = state_raw
        else:
            raw["state_entropy"] = {"skipped": "no proprioceptive key found"}

        ep_len_flag, ep_len_raw = self._check_episode_lengths(batch)
        flags.append(ep_len_flag)
        raw["episode_lengths"] = ep_len_raw

        hints = self._policy_hints(flags, policy_family, raw)

        # Per-episode arrays for Phase 2 comparison/curation (convention: "per_episode_<key>").
        raw["per_episode_length"] = [ep.n_steps for ep in batch.episodes]

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )

    # ── metric: action entropy ───────────────────────────────────────────────

    def _check_action_entropy(self, actions: np.ndarray, n_samples: int) -> tuple[RiskFlag, dict]:
        if actions.shape[0] < 10:
            return self._skip_flag(
                "action_entropy_bits_per_dim", "fewer than 10 action samples"
            ), {}

        entropy = _marginal_entropy_bits(actions, self.n_bins, self.action_range)
        raw = {
            "entropy_bits_per_dim": float(entropy),
            "n_samples": n_samples,
            "action_dim": actions.shape[1] if actions.ndim > 1 else 1,
            "action_range": self.action_range,
        }

        # Down-grade to INFO if sample count is too low for reliable estimation.
        if n_samples < _MIN_SAMPLES_FOR_ENTROPY:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="action_entropy_bits_per_dim",
                observed=ObservedValue(value=entropy, unit="bits/dim"),
                interpretation=(
                    f"Action entropy = {entropy:.2f} bits/dim "
                    f"({n_samples} samples — below reliable estimation threshold)."
                ),
                implication=(
                    "Entropy estimate may be unreliable with < "
                    f"{_MIN_SAMPLES_FOR_ENTROPY} samples. Collect more data "
                    "before acting on this metric."
                ),
            ), raw

        level = _threshold_level_lower(
            entropy, self.action_entropy_warning, self.action_entropy_critical
        )

        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="action_entropy_bits_per_dim",
                observed=ObservedValue(value=entropy, unit="bits/dim"),
                threshold=self.action_entropy_warning,
                interpretation=(f"Action-space coverage is healthy ({entropy:.2f} bits/dim)."),
                implication="No mode collapse risk detected in action space.",
            ), raw

        return RiskFlag(
            level=level,
            metric="action_entropy_bits_per_dim",
            observed=ObservedValue(value=entropy, unit="bits/dim"),
            threshold=self.action_entropy_warning,
            interpretation=(
                f"Low action diversity: {entropy:.2f} bits/dim "
                f"(threshold: >{self.action_entropy_warning:.1f} bits/dim). "
                "The dataset's action distribution is narrow."
            ),
            implication=(
                "Low action entropy indicates mode collapse in demonstrations — "
                "the operator followed similar trajectories each time. Policies "
                "trained on this data will likely fail on out-of-distribution "
                "starting configurations. Consider augmenting with recovery "
                "demonstrations or perturbation data."
            ),
        ), raw

    # ── metric: PCA variance concentration ──────────────────────────────────

    def _check_pca_concentration(self, actions: np.ndarray) -> tuple[RiskFlag, dict]:
        if actions.shape[0] < 10 or (actions.ndim > 1 and actions.shape[1] < 3):
            return self._skip_flag(
                "pca_top2_variance_fraction", "action space is < 3-dimensional"
            ), {}

        frac, explained = _pca_top_k_fraction(actions, k=2)
        raw = {"top2_fraction": float(frac), "explained_per_pc": [float(v) for v in explained]}

        level = _threshold_level_upper(frac, self.pca2_warning, self.pca2_critical)

        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="pca_top2_variance_fraction",
                observed=ObservedValue(value=frac, unit="fraction"),
                threshold=self.pca2_warning,
                interpretation=(
                    f"Action variance is spread across many PCs (top-2 explain {frac:.1%})."
                ),
                implication="No low-rank action collapse detected.",
            ), raw

        return RiskFlag(
            level=level,
            metric="pca_top2_variance_fraction",
            observed=ObservedValue(value=frac, unit="fraction"),
            threshold=self.pca2_warning,
            interpretation=(
                f"Top-2 principal components explain {frac:.1%} of action variance. "
                "The action distribution is near-rank-2."
            ),
            implication=(
                "A near-rank-2 action distribution means the dataset's "
                "demonstrated actions live in a 2D subspace of the full action "
                "space. The policy will likely fail in any task configuration "
                "that requires motion in the unexplored dimensions. This is a "
                "strong indicator of single-strategy demonstrations."
            ),
        ), raw

    # ── metric: state entropy ────────────────────────────────────────────────

    def _check_state_entropy(
        self, state: np.ndarray, n_samples: int, key: str
    ) -> tuple[RiskFlag, dict]:
        if state.shape[0] < 10:
            return self._skip_flag(
                f"state_entropy_bits_per_dim[{key}]", "fewer than 10 state samples"
            ), {}

        entropy = _marginal_entropy_bits(state, self.n_bins, self.state_range)
        raw = {
            "entropy_bits_per_dim": float(entropy),
            "key": key,
            "n_samples": n_samples,
            "state_range": self.state_range,
        }

        if n_samples < _MIN_SAMPLES_FOR_ENTROPY:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric=f"state_entropy_bits_per_dim[{key}]",
                observed=ObservedValue(value=entropy, unit="bits/dim"),
                interpretation=(f"State entropy = {entropy:.2f} bits/dim ({n_samples} samples)."),
                implication="Insufficient samples for reliable state entropy estimate.",
            ), raw

        level = _threshold_level_lower(
            entropy, self.state_entropy_warning, self.state_entropy_critical
        )

        if level == RiskLevel.OK:
            return RiskFlag(
                level=RiskLevel.OK,
                metric=f"state_entropy_bits_per_dim[{key}]",
                observed=ObservedValue(value=entropy, unit="bits/dim"),
                threshold=self.state_entropy_warning,
                interpretation=f"State-space coverage is healthy ({entropy:.2f} bits/dim).",
                implication="No mode collapse risk detected in state space.",
            ), raw

        return RiskFlag(
            level=level,
            metric=f"state_entropy_bits_per_dim[{key}]",
            observed=ObservedValue(value=entropy, unit="bits/dim"),
            threshold=self.state_entropy_warning,
            interpretation=(
                f"Low state diversity on '{key}': {entropy:.2f} bits/dim. "
                "The robot visited a narrow range of configurations."
            ),
            implication=(
                "Low state entropy means demonstrations started and ended in "
                "similar configurations. The policy will overfit to a narrow "
                "operational envelope and fail on novel start states."
            ),
        ), raw

    # ── metric: episode length distribution ──────────────────────────────────

    def _check_episode_lengths(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        lengths = np.array([ep.n_steps for ep in batch.episodes], dtype=np.float64)

        if len(lengths) < 2:
            return self._skip_flag("episode_length_distribution", "need at least 2 episodes"), {}

        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))
        cv = std_len / mean_len if mean_len > 0 else 0.0
        bimodal = _is_bimodal_heuristic(lengths)

        raw = {
            "mean_steps": mean_len,
            "std_steps": std_len,
            "min_steps": float(np.min(lengths)),
            "max_steps": float(np.max(lengths)),
            "cv": cv,
            "bimodal_hint": bimodal,
            "n_episodes": len(lengths),
        }

        if bimodal:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="episode_length_distribution",
                observed=ObservedValue(value=cv, unit="CV"),
                interpretation=(
                    f"Episode length distribution is bimodal "
                    f"(mean={mean_len:.0f} steps, CV={cv:.2f}). "
                    "Dataset may contain mixed demonstration strategies."
                ),
                implication=(
                    "Bimodal episode lengths often indicate a mixture of "
                    "successful short paths and longer recovery trajectories, "
                    "or two distinct task strategies. Consider clustering by "
                    "episode length and training separate policies, or filtering "
                    "to one mode before training."
                ),
            ), raw

        if cv > 0.5:
            return RiskFlag(
                level=RiskLevel.WARNING,
                metric="episode_length_distribution",
                observed=ObservedValue(value=cv, unit="CV"),
                interpretation=(
                    f"High episode length variability (CV={cv:.2f}, "
                    f"range [{raw['min_steps']:.0f}, {raw['max_steps']:.0f}] steps)."
                ),
                implication=(
                    "High length variability makes fixed-horizon policy "
                    "evaluation unreliable and can indicate inconsistent task "
                    "completion criteria in data collection."
                ),
            ), raw

        return RiskFlag(
            level=RiskLevel.OK,
            metric="episode_length_distribution",
            observed=ObservedValue(value=cv, unit="CV"),
            interpretation=(f"Episode lengths are consistent (mean={mean_len:.0f}, CV={cv:.2f})."),
            implication="No episode length irregularity detected.",
        ), raw

    # ── policy hints ─────────────────────────────────────────────────────────

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
        pca2 = raw.get("pca_variance", {}).get("top2_fraction")
        act_entropy = raw.get("action_entropy", {}).get("entropy_bits_per_dim")

        if "diffusion" in pf:
            caveats: list[str] = []
            compatible: Optional[bool] = True
            if pca2 is not None and pca2 > self.pca2_warning:
                caveats.append(
                    f"Diffusion Policy learns a score over the full action space. "
                    f"A near-rank-2 action distribution ({pca2:.1%} top-2 variance) "
                    "can cause the score to be poorly conditioned in low-variance "
                    "dimensions, producing erratic samples outside the training manifold."
                )
                compatible = None
            hints.append(
                CompatibilityHint(
                    policy_family="Diffusion Policy",
                    compatible=compatible,
                    explanation="Diffusion scores benefit from well-spread action distributions.",
                    caveats=caveats,
                )
            )

        if pf in ("act", "action chunking"):
            caveats = []
            compatible = True
            if act_entropy is not None and act_entropy < self.action_entropy_warning:
                caveats.append(
                    "ACT's cross-attention compresses action chunks into a "
                    "latent; low action diversity means the latent will be "
                    "near-constant and the style variable uninformative."
                )
                compatible = None
            hints.append(
                CompatibilityHint(
                    policy_family="ACT",
                    compatible=compatible,
                    explanation="ACT benefits from diverse action chunks for meaningful latent compression.",
                    caveats=caveats,
                )
            )

        return hints

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _skip_flag(metric: str, reason: str) -> RiskFlag:
        return RiskFlag(
            level=RiskLevel.INFO,
            metric=metric,
            observed=ObservedValue(value=None),
            interpretation=f"Metric skipped: {reason}.",
            implication="Cannot assess this metric with available data.",
        )


# ── data collection helpers ──────────────────────────────────────────────────


def _collect_actions(batch: EpisodeBatch) -> np.ndarray:
    """Stack all actions across all episodes → (N_total, action_dim)."""
    parts = [ep.actions for ep in batch.episodes if ep.n_steps > 0]
    if not parts:
        return np.empty((0, 1))
    stacked = np.vstack([p if p.ndim > 1 else p[:, np.newaxis] for p in parts])
    return stacked.astype(np.float64)


def _collect_state(batch: EpisodeBatch, keys: tuple) -> tuple[Optional[str], Optional[np.ndarray]]:
    """Find the first proprioceptive key present and stack across episodes."""
    target_key = None
    for key in keys:
        if any(key in ep.observations for ep in batch.episodes):
            target_key = key
            break

    if target_key is None:
        return None, None

    parts = [
        ep.observations[target_key]
        for ep in batch.episodes
        if target_key in ep.observations and ep.n_steps > 0
    ]
    if not parts:
        return target_key, None

    stacked = np.vstack([p if p.ndim > 1 else p[:, np.newaxis] for p in parts])
    return target_key, stacked.astype(np.float64)


# ── statistical helpers ──────────────────────────────────────────────────────


def _marginal_entropy_bits(
    data: np.ndarray,
    n_bins: int = 50,
    data_range: Optional[tuple[float, float]] = None,
) -> float:
    """
    Average per-dimension marginal Shannon entropy, in bits.

    `data_range` is the critical parameter for mode-collapse detection.
    When provided, bins are fixed over [lo, hi] — so a distribution that
    only occupies a narrow sub-range genuinely populates fewer bins and
    produces lower entropy. Without it, bins span the observed range, which
    makes any continuous distribution appear high-entropy (caveat documented
    on CoverageEntropyAnalyzer.action_range).

    Design note: adaptive bin-width rules (Freedman-Diaconis, Scott) are
    excluded because they rescale to the data's own spread, making a
    concentrated distribution appear as "high entropy" at fine resolution.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    N, D = data.shape
    total = 0.0

    for d in range(D):
        col = data[:, d]

        if data_range is not None:
            lo, hi = float(data_range[0]), float(data_range[1])
        else:
            lo, hi = float(col.min()), float(col.max())

        col_range = hi - lo
        if col_range <= 0:
            continue  # constant dimension contributes 0 entropy

        bin_edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(col, bins=bin_edges)
        total_count = counts.sum()
        if total_count == 0:
            continue
        probs = counts / total_count
        probs = probs[probs > 0]
        total += float(-np.sum(probs * np.log2(probs)))

    return total / D if D > 0 else 0.0


def _pca_top_k_fraction(data: np.ndarray, k: int = 2) -> tuple[float, list[float]]:
    """
    Fraction of total variance explained by the top-k principal components.

    Returns (fraction, [var_pc_1, var_pc_2, ...]) where var values are
    fractions (not raw eigenvalues).
    """
    if data.ndim == 1 or data.shape[1] < 2:
        return 1.0, [1.0]

    centered = data - data.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0, []  # treat SVD failure as fully concentrated (conservative)

    var = s**2
    total_var = float(var.sum())
    if total_var <= 0:
        return 1.0, [1.0]

    fracs = (var / total_var).tolist()
    top_k = min(k, len(fracs))
    return float(sum(fracs[:top_k])), [float(f) for f in fracs[:top_k]]


def _is_bimodal_heuristic(lengths: np.ndarray) -> bool:
    """
    Simple bimodality heuristic: high CV AND the distribution splits
    cleanly around the mean (i.e. few samples near the mean relative
    to those far from it).

    Not a formal statistical test — intended to generate an INFO flag
    for human review, not a decision.
    """
    if len(lengths) < 6:
        return False

    mean = float(np.mean(lengths))
    std = float(np.std(lengths))
    cv = std / mean if mean > 0 else 0.0

    if cv < 0.3:
        return False

    # Count samples within 0.5 std of mean (the "valley" in a bimodal dist).
    near_mean = np.sum(np.abs(lengths - mean) < 0.5 * std)
    valley_fraction = near_mean / len(lengths)
    return bool(valley_fraction < 0.15)


def _threshold_level_upper(value: float, warning: float, critical: float) -> RiskLevel:
    if value >= critical:
        return RiskLevel.CRITICAL
    if value >= warning:
        return RiskLevel.WARNING
    return RiskLevel.OK


def _threshold_level_lower(value: float, warning: float, critical: float) -> RiskLevel:
    if value <= critical:
        return RiskLevel.CRITICAL
    if value <= warning:
        return RiskLevel.WARNING
    return RiskLevel.OK
