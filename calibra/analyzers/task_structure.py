"""
Task Structure Analyzer.

Unlike temporal, smoothness, and coverage analyzers which flag data quality
problems, this analyzer characterizes the NATURE of the demonstrations —
how contact-rich they are, whether multiple strategies coexist, and whether
failed demos contaminate the dataset.

Most flags are INFO (characterization, not a defect). Exceptions:
  - Short episode fraction (WARNING/CRITICAL): suggests failed demonstrations
    were not filtered before training.
  - Trajectory multimodality (WARNING): multiple distinct strategies in one
    dataset create conflicting BC gradients — the policy sees contradictory
    (obs, action) pairs for similar states.

Metrics
-------
1. Contact density
   Fraction of steps in contact/slow phase, estimated via gripper state
   (auto-detected bimodal action dim) and/or velocity envelope. Reported as
   INFO so the engineer can interpret it in the context of the task.

2. Grasp events per episode
   Mean gripper open→close transitions per episode. Zero grasps on a
   pick-and-place task is a red flag; twenty grasps is likely noise.

3. Trajectory diversity (multimodal detection)
   Each episode is summarised as a feature vector (per-dim mean + std of
   actions). These are projected to 2D PCA space and tested with 2-means.
   A low within-cluster variance ratio indicates two distinct trajectory
   clusters. No scipy dependency — 2-means is implemented from scratch.

4. Short episode fraction
   Episodes with step count < Q1 - 1.5×IQR are outliers. High fractions
   suggest data collection artefacts (e.g. early-termination on failure)
   were not cleaned before export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.analyzers.temporal import _bootstrap_ci
from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import (
    AnalyzerResult,
    CompatibilityHint,
    ObservedValue,
    RiskFlag,
    RiskLevel,
)

# ── thresholds ───────────────────────────────────────────────────────────────

_SHORT_EP_WARNING = 0.05  # 5% of episodes are outlier-short
_SHORT_EP_CRITICAL = 0.15  # 15%

_MULTIMODAL_SEP_INFO = 0.35  # weak 2-cluster structure → INFO
_MULTIMODAL_SEP_WARNING = 0.60  # clear 2-cluster structure → WARNING

_GRIPPER_BIMODAL_THRESHOLD = 0.65  # fraction of values near extremes
_GRIPPER_MIDDLE_MAX = 0.10  # max fraction in the [0.3, 0.7] band


@dataclass
class TaskStructureAnalyzer(Analyzer):
    """
    Task-structure characterization for robotics IL datasets.

    Parameters
    ----------
    gripper_dims : explicit list of action column indices to use as gripper
                   signals. If None (default), auto-detected by bimodality.
    vel_slow_threshold : fraction of max speed below which a step is
                         considered a contact/slow phase step.
    action_type : "position" | "velocity". Controls how velocity is
                  derived from actions for contact detection.
    short_ep_warning, short_ep_critical : IQR-outlier fraction thresholds.
    multimodal_sep_info, multimodal_sep_warning : separation score thresholds
                  for the 2-means trajectory diversity test.
    n_bootstrap, ci_level : bootstrap CI parameters.
    """

    gripper_dims: Optional[list[int]] = None
    vel_slow_threshold: float = 0.08
    action_type: str = "position"
    short_ep_warning: float = _SHORT_EP_WARNING
    short_ep_critical: float = _SHORT_EP_CRITICAL
    multimodal_sep_info: float = _MULTIMODAL_SEP_INFO
    multimodal_sep_warning: float = _MULTIMODAL_SEP_WARNING
    n_bootstrap: int = 500
    ci_level: float = 0.95

    @property
    def name(self) -> str:
        return "task_structure"

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

        # Resolve gripper dims once for the whole batch.
        actions_all = _collect_actions(batch)
        g_dims = (
            self.gripper_dims
            if self.gripper_dims is not None
            else _detect_gripper_dims(actions_all)
        )
        raw["detected_gripper_dims"] = g_dims

        contact_flag, contact_raw = self._check_contact_density(batch, g_dims)
        flags.append(contact_flag)
        raw["contact_density"] = contact_raw

        grasp_flag, grasp_raw = self._check_grasp_events(batch, g_dims)
        flags.append(grasp_flag)
        raw["grasp_events"] = grasp_raw

        diversity_flag, diversity_raw = self._check_trajectory_diversity(batch)
        flags.append(diversity_flag)
        raw["trajectory_diversity"] = diversity_raw

        short_flag, short_raw = self._check_short_episodes(batch)
        flags.append(short_flag)
        raw["short_episodes"] = short_raw

        hints = self._policy_hints(flags, policy_family, raw)

        # Per-episode arrays for Phase 2 comparison/curation (convention: "per_episode_<key>").
        raw["per_episode_contact_fraction"] = contact_raw.get("episode_values", [])
        raw["per_episode_grasp_count"] = grasp_raw.get("episode_values", [])

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            hints=hints,
            raw_metrics=raw,
        )

    # ── metric: contact density ──────────────────────────────────────────────

    def _check_contact_density(
        self, batch: EpisodeBatch, g_dims: list[int]
    ) -> tuple[RiskFlag, dict]:
        ep_values = [
            _episode_contact_fraction(ep, g_dims, self.vel_slow_threshold, self.action_type)
            for ep in batch.episodes
        ]
        arr = np.array(ep_values)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)

        raw = {
            "mean_contact_fraction": float(stat),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "gripper_used": len(g_dims) > 0,
            "episode_values": ep_values,
        }

        if len(g_dims) > 0:
            method = "gripper state + velocity envelope"
        else:
            method = "velocity envelope only (no gripper detected)"

        return RiskFlag(
            level=RiskLevel.INFO,
            metric="contact_density",
            observed=ObservedValue(
                value=stat,
                unit="fraction",
                ci_lower=lo,
                ci_upper=hi,
                ci_level=self.ci_level,
                ci_method="bootstrap",
            ),
            interpretation=(
                f"{stat:.1%} of steps are in contact/slow phase (estimated via {method})."
            ),
            implication=(
                "High contact density (> 60%) → dataset is contact-rich; "
                "policy must learn precise contact-phase behaviour. "
                "Low density (< 10%) → mostly free-space motion; "
                "consider adding contact-phase demonstrations for manipulation tasks."
            ),
        ), raw

    # ── metric: grasp events ─────────────────────────────────────────────────

    def _check_grasp_events(self, batch: EpisodeBatch, g_dims: list[int]) -> tuple[RiskFlag, dict]:
        if not g_dims:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="grasp_events_per_episode",
                observed=ObservedValue(value=None),
                interpretation="No gripper dimension detected — grasp count unavailable.",
                implication=(
                    "Specify gripper_dims=[<dim_index>] on TaskStructureAnalyzer "
                    "if the dataset has a gripper."
                ),
            ), {"skipped": "no gripper dims detected", "episode_values": []}

        ep_values = [_episode_grasp_count(ep, g_dims) for ep in batch.episodes]
        valid = [c for c in ep_values if c is not None]
        if not valid:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="grasp_events_per_episode",
                observed=ObservedValue(value=None),
                interpretation="Gripper dim found but too few steps to count events.",
                implication="Episodes are too short to characterise grasp patterns.",
            ), {"skipped": "episodes too short", "episode_values": ep_values}

        arr = np.array(valid, dtype=float)
        stat, lo, hi = _bootstrap_ci(arr, np.mean, self.n_bootstrap, self.ci_level)
        n_zero = int(np.sum(arr == 0))
        raw = {
            "mean_grasps_per_episode": float(stat),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "episodes_with_zero_grasps": n_zero,
            "n_episodes": len(valid),
            "episode_values": ep_values,
        }

        zero_frac = n_zero / len(valid)
        if zero_frac > 0.5:
            return RiskFlag(
                level=RiskLevel.WARNING,
                metric="grasp_events_per_episode",
                observed=ObservedValue(
                    value=stat,
                    ci_lower=lo,
                    ci_upper=hi,
                    ci_level=self.ci_level,
                    ci_method="bootstrap",
                ),
                interpretation=(
                    f"{zero_frac:.1%} of episodes have zero gripper close events "
                    f"(mean {stat:.1f} grasps/episode)."
                ),
                implication=(
                    "More than half the demonstrations never close the gripper. "
                    "If this is a grasping task, many episodes may be incomplete "
                    "or failed attempts that reached the object but did not grasp."
                ),
                affected_fraction=zero_frac,
            ), raw

        return RiskFlag(
            level=RiskLevel.INFO,
            metric="grasp_events_per_episode",
            observed=ObservedValue(
                value=stat,
                ci_lower=lo,
                ci_upper=hi,
                ci_level=self.ci_level,
                ci_method="bootstrap",
            ),
            interpretation=f"Mean {stat:.1f} gripper close events per episode.",
            implication=(
                "Use this to verify task structure matches expectations. "
                "Pick-and-place should have 1-2 grasps; insertion tasks typically 1."
            ),
        ), raw

    # ── metric: trajectory diversity ─────────────────────────────────────────

    def _check_trajectory_diversity(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        if batch.n_episodes < 4:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="trajectory_diversity",
                observed=ObservedValue(value=None),
                interpretation="Too few episodes to assess trajectory diversity.",
                implication="Collect at least 4 episodes for meaningful clustering.",
            ), {"skipped": "too few episodes"}

        n_modes, sep_score, _ = _batch_trajectory_diversity(batch)
        raw = {
            "estimated_modes": n_modes,
            "separation_score": float(sep_score),
            "n_episodes": batch.n_episodes,
        }

        if sep_score >= self.multimodal_sep_warning:
            return RiskFlag(
                level=RiskLevel.WARNING,
                metric="trajectory_diversity",
                observed=ObservedValue(value=sep_score),
                threshold=self.multimodal_sep_warning,
                interpretation=(
                    f"Trajectory diversity score {sep_score:.2f} ≥ "
                    f"{self.multimodal_sep_warning:.2f} — dataset appears to "
                    f"contain ~{n_modes} distinct trajectory modes."
                ),
                implication=(
                    "Multiple distinct demonstration strategies produce conflicting "
                    "BC gradients: the policy receives contradictory action signals "
                    "for similar observations. Standard BC loss will average the "
                    "modes, resulting in poor performance on both. "
                    "Consider: (1) clustering and training separate policies, "
                    "(2) using a multimodal policy (Diffusion, GMM), or "
                    "(3) filtering to one strategy."
                ),
            ), raw

        if sep_score >= self.multimodal_sep_info:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="trajectory_diversity",
                observed=ObservedValue(value=sep_score),
                threshold=self.multimodal_sep_warning,
                interpretation=(
                    f"Trajectory diversity score {sep_score:.2f} — weak evidence "
                    "of multiple strategies. May be within-task variation."
                ),
                implication=(
                    "Consider visualising the episode PCA projection to determine "
                    "whether the spread represents genuine strategy variation or "
                    "normal trajectory noise."
                ),
            ), raw

        return RiskFlag(
            level=RiskLevel.OK,
            metric="trajectory_diversity",
            observed=ObservedValue(value=sep_score),
            threshold=self.multimodal_sep_info,
            interpretation=(
                f"Trajectory diversity score {sep_score:.2f} — demonstrations "
                "appear to follow a single primary strategy."
            ),
            implication="No multimodal strategy risk detected.",
        ), raw

    # ── metric: short episode fraction ───────────────────────────────────────

    def _check_short_episodes(self, batch: EpisodeBatch) -> tuple[RiskFlag, dict]:
        lengths = np.array([ep.n_steps for ep in batch.episodes], dtype=float)

        if len(lengths) < 4:
            return RiskFlag(
                level=RiskLevel.INFO,
                metric="short_episode_fraction",
                observed=ObservedValue(value=None),
                interpretation="Too few episodes for IQR outlier detection.",
                implication="Need at least 4 episodes.",
            ), {"skipped": "too few episodes"}

        q1, q3 = np.percentile(lengths, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr

        outlier_mask = lengths < lower_fence
        frac = float(np.mean(outlier_mask))
        outlier_ids = [batch.episodes[i].metadata.episode_id for i in np.where(outlier_mask)[0]]
        raw = {
            "short_episode_fraction": frac,
            "lower_fence_steps": float(lower_fence),
            "q1_steps": float(q1),
            "q3_steps": float(q3),
            "iqr_steps": float(iqr),
            "outlier_episode_ids": outlier_ids[:20],  # cap for serialisation
            "n_outliers": int(np.sum(outlier_mask)),
        }

        level = _threshold_level_upper(frac, self.short_ep_warning, self.short_ep_critical)

        if level == RiskLevel.OK or frac == 0.0:
            return RiskFlag(
                level=RiskLevel.OK,
                metric="short_episode_fraction",
                observed=ObservedValue(value=frac, unit="fraction"),
                threshold=self.short_ep_warning,
                interpretation="No suspiciously short episodes detected.",
                implication="Episode length distribution appears consistent.",
            ), raw

        return RiskFlag(
            level=level,
            metric="short_episode_fraction",
            observed=ObservedValue(value=frac, unit="fraction"),
            threshold=self.short_ep_warning,
            interpretation=(
                f"{frac:.1%} of episodes ({int(np.sum(outlier_mask))}/{len(lengths)}) "
                f"have fewer than {lower_fence:.0f} steps (lower IQR fence). "
                f"IDs: {outlier_ids[:5]}{'...' if len(outlier_ids) > 5 else ''}"
            ),
            implication=(
                "Short-outlier episodes are likely failed demonstrations that were "
                "not filtered: the operator aborted early, hit a safety limit, or "
                "the task infrastructure triggered early termination. "
                "BC policies trained on these episodes learn to abort tasks prematurely. "
                "Filter episodes below the lower fence before training."
            ),
            affected_fraction=frac,
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

        contact_frac = raw.get("contact_density", {}).get("mean_contact_fraction")
        sep_score = raw.get("trajectory_diversity", {}).get("separation_score")
        n_modes = raw.get("trajectory_diversity", {}).get("estimated_modes", 1)
        short_frac = raw.get("short_episodes", {}).get("short_episode_fraction")

        if "diffusion" in pf:
            caveats: list[str] = []
            compatible: Optional[bool] = True
            if sep_score is not None and sep_score >= self.multimodal_sep_info:
                caveats.append(
                    f"Diffusion Policy explicitly models multi-modal action "
                    f"distributions — it is well-suited for datasets with "
                    f"~{n_modes} trajectory modes."
                )
                # Multimodal → diffusion is MORE compatible, not less.
            if short_frac is not None and short_frac > self.short_ep_warning:
                caveats.append(
                    "Short/failed episodes will corrupt the diffusion score "
                    "at the early-episode part of the trajectory distribution."
                )
                compatible = None
            hints.append(
                CompatibilityHint(
                    policy_family="Diffusion Policy",
                    compatible=compatible,
                    explanation=(
                        "Diffusion Policy handles contact-rich and multi-modal data "
                        "well, but is sensitive to failed-demo contamination."
                    ),
                    caveats=caveats,
                )
            )

        if pf in ("act", "action chunking"):
            caveats = []
            compatible = True
            if sep_score is not None and sep_score >= self.multimodal_sep_warning:
                caveats.append(
                    f"ACT encodes demonstrations into a single latent style; "
                    f"with {n_modes} trajectory modes the style variable will be "
                    "overloaded and the CVAE posterior may collapse."
                )
                compatible = None
            if contact_frac is not None and contact_frac > 0.7:
                caveats.append(
                    "Very high contact density: ACT action chunks may span "
                    "contact transitions, requiring careful chunk length tuning."
                )
            hints.append(
                CompatibilityHint(
                    policy_family="ACT",
                    compatible=compatible,
                    explanation="ACT performs well on contact-rich single-strategy datasets.",
                    caveats=caveats,
                )
            )

        if "transformer" in pf or "bc" in pf:
            caveats = []
            compatible = True
            if sep_score is not None and sep_score >= self.multimodal_sep_warning:
                caveats.append(
                    "Standard BC with MSE loss averages conflicting modes — "
                    "the policy will hover between strategies and succeed at neither."
                )
                compatible = False
            hints.append(
                CompatibilityHint(
                    policy_family="Transformer BC" if "transformer" in pf else "Vanilla BC",
                    compatible=compatible,
                    explanation="Standard BC struggles with multi-modal demonstration sets.",
                    caveats=caveats,
                )
            )

        return hints


# ── per-episode and per-batch helpers ────────────────────────────────────────


def _collect_actions(batch: EpisodeBatch) -> np.ndarray:
    parts = [ep.actions for ep in batch.episodes if ep.n_steps > 0]
    if not parts:
        return np.empty((0, 1))
    return np.vstack([p if p.ndim > 1 else p[:, np.newaxis] for p in parts]).astype(np.float64)


def _detect_gripper_dims(
    actions: np.ndarray,
    bimodal_threshold: float = _GRIPPER_BIMODAL_THRESHOLD,
    middle_max: float = _GRIPPER_MIDDLE_MAX,
) -> list[int]:
    """
    Find action dimensions that look like discrete/binary gripper signals.

    A gripper dim has most values concentrated near the two extremes of its
    range, with few values in the middle band [0.3, 0.7] after normalisation.
    """
    if actions.ndim == 1 or actions.shape[0] < 10:
        return []

    detected = []
    for d in range(actions.shape[1]):
        col = actions[:, d].astype(np.float64)
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-8:
            continue  # constant dim

        col_norm = (col - lo) / (hi - lo)
        near_extremes = float(np.mean((col_norm < 0.2) | (col_norm > 0.8)))
        in_middle = float(np.mean((col_norm > 0.3) & (col_norm < 0.7)))

        if near_extremes >= bimodal_threshold and in_middle <= middle_max:
            detected.append(d)

    return detected


def _episode_contact_fraction(
    ep: Episode,
    gripper_dims: list[int],
    vel_slow_threshold: float = 0.08,
    action_type: str = "position",
) -> float:
    """
    Estimate fraction of steps in contact/slow phase.

    Priority: gripper-based detection first; velocity-based as supplement.
    """
    n = ep.n_steps
    if n < 2:
        return 0.0

    acts = ep.actions if ep.actions.ndim > 1 else ep.actions[:, np.newaxis]
    contact = np.zeros(n, dtype=bool)

    # Gripper-based detection
    for gd in gripper_dims:
        if gd >= acts.shape[1]:
            continue
        col = acts[:, gd].astype(np.float64)
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-8:
            continue
        col_norm = (col - lo) / (hi - lo)
        # Determine which extreme is "closed" by which side has more mass
        frac_low = float(np.mean(col_norm < 0.3))
        frac_high = float(np.mean(col_norm > 0.7))
        if frac_low > frac_high:
            contact |= col_norm < 0.5  # low = closed
        else:
            contact |= col_norm > 0.5  # high = closed

    # Velocity-based detection (supplement)
    active = [d for d in range(acts.shape[1]) if d not in gripper_dims]
    if active and action_type == "position":
        sub = acts[:, active].astype(np.float64)
        dt = float(np.median(np.diff(ep.timestamps)))
        if dt > 0 and len(sub) >= 2:
            vel = np.diff(sub, axis=0) / dt  # (T-1, D)
            speed = np.linalg.norm(vel, axis=-1)  # (T-1,)
            v_max = float(np.max(speed))
            if v_max > 0:
                slow = speed < vel_slow_threshold * v_max
                contact[1:] |= slow  # align with step indices (vel is T-1)

    return float(np.mean(contact))


def _episode_grasp_count(ep: Episode, gripper_dims: list[int]) -> Optional[int]:
    """Count gripper open→close transitions (proxy for grasp events)."""
    if not gripper_dims or ep.n_steps < 3:
        return None

    acts = ep.actions if ep.actions.ndim > 1 else ep.actions[:, np.newaxis]
    gd = gripper_dims[0]
    if gd >= acts.shape[1]:
        return None

    col = acts[:, gd].astype(np.float64)
    lo, hi = col.min(), col.max()
    if hi - lo < 1e-8:
        return 0  # constant — no transitions

    col_norm = (col - lo) / (hi - lo)
    frac_low = float(np.mean(col_norm < 0.3))
    frac_high = float(np.mean(col_norm > 0.7))
    is_closed = col_norm > 0.5 if frac_high >= frac_low else col_norm < 0.5

    transitions = np.diff(is_closed.astype(np.int8))
    return int(np.sum(transitions > 0))  # open→closed


def _batch_trajectory_diversity(
    batch: EpisodeBatch,
) -> tuple[int, float, np.ndarray]:
    """
    Test for multi-modal trajectory distribution.

    Returns (estimated_n_modes, separation_score, X_2d).
    separation_score is in [0, 1]: 0 = all episodes identical,
    1 = perfect two-cluster separation.
    """
    features = []
    for ep in batch.episodes:
        if ep.n_steps > 0:
            acts = ep.actions.astype(np.float64)
            if acts.ndim == 1:
                acts = acts[:, np.newaxis]
            feat = np.concatenate([acts.mean(axis=0), acts.std(axis=0)])
            features.append(feat)

    if len(features) < 4:
        return 1, 0.0, np.empty((0, 2))

    X = np.array(features)
    centered = X - X.mean(axis=0)

    # PCA to 2D (or fewer if action_dim is 1)
    n_components = min(2, X.shape[1])
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        X_2d = centered @ Vt[:n_components].T
    except np.linalg.LinAlgError:
        return 1, 0.0, centered[:, :n_components]

    if X_2d.shape[1] < 2:
        X_2d = np.column_stack([X_2d, np.zeros(len(X_2d))])

    labels, inertia_ratio = _two_means(X_2d)

    sep_score = float(max(0.0, 1.0 - inertia_ratio))
    n_modes = sum(1 for k in range(2) if float(np.mean(labels == k)) > 0.10)
    return n_modes, sep_score, X_2d


def _two_means(
    X: np.ndarray,
    n_iter: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    2-means clustering. Returns (labels, within_cluster_variance_ratio).

    within_cluster_variance_ratio = (weighted within-cluster var) / (total var).
    Low ratio → compact, well-separated clusters → high separation score.

    Uses 5 random restarts and returns the best (lowest inertia_ratio) result.
    """
    if len(X) < 2:
        return np.zeros(len(X), dtype=int), 1.0

    rng = np.random.default_rng(seed)
    total_var = float(np.var(X, axis=0).sum())
    if total_var <= 0:
        return np.zeros(len(X), dtype=int), 0.0

    best_labels = np.zeros(len(X), dtype=int)
    best_ratio = 1.0

    for _ in range(5):  # random restarts
        idx = rng.choice(len(X), 2, replace=False)
        centers = X[idx].astype(float).copy()
        labels = np.zeros(len(X), dtype=int)

        for _step in range(n_iter):
            d0 = np.linalg.norm(X - centers[0], axis=1)
            d1 = np.linalg.norm(X - centers[1], axis=1)
            new_labels = (d1 < d0).astype(int)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(2):
                mask = labels == k
                if mask.any():
                    centers[k] = X[mask].mean(axis=0)

        within = 0.0
        for k in range(2):
            mask = labels == k
            if mask.sum() > 1:
                within += float(np.var(X[mask], axis=0).sum()) * float(mask.mean())

        ratio = min(1.0, within / total_var)
        if ratio < best_ratio:
            best_ratio = ratio
            best_labels = labels.copy()

    return best_labels, best_ratio


def _threshold_level_upper(value: float, warning: float, critical: float) -> RiskLevel:
    if value >= critical:
        return RiskLevel.CRITICAL
    if value >= warning:
        return RiskLevel.WARNING
    return RiskLevel.OK
