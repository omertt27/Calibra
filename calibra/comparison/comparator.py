"""
DatasetComparator: compare two pre-computed DiagnosticReports.

Accepts DiagnosticReport objects (not raw batches) to support caching Phase 1
results and running multiple comparisons without recomputing analyzers.

Permutation tests use per-episode values stored in raw_metrics under the
"per_episode_<key>" convention (populated by each Phase 1 analyzer). For
batch-level metrics without per-episode data (e.g. entropy, PCA variance),
the comparator falls back to a CI non-overlap significance heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.schema.report import DiagnosticReport, RiskFlag, RiskLevel
from calibra.schema.comparison import ComparisonReport, DriftFlag

# ── metric direction registry ─────────────────────────────────────────────────
# (pattern, rule): rule is "up_better" (higher = better) or "down_better"
# or "ambiguous" (no inherent good direction).

_DIRECTION_RULES: list[tuple[str, str]] = [
    ("ldlj", "up_better"),
    ("action_entropy", "up_better"),
    ("state_entropy", "up_better"),
    ("timestamp_jitter_cv", "down_better"),
    ("timestamp_dropout_rate", "down_better"),
    ("jerk_spike_rate", "down_better"),
    ("velocity_discontinuity_rate", "down_better"),
    ("pca_top2_variance_fraction", "down_better"),
    ("short_episode_fraction", "down_better"),
    ("trajectory_diversity", "down_better"),  # lower sep_score = more unimodal = better
    ("camera_lag_std", "down_better"),
    ("action_obs_misalignment", "down_better"),
    ("contact_density", "ambiguous"),
    ("grasp_events_per_episode", "ambiguous"),
    ("episode_length_distribution", "ambiguous"),
]


def _metric_direction(metric: str) -> str:
    """Return "up_better", "down_better", or "ambiguous" for a metric name."""
    m = metric.lower()
    for pattern, rule in _DIRECTION_RULES:
        if pattern in m:
            return rule
    return "ambiguous"


def _infer_flag_direction(metric: str, delta: float) -> str:
    """
    Given metric name and delta = candidate.value - baseline.value, return
    "degraded", "improved", or "ambiguous".
    """
    if delta == 0.0:
        return "ambiguous"
    rule = _metric_direction(metric)
    if rule == "ambiguous":
        return "ambiguous"
    if rule == "up_better":
        return "improved" if delta > 0 else "degraded"
    # down_better
    return "degraded" if delta > 0 else "improved"


def _extract_ep_data(report: DiagnosticReport) -> dict[str, list]:
    """
    Build a flat index of per-episode arrays from a report.

    Searches all AnalyzerResult.raw_metrics for keys matching
    "per_episode_*" (the convention used by Phase 1 analyzers).
    """
    ep_data: dict[str, list] = {}
    for result in report.analyzer_results:
        for k, v in result.raw_metrics.items():
            if k.startswith("per_episode_") and isinstance(v, list):
                ep_data[k] = v
    return ep_data


# Maps a RiskFlag.metric name → the "per_episode_*" key in raw_metrics.
# Only metrics with a natural per-episode decomposition are listed.
# Metrics absent from this map fall back to CI-overlap significance.
_METRIC_TO_EP_KEY: dict[str, str] = {
    "timestamp_jitter_cv": "per_episode_jitter_cv",
    "timestamp_dropout_rate": "per_episode_dropout_fraction",
    "ldlj": "per_episode_ldlj",
    "jerk_spike_rate": "per_episode_spike_rate",
    "velocity_discontinuity_rate": "per_episode_vel_disc_rate",
    "contact_density": "per_episode_contact_fraction",
    "grasp_events_per_episode": "per_episode_grasp_count",
    "episode_length_distribution": "per_episode_length",
    "short_episode_fraction": "per_episode_length",
}


def _permutation_test(
    vals_a: list,
    vals_b: list,
    n_permutations: int = 199,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Two-sample permutation test on the difference of means.

    Returns (observed_delta, p_value) where:
      observed_delta = mean(b) - mean(a)
      p_value = fraction of permutations where |perm_delta| >= |observed_delta|

    None values are filtered from both lists before testing.
    Returns (delta, 1.0) when either group has fewer than 4 valid values,
    indicating insufficient data for a reliable test.
    """
    a = np.array([v for v in vals_a if v is not None], dtype=float)
    b = np.array([v for v in vals_b if v is not None], dtype=float)

    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0

    observed = float(np.mean(b) - np.mean(a))

    if len(a) < 4 or len(b) < 4:
        return observed, 1.0

    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)

    null_deltas = np.empty(n_permutations)
    for i in range(n_permutations):
        idx = rng.permutation(len(pooled))
        null_deltas[i] = np.mean(pooled[idx[n_a:]]) - np.mean(pooled[idx[:n_a]])

    p_value = float(np.mean(np.abs(null_deltas) >= abs(observed)))
    return observed, p_value


def _ci_overlap_significant(flag_a: RiskFlag, flag_b: RiskFlag) -> bool:
    """True if the bootstrap CIs of the two flags do not overlap."""
    lo_a = flag_a.observed.ci_lower
    hi_a = flag_a.observed.ci_upper
    lo_b = flag_b.observed.ci_lower
    hi_b = flag_b.observed.ci_upper
    if None in (lo_a, hi_a, lo_b, hi_b):
        return False
    return bool((hi_a < lo_b) or (hi_b < lo_a))


def _drift_risk_level(direction: str, significant: bool, candidate_level: RiskLevel) -> RiskLevel:
    """Assign a DriftFlag risk level based on direction and significance."""
    if not significant:
        return RiskLevel.INFO
    if direction == "improved":
        return RiskLevel.OK
    if direction == "degraded":
        return candidate_level  # absolute severity tracks the candidate's state
    return RiskLevel.INFO


def _drift_texts(
    metric: str,
    b_val: float,
    c_val: float,
    delta: float,
    direction: str,
    significant: bool,
    baseline_name: str,
    candidate_name: str,
) -> tuple[str, str]:
    sig_str = "significantly " if significant else ""
    dir_word = {"degraded": "worsened", "improved": "improved", "ambiguous": "changed"}.get(
        direction, "changed"
    )
    interp = (
        f"{metric} has {sig_str}{dir_word}: "
        f"{baseline_name}={b_val:.4g} → {candidate_name}={c_val:.4g} "
        f"(Δ={delta:+.4g})."
    )
    if direction == "degraded" and significant:
        impl = (
            f"Candidate dataset is worse on {metric}. "
            "Investigate changes to data collection or preprocessing."
        )
    elif direction == "improved" and significant:
        impl = f"Candidate dataset improved on {metric}."
    else:
        impl = "Change is not statistically significant or direction is ambiguous."
    return interp, impl


@dataclass
class DatasetComparator:
    """
    Compare two pre-computed DiagnosticReports metric by metric.

    Accepts DiagnosticReport objects (not raw batches) to enable caching
    Phase 1 results and running multiple comparisons without recomputing
    analyzers.

    Permutation tests use per-episode arrays stored in raw_metrics under the
    "per_episode_<key>" convention. For batch-level metrics without per-episode
    data (e.g. entropy, PCA variance), the test falls back to CI non-overlap.

    Parameters
    ----------
    alpha : significance threshold for permutation tests (default 0.05).
    n_permutations : controls the precision/speed tradeoff:
        - 199  → minimum useful resolution (p-values in steps of ~0.005);
                 adequate for development and fast CI.
        - 999  → publication-quality resolution (steps of ~0.001); ~5× slower.
        - 9999 → high precision for detecting marginal effects; ~50× slower.
        The default of 199 is intentionally conservative for speed. Users
        copying this into production configs should consider 999 or higher,
        particularly when small regressions in safety-critical metrics matter.
    """

    alpha: float = 0.05
    n_permutations: int = 199

    def compare(
        self,
        baseline: DiagnosticReport,
        candidate: DiagnosticReport,
    ) -> ComparisonReport:
        """
        Compare two pre-computed DiagnosticReports.

        Returns a ComparisonReport with one DriftFlag per metric present in
        both reports. Access significant regressions via report.degraded and
        significant improvements via report.improved.
        """
        baseline_ep = _extract_ep_data(baseline)
        candidate_ep = _extract_ep_data(candidate)

        # Index flags by metric name from each report.
        base_flags: dict[str, tuple[str, RiskFlag]] = {}
        for result in baseline.analyzer_results:
            for flag in result.flags:
                base_flags[flag.metric] = (result.analyzer_name, flag)

        cand_flags: dict[str, tuple[str, RiskFlag]] = {}
        for result in candidate.analyzer_results:
            for flag in result.flags:
                cand_flags[flag.metric] = (result.analyzer_name, flag)

        drift_flags: list[DriftFlag] = []

        for metric, (analyzer_name, base_flag) in base_flags.items():
            if metric not in cand_flags:
                continue
            _, cand_flag = cand_flags[metric]

            b_val = base_flag.observed.value
            c_val = cand_flag.observed.value
            if b_val is None or c_val is None:
                continue  # skip metrics that could not be computed in either report

            delta = c_val - b_val
            rel = delta / abs(b_val) if abs(b_val) > 1e-12 else None

            # Attempt permutation test; fall back to CI overlap.
            ep_key = _METRIC_TO_EP_KEY.get(metric)
            p_value: Optional[float] = None
            significant = False

            if ep_key and ep_key in baseline_ep and ep_key in candidate_ep:
                _, p_value = _permutation_test(
                    baseline_ep[ep_key],
                    candidate_ep[ep_key],
                    self.n_permutations,
                )
                significant = p_value < self.alpha
            else:
                significant = _ci_overlap_significant(base_flag, cand_flag)

            direction = _infer_flag_direction(metric, delta)
            level = _drift_risk_level(direction, significant, cand_flag.level)
            interp, impl = _drift_texts(
                metric,
                b_val,
                c_val,
                delta,
                direction,
                significant,
                baseline.dataset_name,
                candidate.dataset_name,
            )

            drift_flags.append(
                DriftFlag(
                    metric=metric,
                    analyzer_name=analyzer_name,
                    baseline_observed=base_flag.observed,
                    candidate_observed=cand_flag.observed,
                    delta=delta,
                    relative_change=rel,
                    p_value=p_value,
                    significant=significant,
                    direction=direction,
                    level=level,
                    interpretation=interp,
                    implication=impl,
                )
            )

        return ComparisonReport(
            baseline_name=baseline.dataset_name,
            candidate_name=candidate.dataset_name,
            baseline_n_episodes=baseline.n_episodes,
            candidate_n_episodes=candidate.n_episodes,
            n_permutations=self.n_permutations,
            alpha=self.alpha,
            drift_flags=drift_flags,
        )
