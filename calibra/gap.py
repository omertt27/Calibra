"""
calibra gap — Coverage gap analysis: exactly what to collect next.

Analyses a training dataset's action-space coverage, identifies underrepresented
regions (gaps), and outputs a concrete collection brief:

  • How many episodes of each type you need
  • Which action dimensions to focus on
  • Estimated success rate improvement per cluster collected

Algorithm
---------
  1. Run the full Calibra diagnostic pipeline on the dataset.
  2. Build a per-episode action feature vector (mean + std per action dim).
  3. K-means cluster the feature space into N regions.
  4. For each cluster: compute current coverage and a target based on
     uniform coverage at the requested --keep fraction.
  5. Rank gaps by (target - current) weighted by cluster entropy value.
  6. Estimate the incremental success-rate impact of closing each gap using
     the action-entropy weight from the predict scoring rubric.
  7. Sum to a total collection brief: N targeted episodes to reach --target-success.

Usage
-----
    calibra gap /data/my_demos --policy diffusion
    calibra gap /data/my_demos --policy diffusion --target-success 0.85
    calibra gap /data/my_demos --policy act --clusters 12 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra import __version__
from calibra.pipeline import Pipeline
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN = "─" * _WIDTH

# Entropy weight from predict.py scoring rubric (penalty at warning per bit/dim deficit)
_ENTROPY_WEIGHT_PER_BIT = 10.0 / (3.5 - 2.0)  # 10pt penalty over 1.5 bit range


# ── result schema ──────────────────────────────────────────────────────────────

@dataclass
class CoverageGap:
    cluster_id: int
    current_episodes: int
    target_episodes: int
    needed: int
    action_dim_ranges: dict[str, tuple[float, float]]   # dim_name → (lo, hi)
    estimated_success_delta: float                       # percentage points
    coverage_fraction: float                             # current / target


@dataclass
class GapAnalysisResult:
    dataset_name: str
    n_episodes: int
    policy_family: str
    current_success: float
    target_success: float
    n_clusters: int
    gaps: list[CoverageGap]
    total_needed: int
    estimated_success_after: float

    def summary(self) -> str:
        lines = [
            _THICK,
            "  CALIBRA COVERAGE GAP ANALYSIS",
            _THICK,
            f"  Dataset  : {self.dataset_name}  ·  {self.n_episodes} episodes  ·  policy: {self.policy_family}",
            f"  Current predicted success : {self.current_success:.0f}%",
            f"  Target success           : {self.target_success:.0f}%",
            _THIN,
            "  COVERAGE GAPS  (sorted by impact)",
            _THIN,
        ]

        for i, gap in enumerate(self.gaps[:6], 1):
            lines.append(f"  {i}. Cluster {gap.cluster_id}  "
                         f"({gap.current_episodes} episodes now → need {gap.target_episodes})")
            for dim, (lo, hi) in list(gap.action_dim_ranges.items())[:4]:
                lines.append(f"       {dim}: [{lo:.3f}, {hi:.3f}]")
            lines.append(f"     Collect {gap.needed} more  ·  "
                         f"est. impact: +{gap.estimated_success_delta:.1f}% success")
            lines.append("")

        lines += [
            _THIN,
            "  COLLECTION BRIEF",
            _THIN,
            f"  Collect {self.total_needed} targeted episodes to reach "
            f"~{self.estimated_success_after:.0f}% predicted success.",
        ]

        if self.gaps:
            priority = ", ".join(
                f"cluster {g.cluster_id} ({g.needed} eps)"
                for g in self.gaps[:3]
            )
            lines.append(f"  Priority: {priority}")

        lines.append(_THICK)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "n_episodes": self.n_episodes,
            "policy_family": self.policy_family,
            "current_success": self.current_success,
            "target_success": self.target_success,
            "n_clusters": self.n_clusters,
            "total_needed": self.total_needed,
            "estimated_success_after": self.estimated_success_after,
            "gaps": [
                {
                    "cluster_id": g.cluster_id,
                    "current_episodes": g.current_episodes,
                    "target_episodes": g.target_episodes,
                    "needed": g.needed,
                    "action_dim_ranges": {
                        k: list(v) for k, v in g.action_dim_ranges.items()
                    },
                    "estimated_success_delta": g.estimated_success_delta,
                    "coverage_fraction": g.coverage_fraction,
                }
                for g in self.gaps
            ],
        }


# ── core analysis ──────────────────────────────────────────────────────────────

def analyze_coverage_gap(
    batch: EpisodeBatch,
    report: DiagnosticReport,
    target_success: float = 0.80,
    policy_family: Optional[str] = None,
    n_clusters: int = 8,
) -> GapAnalysisResult:
    """
    Compute the action-space coverage gap for a dataset.

    Parameters
    ----------
    batch          : loaded EpisodeBatch
    report         : DiagnosticReport from Pipeline.run()
    target_success : desired policy success rate (0.0–1.0)
    policy_family  : policy family for prediction weighting
    n_clusters     : number of action-space clusters (k-means k)
    """
    from calibra.predict import predict_outcome

    pred = predict_outcome(report, policy_family=policy_family)
    current_success = pred["predicted_score"]  # 0–100

    # Build per-episode feature matrix: (n_episodes, action_dim * 2)
    features, raw_actions_list, action_dim = _build_episode_features(batch)

    n_ep = len(batch.episodes)
    if n_ep < 2 or features.shape[0] < 2:
        return GapAnalysisResult(
            dataset_name=report.dataset_name,
            n_episodes=n_ep,
            policy_family=policy_family or "generic",
            current_success=current_success,
            target_success=target_success * 100,
            n_clusters=n_clusters,
            gaps=[],
            total_needed=0,
            estimated_success_after=current_success,
        )

    # K-means cluster the feature space
    n_clusters = min(n_clusters, n_ep)
    labels, centers = _kmeans(features, n_clusters, seed=42)

    # Coverage stats per cluster
    cluster_counts = np.bincount(labels, minlength=n_clusters)
    target_per_cluster = max(1, round(n_ep / n_clusters))

    # Estimate entropy improvement from filling each gap
    # Uses the action entropy weight from the predict rubric
    current_entropy = _marginal_entropy(features)
    gaps: list[CoverageGap] = []

    for c in range(n_clusters):
        count = int(cluster_counts[c])
        needed = max(0, target_per_cluster - count)
        if needed == 0:
            continue

        # Indices of episodes in this cluster
        member_mask = labels == c
        if member_mask.sum() > 0:
            cluster_actions = np.vstack([
                raw_actions_list[i]
                for i, m in enumerate(member_mask) if m
            ])
        else:
            cluster_actions = centers[c:c+1]  # use centroid as placeholder

        # Dim ranges: mean ± 2*std of cluster members
        dim_ranges: dict[str, tuple[float, float]] = {}
        if cluster_actions.ndim == 1:
            cluster_actions = cluster_actions[:, np.newaxis]
        for d in range(min(action_dim, cluster_actions.shape[1])):
            col = cluster_actions[:, d]
            mu, sigma = float(col.mean()), float(col.std()) + 1e-6
            dim_ranges[f"action_dim_{d}"] = (
                round(mu - 2 * sigma, 4),
                round(mu + 2 * sigma, 4),
            )

        # Estimate entropy gain from adding `needed` episodes to this cluster
        entropy_gain = _entropy_gain_estimate(features, labels, c, needed)
        success_delta = entropy_gain * _ENTROPY_WEIGHT_PER_BIT

        gaps.append(CoverageGap(
            cluster_id=c,
            current_episodes=count,
            target_episodes=target_per_cluster,
            needed=needed,
            action_dim_ranges=dim_ranges,
            estimated_success_delta=round(success_delta, 2),
            coverage_fraction=round(count / max(target_per_cluster, 1), 3),
        ))

    # Sort by estimated impact (descending)
    gaps.sort(key=lambda g: g.estimated_success_delta, reverse=True)

    total_needed = sum(g.needed for g in gaps)
    total_delta = min(
        sum(g.estimated_success_delta for g in gaps),
        target_success * 100 - current_success,
    )
    estimated_after = min(current_success + total_delta, 100.0)

    return GapAnalysisResult(
        dataset_name=report.dataset_name,
        n_episodes=n_ep,
        policy_family=policy_family or "generic",
        current_success=current_success,
        target_success=target_success * 100,
        n_clusters=n_clusters,
        gaps=gaps,
        total_needed=total_needed,
        estimated_success_after=round(estimated_after, 1),
    )


# ── feature helpers ────────────────────────────────────────────────────────────

def _build_episode_features(batch: EpisodeBatch) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    Per-episode feature vector: action mean + std per dimension.
    Returns (features (N, 2D), list of raw action arrays, action_dim).
    """
    raw_list: list[np.ndarray] = []
    rows: list[np.ndarray] = []

    for ep in batch.episodes:
        acts = ep.actions
        if acts is None or acts.size == 0:
            continue
        if acts.ndim == 1:
            acts = acts[:, np.newaxis]
        raw_list.append(acts)
        mu = np.mean(acts, axis=0)
        sigma = np.std(acts, axis=0)
        rows.append(np.concatenate([mu, sigma]))

    if not rows:
        return np.zeros((0, 2)), [], 0

    mat = np.stack(rows, axis=0).astype(np.float64)
    action_dim = mat.shape[1] // 2

    # Normalise
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    span = col_max - col_min
    span[span == 0] = 1.0
    mat = (mat - col_min) / span

    return mat, raw_list, action_dim


def _kmeans(X: np.ndarray, k: int, seed: int = 42, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Simple Lloyd k-means, no external dependency."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        # Assign
        diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=-1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)

    return labels, centers


def _marginal_entropy(X: np.ndarray, n_bins: int = 20) -> float:
    """Average per-column Shannon entropy (bits)."""
    if X.shape[0] < 2:
        return 0.0
    D = X.shape[1]
    total = 0.0
    for d in range(D):
        counts, _ = np.histogram(X[:, d], bins=n_bins, range=(0.0, 1.0))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        total += float(-np.sum(probs * np.log2(probs + 1e-12)))
    return total / D


def _entropy_gain_estimate(
    features: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    n_to_add: int,
    n_bins: int = 20,
) -> float:
    """
    Estimate the entropy gain (bits/dim) from adding `n_to_add` synthetic
    episodes uniformly within cluster `cluster_id`.
    """
    mask = labels == cluster_id
    if mask.sum() == 0:
        return 0.0

    cluster_features = features[mask]
    col_min = cluster_features.min(axis=0)
    col_max = cluster_features.max(axis=0)
    span = col_max - col_min
    span[span == 0] = 0.05  # give zero-variance dims a small range

    rng = np.random.default_rng(seed=cluster_id)
    synthetic = col_min + rng.random((n_to_add, features.shape[1])) * span

    augmented = np.vstack([features, synthetic])

    old_h = _marginal_entropy(features, n_bins)
    new_h = _marginal_entropy(augmented, n_bins)
    return max(0.0, new_h - old_h)


# ── CLI ────────────────────────────────────────────────────────────────────────

def run_gap(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra gap",
        description=(
            "Analyse action-space coverage gaps and output a targeted collection brief. "
            "Tells you exactly how many episodes of each type to collect next."
        ),
    )
    p.add_argument("path", help="Dataset path or HuggingFace Hub ID")
    p.add_argument(
        "--policy", "-p", metavar="FAMILY", default=None,
        help="Target policy family (diffusion, act, gr00t)",
    )
    p.add_argument(
        "--target-success", "-t", type=float, default=0.80, metavar="RATE",
        help="Target policy success rate 0.0–1.0 (default: 0.80)",
    )
    p.add_argument(
        "--clusters", "-k", type=int, default=8, metavar="N",
        help="Number of action-space clusters (default: 8)",
    )
    p.add_argument(
        "--format", "-f",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force format adapter",
    )
    p.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    args = p.parse_args(argv)

    if not 0.0 < args.target_success <= 1.0:
        print("error: --target-success must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {args.path!r} ...")

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    try:
        from calibra.ingestion.registry import load
        batch = load(args.path, reader=reader)
        report = Pipeline().run(batch, policy_family=args.policy)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {report.n_episodes} episodes  ·  {report.n_samples:,} steps")
    log("Computing coverage gaps ...")

    result = analyze_coverage_gap(
        batch,
        report,
        target_success=args.target_success,
        policy_family=args.policy,
        n_clusters=args.clusters,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
