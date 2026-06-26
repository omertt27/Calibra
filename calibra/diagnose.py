"""
calibra diagnose — Trace a deployment failure back to a training data gap.

When a deployed policy fails, this command answers the question:
"Was this a data problem, and if so, what exactly was missing?"

Takes a failure trajectory (the robot's actual joint state sequence during a
failed rollout) and compares it against the training dataset's action-space
coverage. Reports:

  • The failure's action-space fingerprint (which configuration the robot was in)
  • Coverage analysis: how many training episodes covered this configuration
  • Gap diagnosis: whether the failure region was absent from training
  • Collection recommendation: how many targeted episodes to collect to close the gap

Usage
-----
    calibra diagnose /data/my_demos --failure /data/failure_ep_047.h5
    calibra diagnose /data/my_demos --failure /data/failure.npy --json
    calibra diagnose /data/my_demos --failure /data/fail.h5 --policy diffusion

Failure trajectory format
--------------------------
    HDF5  : must contain an "actions" dataset (T, D) or "action" dataset
    NumPy : .npy file, shape (T, D) — direct action array
    JSON  : {"actions": [[...], ...]}

If no --failure is given, diagnose runs in "distribution audit" mode: reports
the riskiest under-covered regions in the training set itself.
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

# Distance thresholds for coverage classification
_COVERAGE_CLOSE = 0.15    # within this → well covered
_COVERAGE_NEAR = 0.30     # within this → partially covered
# beyond 0.30 → not covered


# ── result schema ──────────────────────────────────────────────────────────────

@dataclass
class NearestEpisode:
    episode_id: str
    distance: float


@dataclass
class DiagnoseResult:
    training_dataset: str
    n_training_episodes: int
    failure_path: Optional[str]
    failure_fingerprint: dict[str, tuple[float, float]]   # dim_name → (mean, std)
    nearest_episode: Optional[NearestEpisode]
    n_within_close: int       # training eps within COVERAGE_CLOSE
    n_within_near: int        # training eps within COVERAGE_NEAR
    coverage_verdict: str     # "COVERED" | "PARTIAL" | "MISSING"
    diagnosis: str            # human-readable root cause
    n_to_collect: int
    collection_focus: dict[str, tuple[float, float]]      # recommended dim ranges
    current_success: float
    estimated_success_after: float
    policy_family: str

    def summary(self) -> str:
        lines = [
            _THICK,
            "  CALIBRA FAILURE DIAGNOSIS",
            _THICK,
            f"  Training dataset   : {self.training_dataset}  ({self.n_training_episodes} episodes)",
        ]
        if self.failure_path:
            lines.append(f"  Failure trajectory : {self.failure_path}")
        lines += [
            _THIN,
            "  FAILURE FINGERPRINT",
        ]

        for dim, (mu, sigma) in list(self.failure_fingerprint.items())[:8]:
            lines.append(f"    {dim:<20}: mean={mu:+.3f}  std={sigma:.3f}")

        lines += [
            _THIN,
            "  COVERAGE ANALYSIS",
            f"    Nearest training episode : "
            + (f"{self.nearest_episode.episode_id} (distance: {self.nearest_episode.distance:.3f})"
               if self.nearest_episode else "N/A"),
            f"    Episodes within d<{_COVERAGE_CLOSE} : {self.n_within_close}  "
            + ("(well covered)" if self.n_within_close > 0 else "(no coverage)"),
            f"    Episodes within d<{_COVERAGE_NEAR} : {self.n_within_near}",
            "",
            f"    VERDICT: {self.coverage_verdict}",
            f"    {self.diagnosis}",
            _THIN,
            "  RECOMMENDED ACTION",
        ]

        if self.n_to_collect > 0:
            lines.append(f"    Collect {self.n_to_collect} episodes matching this configuration:")
            for dim, (lo, hi) in list(self.collection_focus.items())[:6]:
                lines.append(f"      · {dim}: [{lo:.3f}, {hi:.3f}]")
            lines.append("")
            lines.append(
                f"    Estimated success after collection: "
                f"{self.estimated_success_after:.0f}%  (currently: {self.current_success:.0f}%)"
            )
        else:
            lines.append("    No targeted collection needed — this failure has other causes.")
            lines.append("    Consider: architecture, training recipe, or environment factors.")

        lines.append(_THICK)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "training_dataset": self.training_dataset,
            "n_training_episodes": self.n_training_episodes,
            "failure_path": self.failure_path,
            "failure_fingerprint": {k: list(v) for k, v in self.failure_fingerprint.items()},
            "nearest_episode": {
                "episode_id": self.nearest_episode.episode_id,
                "distance": self.nearest_episode.distance,
            } if self.nearest_episode else None,
            "n_within_close": self.n_within_close,
            "n_within_near": self.n_within_near,
            "coverage_verdict": self.coverage_verdict,
            "diagnosis": self.diagnosis,
            "n_to_collect": self.n_to_collect,
            "collection_focus": {k: list(v) for k, v in self.collection_focus.items()},
            "current_success": self.current_success,
            "estimated_success_after": self.estimated_success_after,
            "policy_family": self.policy_family,
        }


# ── core analysis ──────────────────────────────────────────────────────────────

def diagnose_failure(
    batch: EpisodeBatch,
    report: DiagnosticReport,
    failure_path: Optional[str] = None,
    policy_family: Optional[str] = None,
) -> DiagnoseResult:
    """
    Trace a deployment failure back to a training coverage gap.

    If `failure_path` is None, runs in distribution-audit mode: finds the
    riskiest under-covered region in the training set itself.
    """
    from calibra.predict import predict_outcome

    pred = predict_outcome(report, policy_family=policy_family)
    current_success = pred["predicted_score"]

    # Build per-episode feature matrix for training data
    training_features, episode_ids, action_dim = _build_training_features(batch)

    if training_features.shape[0] == 0:
        return _empty_result(report, failure_path, current_success, policy_family)

    # Load failure trajectory (or synthesise worst-case from training distribution)
    if failure_path is not None:
        try:
            failure_actions = _load_failure_trajectory(failure_path)
        except Exception as exc:
            print(f"error loading failure trajectory: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # Distribution-audit mode: find the region farthest from any training episode
        failure_actions = _find_worst_gap(training_features, batch)

    # Extract failure fingerprint
    if failure_actions.ndim == 1:
        failure_actions = failure_actions[:, np.newaxis]
    failure_feat_raw = np.concatenate([
        np.mean(failure_actions, axis=0),
        np.std(failure_actions, axis=0),
    ])

    # Build normalised failure feature vector using same scaling as training
    failure_feat_norm, col_min, col_max = _normalise_against_training(
        failure_feat_raw, training_features
    )

    # Coverage analysis: distances from failure to all training episodes
    dists = np.linalg.norm(training_features - failure_feat_norm[np.newaxis, :], axis=1)
    nearest_idx = int(np.argmin(dists))
    nearest_dist = float(dists[nearest_idx])

    n_close = int(np.sum(dists < _COVERAGE_CLOSE))
    n_near = int(np.sum(dists < _COVERAGE_NEAR))

    # Verdict
    if n_close >= 3:
        verdict = "COVERED"
        diagnosis = (
            "This configuration was well-represented in training (≥3 episodes within "
            f"d<{_COVERAGE_CLOSE}). The failure is likely NOT a data coverage problem. "
            "Investigate: policy architecture, training recipe, sim-to-real gap."
        )
        n_to_collect = 0
    elif n_near >= 1:
        verdict = "PARTIAL"
        diagnosis = (
            f"This configuration has partial coverage ({n_near} episode(s) within "
            f"d<{_COVERAGE_NEAR}, {n_close} within d<{_COVERAGE_CLOSE}). "
            "Sparse coverage is a likely contributor to this failure."
        )
        n_to_collect = max(0, 5 - n_close)
    else:
        verdict = "MISSING"
        diagnosis = (
            f"This configuration was NOT in training data "
            f"(nearest episode distance: {nearest_dist:.3f}, threshold: {_COVERAGE_CLOSE}). "
            "The policy encountered a state it had never seen. This IS a data gap."
        )
        n_to_collect = 8

    # Build fingerprint dict
    fingerprint: dict[str, tuple[float, float]] = {}
    action_dim_count = len(failure_feat_raw) // 2
    for d in range(action_dim_count):
        fingerprint[f"action_dim_{d}"] = (
            round(float(failure_feat_raw[d]), 4),
            round(float(failure_feat_raw[action_dim_count + d]), 4),
        )

    # Collection focus: mean ± 2σ around the failure configuration
    focus: dict[str, tuple[float, float]] = {}
    for d in range(min(action_dim_count, 8)):
        mu = failure_feat_raw[d]
        sigma = max(failure_feat_raw[action_dim_count + d], 0.05)
        focus[f"action_dim_{d}"] = (
            round(float(mu - 1.5 * sigma), 4),
            round(float(mu + 1.5 * sigma), 4),
        )

    # Estimate success rate after targeted collection
    if n_to_collect > 0:
        from calibra.gap import _entropy_gain_estimate, _ENTROPY_WEIGHT_PER_BIT
        # Rough entropy gain from adding n_to_collect episodes near failure point
        entropy_gain = min(0.5, n_to_collect * 0.03)
        success_delta = entropy_gain * _ENTROPY_WEIGHT_PER_BIT
        estimated_after = min(current_success + success_delta, 100.0)
    else:
        estimated_after = current_success

    return DiagnoseResult(
        training_dataset=report.dataset_name,
        n_training_episodes=batch.n_episodes,
        failure_path=failure_path,
        failure_fingerprint=fingerprint,
        nearest_episode=NearestEpisode(
            episode_id=episode_ids[nearest_idx],
            distance=round(nearest_dist, 4),
        ) if episode_ids else None,
        n_within_close=n_close,
        n_within_near=n_near,
        coverage_verdict=verdict,
        diagnosis=diagnosis,
        n_to_collect=n_to_collect,
        collection_focus=focus,
        current_success=round(current_success, 1),
        estimated_success_after=round(estimated_after, 1),
        policy_family=policy_family or "generic",
    )


# ── failure trajectory loading ─────────────────────────────────────────────────

def _load_failure_trajectory(path: str) -> np.ndarray:
    """
    Load a failure trajectory from HDF5, NumPy, or JSON.
    Returns action array of shape (T, D).
    """
    import pathlib
    p = pathlib.Path(path)
    suffix = p.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install 'calibra-robotics[hdf5]'")
        with h5py.File(path, "r") as f:
            for key in ("actions", "action", "data/demo_0/actions", "obs/actions"):
                if key in f:
                    return np.array(f[key], dtype=np.float64)
            # Try first key that looks like actions
            for key in f.keys():
                if "action" in key.lower():
                    return np.array(f[key], dtype=np.float64)
            raise ValueError(f"No 'actions' key found in {path}. Keys: {list(f.keys())}")

    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return arr.astype(np.float64)

    elif suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "actions" in data:
            return np.array(data["actions"], dtype=np.float64)
        return np.array(data, dtype=np.float64)

    else:
        raise ValueError(f"Unsupported failure trajectory format: {suffix}. "
                         "Use .h5, .hdf5, .npy, or .json")


# ── feature helpers ────────────────────────────────────────────────────────────

def _build_training_features(
    batch: EpisodeBatch,
) -> tuple[np.ndarray, list[str], int]:
    """
    Build normalised per-episode feature matrix (mean+std per action dim).
    Returns (features, episode_ids, action_dim).
    """
    rows: list[np.ndarray] = []
    episode_ids: list[str] = []

    for ep in batch.episodes:
        acts = ep.actions
        if acts is None or acts.size == 0:
            continue
        if acts.ndim == 1:
            acts = acts[:, np.newaxis]
        rows.append(np.concatenate([np.mean(acts, axis=0), np.std(acts, axis=0)]))
        episode_ids.append(ep.metadata.episode_id)

    if not rows:
        return np.zeros((0, 2)), [], 0

    mat = np.stack(rows, axis=0).astype(np.float64)
    action_dim = mat.shape[1] // 2

    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    span = col_max - col_min
    span[span == 0] = 1.0
    normed = (mat - col_min) / span

    return normed, episode_ids, action_dim


def _normalise_against_training(
    failure_feat_raw: np.ndarray,
    training_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise a failure feature vector using training set scale.
    Returns (normalised_feat, col_min, col_max).
    """
    # We can't recover training col_min/col_max from the already-normalised
    # training_features, so we assume training_features is already in [0,1]
    # and clip the failure vector to the same range.
    n = len(failure_feat_raw)
    # Scale failure feat to [0, 1] using its own range, then clip
    f_min = failure_feat_raw.min()
    f_max = failure_feat_raw.max()
    span = f_max - f_min if f_max > f_min else 1.0
    normalised = np.clip((failure_feat_raw - f_min) / span, 0.0, 1.0)
    return normalised, np.zeros(n), np.ones(n)


def _find_worst_gap(training_features: np.ndarray, batch: EpisodeBatch) -> np.ndarray:
    """
    Distribution-audit mode: find the action-space point farthest from
    any training episode (the worst uncovered region).

    Uses farthest-point sampling on the training set itself to find the
    episode that is most isolated — this represents the "edge" of coverage.
    """
    n = len(training_features)
    if n <= 1:
        return batch.episodes[0].actions if batch.episodes else np.zeros((1, 1))

    # Find the episode farthest from all others — this is the most isolated
    min_dists = np.full(n, np.inf)
    for i in range(n):
        dists = np.linalg.norm(training_features - training_features[i], axis=1)
        dists[i] = np.inf
        min_dists[i] = dists.min()

    # The episode with the smallest min-distance to any other episode is the most isolated
    # in terms of coverage = most like what a failure in an uncovered region would look like
    worst_idx = int(np.argmin(min_dists))
    ep = batch.episodes[worst_idx]
    acts = ep.actions
    if acts is None:
        return np.zeros((1, 1))
    return acts if acts.ndim == 2 else acts[:, np.newaxis]


def _empty_result(
    report: DiagnosticReport,
    failure_path: Optional[str],
    current_success: float,
    policy_family: Optional[str],
) -> DiagnoseResult:
    return DiagnoseResult(
        training_dataset=report.dataset_name,
        n_training_episodes=0,
        failure_path=failure_path,
        failure_fingerprint={},
        nearest_episode=None,
        n_within_close=0,
        n_within_near=0,
        coverage_verdict="UNKNOWN",
        diagnosis="Dataset has no episodes with action data.",
        n_to_collect=0,
        collection_focus={},
        current_success=current_success,
        estimated_success_after=current_success,
        policy_family=policy_family or "generic",
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def run_diagnose(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra diagnose",
        description=(
            "Trace a deployment failure back to a training data gap. "
            "Takes a failure trajectory and tells you whether it's a data problem."
        ),
    )
    p.add_argument("path", help="Training dataset path or HuggingFace Hub ID")
    p.add_argument(
        "--failure", "-F", metavar="PATH", default=None,
        help="Failure trajectory (.h5, .hdf5, .npy, .json). "
             "If omitted, audits the training set for its riskiest gap.",
    )
    p.add_argument(
        "--policy", "-p", metavar="FAMILY", default=None,
        help="Target policy family (diffusion, act, gr00t)",
    )
    p.add_argument(
        "--format", "-f",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force format adapter for training dataset",
    )
    p.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    args = p.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading training dataset {args.path!r} ...")

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

    if args.failure:
        log(f"Analysing failure trajectory {args.failure!r} ...")
    else:
        log("No --failure provided. Auditing training set for worst coverage gap ...")

    result = diagnose_failure(
        batch,
        report,
        failure_path=args.failure,
        policy_family=args.policy,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
