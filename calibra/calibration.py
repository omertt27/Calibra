"""
Calibra Detector Calibration Registry

Tracks empirically measured benign firing rates — the fraction of episodes each
detector flags on known-clean datasets (no synthetic corruption, episodes treated
as ground-truth valid).

Design principles
-----------------
1. Every profile is tied to a specific dataset, task family, detector version,
   and configuration hash. A PushT baseline is never silently applied to ALOHA.

2. When no matching profile exists, `lookup()` returns None and the caller should
   report "baseline unavailable", not substitute a generic number.

3. Profiles ship with Calibra as initial estimates, but the benchmark script
   (experiments/benign_firing_rate_benchmark.py) produces empirically-measured
   values that can be loaded via CalibrationRegistry.load_json().

4. Confidence intervals are first-class — callers can display "(1.4%, 6.8%)"
   alongside the point estimate so users understand the uncertainty.

Usage
-----
    from calibra.calibration import DEFAULT_REGISTRY

    rate, source = DEFAULT_REGISTRY.benign_firing_rate(
        detector="jitter_cv",
        dataset="lerobot/pusht",
    )
    if rate is None:
        print("baseline unavailable for this detector/dataset combination")
    else:
        print(f"benign firing rate: {rate:.1%} (from {source})")

Updated by: experiments/benign_firing_rate_benchmark.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class CalibrationProfile:
    """
    One empirical measurement: how often a specific detector fires on a specific
    known-clean dataset.

    All fields that could bias interpretation are part of the identity:
    dataset, task family, detector version, and config hash.

    `provenance` distinguishes hand-estimated built-in values from profiles
    measured by the benchmark script (experiments/benign_firing_rate_benchmark.py).
    Always show "(initial estimate)" when provenance == "builtin_estimate" so
    users know whether the number came from an actual benchmark run.
    """

    dataset: str  # "lerobot/pusht"
    task_family: str  # "pusht" | "aloha" | "so100" | ...
    detector: str  # "jitter_cv" | "dropout_rate" | "spike_rate" | ...
    detector_version: str  # calibra version used during measurement
    config_hash: Optional[str]  # sha256[:16] of detector config; None = default config
    n_episodes: int
    n_flagged: int
    firing_rate: float  # n_flagged / n_episodes
    ci_lower: Optional[float] = None  # 95% Wilson score CI lower bound
    ci_upper: Optional[float] = None  # 95% Wilson score CI upper bound
    dataset_version: Optional[str] = None  # HF dataset revision / git SHA
    measured_at: str = ""  # ISO date of measurement (YYYY-MM-DD)
    provenance: Literal["builtin_estimate", "benchmark_run"] = "builtin_estimate"
    notes: str = ""

    def source_description(self) -> str:
        """Short string identifying this baseline for display, including provenance tag."""
        tag = " — initial estimate" if self.provenance == "builtin_estimate" else ""
        return f"{self.dataset} (n={self.n_episodes}, v{self.detector_version}{tag})"


class CalibrationRegistry:
    """
    In-memory registry of CalibrationProfile entries.

    Lookup priority (most specific wins):
      1. Exact dataset match
      2. Task family match
      3. No match → returns None
    """

    def __init__(self, profiles: list[CalibrationProfile] | None = None) -> None:
        self._profiles: list[CalibrationProfile] = list(profiles or [])

    def add(self, profile: CalibrationProfile) -> None:
        self._profiles.append(profile)

    def all_profiles(self) -> list[CalibrationProfile]:
        return list(self._profiles)

    def lookup(
        self,
        detector: str,
        dataset: Optional[str] = None,
        task_family: Optional[str] = None,
    ) -> Optional[CalibrationProfile]:
        """
        Find the most relevant CalibrationProfile for a detector.

        Priority: exact dataset match > task_family match > no match.
        Returns None when no baseline is available — callers must not
        substitute a generic number in that case.
        """
        candidates = [p for p in self._profiles if p.detector == detector]
        if not candidates:
            return None

        if dataset:
            exact = [p for p in candidates if p.dataset == dataset]
            if exact:
                return max(exact, key=lambda p: p.n_episodes)

        if task_family:
            family_match = [p for p in candidates if p.task_family == task_family]
            if family_match:
                return max(family_match, key=lambda p: p.n_episodes)

        return None

    def benign_firing_rate(
        self,
        detector: str,
        dataset: Optional[str] = None,
        task_family: Optional[str] = None,
    ) -> tuple[Optional[float], Optional[str]]:
        """
        Return (firing_rate, source_description) or (None, None) when no
        baseline is available.

        The source_description lets the caller tell the user exactly which
        dataset the baseline came from, so they can judge relevance.
        """
        p = self.lookup(detector, dataset=dataset, task_family=task_family)
        if p is None:
            return None, None
        return p.firing_rate, p.source_description()

    def calibration_context(
        self,
        detector: str,
        observed_fraction: float,
        dataset: Optional[str] = None,
        task_family: Optional[str] = None,
    ) -> str:
        """
        One-line context string comparing an observed rate to the known baseline.

        Examples:
          "7.8% flag rate (clean baseline 3.2% from lerobot/pusht — 2.4× above)"
          "1.1% flag rate (clean baseline 1.4% from lerobot/pusht — within range)"
          "7.8% flag rate (no baseline available for this detector/task)"
        """
        pct = f"{observed_fraction:.1%}"
        p = self.lookup(detector, dataset=dataset, task_family=task_family)
        if p is None:
            return f"{pct} flag rate (no baseline available for this detector/task)"
        rate = p.firing_rate
        source = p.source_description()
        baseline_pct = f"{rate:.1%}"
        if rate < 1e-9:
            return f"{pct} flag rate (baseline: {baseline_pct} from {source})"
        ratio = observed_fraction / rate
        if abs(observed_fraction - rate) < 0.005:
            comparison = "within range"
        elif ratio >= 2.0:
            comparison = f"{ratio:.1f}× above baseline"
        elif ratio <= 0.5:
            comparison = "below baseline"
        else:
            comparison = "near baseline"
        return f"{pct} flag rate (clean baseline {baseline_pct} from {source} — {comparison})"

    def save_json(self, path: str) -> None:
        """Write all profiles to a JSON file."""
        records = [
            {
                "dataset": p.dataset,
                "task_family": p.task_family,
                "detector": p.detector,
                "detector_version": p.detector_version,
                "config_hash": p.config_hash,
                "n_episodes": p.n_episodes,
                "n_flagged": p.n_flagged,
                "firing_rate": p.firing_rate,
                "ci_lower": p.ci_lower,
                "ci_upper": p.ci_upper,
                "dataset_version": p.dataset_version,
                "measured_at": p.measured_at,
                "notes": p.notes,
            }
            for p in self._profiles
        ]
        Path(path).write_text(json.dumps({"profiles": records}, indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str) -> "CalibrationRegistry":
        """
        Load profiles from a JSON file (output of the benchmark script).

        Extra fields in the JSON (e.g. corrupted_episode_detection_rate recorded
        by the benchmark) are silently ignored — CalibrationProfile stores only
        benign firing rate data.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in dataclass_fields(CalibrationProfile)}
        profiles = [
            CalibrationProfile(**{k: v for k, v in r.items() if k in known})
            for r in data["profiles"]
        ]
        return CalibrationRegistry(profiles)

    def merge(self, other: "CalibrationRegistry") -> "CalibrationRegistry":
        """Return a new registry combining profiles from both, other takes precedence."""
        merged = {
            (p.detector, p.dataset, p.task_family): p for p in self._profiles
        }
        for p in other._profiles:
            merged[(p.detector, p.dataset, p.task_family)] = p
        return CalibrationRegistry(list(merged.values()))


# ── built-in profiles ─────────────────────────────────────────────────────────
# Measured by experiments/benign_firing_rate_benchmark.py on 2026-09-10.
# Within-dataset MAD-based outlier rates: even clean datasets have tail episodes.
# Thresholds: jitter_cv/dropout_rate/spike_rate/vel_disc_rate at 3.0x MAD;
#             ldlj at 4.0x MAD (higher natural variance across tasks).
# Re-run the benchmark script when detector thresholds or logic change.

_BUILTIN_PROFILES: list[CalibrationProfile] = [
    # ── lerobot/pusht (2D pushing, n=206 episodes) ────────────────────────────
    CalibrationProfile(
        dataset="lerobot/pusht",
        task_family="pusht",
        detector="jitter_cv",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=206,
        n_flagged=14,
        firing_rate=0.068,
        ci_lower=0.0409,
        ci_upper=0.1108,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x.",
    ),
    CalibrationProfile(
        dataset="lerobot/pusht",
        task_family="pusht",
        detector="dropout_rate",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=206,
        n_flagged=0,
        firing_rate=0.0,
        ci_lower=0.0,
        ci_upper=0.0183,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x. PushT timestamps are synthetic/uniform.",
    ),
    CalibrationProfile(
        dataset="lerobot/pusht",
        task_family="pusht",
        detector="spike_rate",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=206,
        n_flagged=3,
        firing_rate=0.0146,
        ci_lower=0.005,
        ci_upper=0.0419,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x.",
    ),
    CalibrationProfile(
        dataset="lerobot/pusht",
        task_family="pusht",
        detector="vel_disc_rate",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=206,
        n_flagged=8,
        firing_rate=0.0388,
        ci_lower=0.0198,
        ci_upper=0.0747,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x.",
    ),
    CalibrationProfile(
        dataset="lerobot/pusht",
        task_family="pusht",
        detector="ldlj",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=206,
        n_flagged=0,
        firing_rate=0.0,
        ci_lower=0.0,
        ci_upper=0.0183,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 4.0x. PushT 2D planar task produces uniformly smooth LDLJ.",
    ),
    # ── lerobot/aloha_sim_insertion_scripted (bimanual sim, n=50 episodes) ─────
    # n=50 makes CIs wide. Treat as task-family signal, not a point estimate.
    CalibrationProfile(
        dataset="lerobot/aloha_sim_insertion_scripted",
        task_family="aloha",
        detector="jitter_cv",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=50,
        n_flagged=0,
        firing_rate=0.0,
        ci_lower=0.0,
        ci_upper=0.0714,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x. Scripted sim data has highly uniform timestamps.",
    ),
    CalibrationProfile(
        dataset="lerobot/aloha_sim_insertion_scripted",
        task_family="aloha",
        detector="dropout_rate",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=50,
        n_flagged=0,
        firing_rate=0.0,
        ci_lower=0.0,
        ci_upper=0.0714,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x. No dropout in scripted sim data.",
    ),
    CalibrationProfile(
        dataset="lerobot/aloha_sim_insertion_scripted",
        task_family="aloha",
        detector="spike_rate",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=50,
        n_flagged=4,
        firing_rate=0.08,
        ci_lower=0.0315,
        ci_upper=0.1884,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x. Scripted waypoint transitions create natural jerk.",
    ),
    CalibrationProfile(
        dataset="lerobot/aloha_sim_insertion_scripted",
        task_family="aloha",
        detector="vel_disc_rate",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=50,
        n_flagged=0,
        firing_rate=0.0,
        ci_lower=0.0,
        ci_upper=0.0714,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 3.0x.",
    ),
    CalibrationProfile(
        dataset="lerobot/aloha_sim_insertion_scripted",
        task_family="aloha",
        detector="ldlj",
        detector_version="0.10.0",
        config_hash=None,
        n_episodes=50,
        n_flagged=0,
        firing_rate=0.0,
        ci_lower=0.0,
        ci_upper=0.0714,
        measured_at="2026-09-10",
        provenance="benchmark_run",
        notes="Within-dataset MAD outlier at 4.0x.",
    ),
]

DEFAULT_REGISTRY: CalibrationRegistry = CalibrationRegistry(_BUILTIN_PROFILES)
