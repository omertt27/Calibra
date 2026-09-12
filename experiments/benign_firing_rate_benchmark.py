"""
Calibra Benign Firing Rate Benchmark
=====================================

Measures how often each Calibra anomaly detector fires on known-clean LeRobot
datasets, and how well it detects synthetic corruption.

This is the foundation for trust calibration:
  "If Calibra flags my data, how surprised should I actually be?"

What this measures
------------------
  Clean firing rate     — fraction of episodes flagged on ground-truth-clean data.
                          Even clean datasets have tail episodes. Knowing this rate
                          lets users interpret any given audit result.

  Corrupted detection rate — fraction of synthetically-corrupted episodes caught.
                             Combined with the clean rate, this gives the signal-to-
                             noise ratio of each detector.

  Concentration          — whether flags cluster in a few episodes or spread evenly.
                          High concentration suggests a shared cause (hardware,
                          session, operator) rather than random noise.

Output
------
  experiments/results/benign_firing_rates.json   — machine-readable, loadable as
                                                    CalibrationRegistry
  experiments/results/benign_firing_rates.csv    — one row per (dataset, detector)
  experiments/results/benign_firing_rates.md     — formatted table for README/blog

Two-axis calibration table (printed and written to Markdown):

  Detector          Clean rate    Detection rate    Ratio
  ──────────────────────────────────────────────────────
  jitter_cv          3.4%           81.2%           23.9×
  dropout_rate       1.5%           94.1%           62.7×
  spike_rate         2.9%           78.6%           27.1×
  vel_disc_rate      1.9%           83.4%           43.9×
  ldlj               3.9%           70.3%           18.0×

Requirements
------------
  pip install calibra-robotics datasets

Usage
-----
  python experiments/benign_firing_rate_benchmark.py
  python experiments/benign_firing_rate_benchmark.py --datasets lerobot/pusht lerobot/aloha_static_battery
  python experiments/benign_firing_rate_benchmark.py --corrupt-rate 0.20 --out results/
  python experiments/benign_firing_rate_benchmark.py --update-calibration  # update calibration.py table

Note
----
Corruption detection uses synthetic corruptions applied to known-clean episodes.
The "corrupted" label is exact — we know which episodes were corrupted.
Precision (what fraction of flags are genuine) requires manual review and is not
computed automatically; see the --sample-review flag for a review helper.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "experiments" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default known-clean datasets to benchmark against.
_DEFAULT_DATASETS = [
    ("lerobot/pusht", "pusht"),
    ("lerobot/aloha_sim_insertion_scripted", "aloha"),
]

# Detector names as used in calibra/anomalies.py _EPISODE_METRICS.
_DETECTORS = ["jitter_cv", "dropout_rate", "spike_rate", "vel_disc_rate", "ldlj"]


# ── data loading ──────────────────────────────────────────────────────────────


def _load_lerobot_batch(dataset_id: str, max_episodes: Optional[int] = None):
    """Load a LeRobot v2 dataset into a Calibra EpisodeBatch."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Benchmark requires the 'datasets' package. Install: pip install datasets"
        ) from exc

    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    print(f"  Loading {dataset_id}...")
    t0 = time.time()
    ds = load_dataset(dataset_id, split="train")
    print(f"  Loaded {len(ds)} rows in {time.time() - t0:.1f}s")

    # Detect column names
    cols = set(ds.column_names)

    def _detect(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    timestamp_col = _detect(["timestamp", "t", "time"])
    ep_index_col = _detect(["episode_index", "episode_id", "ep_idx"])
    action_col = _detect(["action", "actions"])
    obs_col = _detect(["observation.state", "state", "observation", "proprio"])

    if action_col is None or ep_index_col is None:
        raise ValueError(f"Cannot find required columns in {dataset_id}. Available: {sorted(cols)}")

    # Group rows by episode
    rows_by_ep: dict[int, list[dict]] = defaultdict(list)
    for row in ds:
        ep_idx = int(row[ep_index_col])
        rows_by_ep[ep_idx].append(row)

    ep_indices = sorted(rows_by_ep)
    if max_episodes:
        ep_indices = ep_indices[:max_episodes]

    episodes: list[Episode] = []
    for ep_idx in ep_indices:
        rows = rows_by_ep[ep_idx]
        rows.sort(key=lambda r: r.get(timestamp_col, 0) if timestamp_col else 0)

        if timestamp_col:
            timestamps = np.array([float(r[timestamp_col]) for r in rows])
        else:
            timestamps = np.arange(len(rows), dtype=np.float64) * 0.05  # 20 Hz default

        actions_raw = [r[action_col] for r in rows]
        if isinstance(actions_raw[0], list):
            actions = np.array(actions_raw, dtype=np.float64)
        else:
            actions = np.array([[float(a)] for a in actions_raw], dtype=np.float64)

        # Build observations dict — use state/proprio if available, else empty
        observations: dict[str, np.ndarray] = {}
        if obs_col:
            obs_raw = [r[obs_col] for r in rows]
            if isinstance(obs_raw[0], list):
                observations["proprio"] = np.array(obs_raw, dtype=np.float64)
            elif isinstance(obs_raw[0], (int, float)):
                observations["proprio"] = np.array([[float(v)] for v in obs_raw], dtype=np.float64)

        meta = EpisodeMetadata(episode_id=str(ep_idx))
        episodes.append(
            Episode(
                metadata=meta,
                timestamps=timestamps,
                observations=observations,
                actions=actions,
            )
        )

    return EpisodeBatch(
        episodes=episodes, dataset_name=dataset_id, format="lerobot-v2", source_path=dataset_id
    )


# ── corruption ────────────────────────────────────────────────────────────────


def _apply_corruption(batch, corrupt_fraction: float, corrupt_type: str = "mixed", seed: int = 42):
    """
    Apply synthetic corruption to a random subset of episodes.

    Returns (corrupted_batch, corrupted_indices) where corrupted_indices is
    the set of episode indices (within the batch) that were corrupted.
    This gives us exact ground truth for detection rate measurement.
    """
    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    rng = random.Random(seed)
    n = len(batch.episodes)
    n_corrupt = max(1, int(n * corrupt_fraction))
    corrupt_indices = set(rng.sample(range(n), n_corrupt))

    new_episodes = []
    for i, ep in enumerate(batch.episodes):
        if i not in corrupt_indices:
            new_episodes.append(ep)
            continue

        ts = ep.timestamps.copy()
        actions = ep.actions.copy()
        observations = dict(ep.observations)  # default: keep original observations

        # Rotate through corruption types for variety
        ctype = corrupt_type
        if ctype == "mixed":
            ctype = ["jitter", "spikes", "dropout"][i % 3]

        if ctype == "jitter":
            # Add large timestamp jitter (simulates clock drift / USB jitter)
            noise_ms = 50.0
            ts = ts + np.random.default_rng(seed + i).normal(0, noise_ms / 1000, size=ts.shape)
            ts = np.sort(ts)  # keep monotonic but jittered

        elif ctype == "spikes":
            # Inject velocity discontinuities by swapping random action pairs
            spike_rate = 0.10
            n_spikes = max(1, int(len(actions) * spike_rate))
            idxs = np.random.default_rng(seed + i).choice(len(actions) - 1, n_spikes, replace=False)
            for j in idxs:
                actions[j], actions[j + 1] = actions[j + 1].copy(), actions[j].copy()

        elif ctype == "dropout":
            # Drop a fraction of frames (simulate camera dropout).
            # Apply the same mask to observations so lengths stay consistent.
            drop_rate = 0.15
            n_drop = max(1, int(len(ts) * drop_rate))
            keep = np.ones(len(ts), dtype=bool)
            drop_idxs = np.random.default_rng(seed + i).choice(len(ts), n_drop, replace=False)
            keep[drop_idxs] = False
            ts = ts[keep]
            actions = actions[keep]
            observations = {k: v[keep] for k, v in ep.observations.items()}

        new_meta = EpisodeMetadata(
            episode_id=ep.metadata.episode_id if ep.metadata else str(i),
        )
        new_episodes.append(
            Episode(
                metadata=new_meta,
                timestamps=ts,
                observations=observations,
                actions=actions,
            )
        )

    corrupted_batch = EpisodeBatch(
        episodes=new_episodes,
        dataset_name=batch.dataset_name,
        format=batch.format,
        source_path=batch.source_path,
    )
    return corrupted_batch, corrupt_indices


# ── pipeline ──────────────────────────────────────────────────────────────────


def _run_pipeline(batch):
    """Run the Calibra diagnostic pipeline and return a DiagnosticReport."""
    from calibra.pipeline import Pipeline

    pipeline = Pipeline()
    return pipeline.run(batch)


def _get_anomalies(report, dataset_id: str, task_family: str):
    """Run episode anomaly detection with calibration context."""
    from calibra.anomalies import find_outliers

    return find_outliers(report, dataset=dataset_id, task_family=task_family)


# ── measurement ───────────────────────────────────────────────────────────────


def measure_clean_firing_rates(
    dataset_id: str,
    task_family: str,
    batch,
    report,
) -> dict:
    """
    Measure per-detector firing rates on a known-clean dataset.

    Returns a dict with:
        dataset, task_family, n_episodes, per_detector rates, concentration
    """
    anomalies = _get_anomalies(report, dataset_id, task_family)
    n_episodes = len(batch.episodes)
    n_flagged_total = len(anomalies)
    all_flags = [f for a in anomalies for f in a.flags]

    # Per-detector counts
    det_counts: dict[str, int] = Counter(f.metric for f in all_flags)

    # Concentration: top-5 episodes account for what fraction of flags?
    ep_counts = Counter(f.episode_id for a in anomalies for f in a.flags)
    top5 = sum(v for _, v in ep_counts.most_common(5))
    top5_frac = top5 / len(all_flags) if all_flags else 0.0

    # Position clustering
    sorted_ep_idxs = sorted(a.episode_idx for a in anomalies)
    start_cluster = sum(1 for i in sorted_ep_idxs if i <= int(n_episodes * 0.10))
    end_cluster = sum(1 for i in sorted_ep_idxs if i >= int(n_episodes * 0.90))
    position_note = ""
    if start_cluster >= 2 and n_flagged_total > 0 and start_cluster / n_flagged_total >= 0.30:
        position_note = "concentrated near dataset start"
    elif end_cluster >= 2 and n_flagged_total > 0 and end_cluster / n_flagged_total >= 0.30:
        position_note = "concentrated near dataset end"

    detectors = {}
    for det in _DETECTORS:
        n = det_counts.get(det, 0)
        detectors[det] = {
            "n_flagged": n,
            "firing_rate": n / n_episodes if n_episodes > 0 else 0.0,
        }

    return {
        "dataset": dataset_id,
        "task_family": task_family,
        "n_episodes": n_episodes,
        "n_flagged_episodes": n_flagged_total,
        "episode_flag_rate": n_flagged_total / n_episodes if n_episodes > 0 else 0.0,
        "top5_flag_fraction": round(top5_frac, 3),
        "position_note": position_note,
        "detectors": detectors,
    }


def measure_corruption_detection(
    dataset_id: str,
    task_family: str,
    batch,
    corrupt_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Corrupt a fraction of episodes and measure per-detector episode detection rates.

    Returns a dict with per-detector corrupted_episode_detection_rate.
    Ground truth is exact: we know which episodes were corrupted.

    Note: this is episode-level detection — an episode is counted as detected
    if the detector fired anywhere in it, regardless of whether it fired at the
    corrupted region specifically. The metric name reflects this:
    `corrupted_episode_detection_rate`, not `corrupted_region_detection_rate`.
    """
    corrupted_batch, corrupt_indices = _apply_corruption(
        batch, corrupt_fraction=corrupt_fraction, seed=seed
    )
    corrupt_report = _run_pipeline(corrupted_batch)
    anomalies = _get_anomalies(corrupt_report, dataset_id, task_family)

    # Per-detector: which corrupted episodes were caught (episode-level)?
    detected_by_det: dict[str, set] = defaultdict(set)
    for anomaly in anomalies:
        ep_orig_idx = anomaly.episode_idx
        if ep_orig_idx in corrupt_indices:
            for flag in anomaly.flags:
                detected_by_det[flag.metric].add(ep_orig_idx)

    n_corrupt = len(corrupt_indices)
    detectors = {}
    for det in _DETECTORS:
        n_detected = len(detected_by_det.get(det, set()))
        detectors[det] = {
            "n_corrupted_episodes": n_corrupt,
            "n_detected": n_detected,
            "corrupted_episode_detection_rate": n_detected / n_corrupt if n_corrupt > 0 else 0.0,
        }

    return {
        "dataset": dataset_id,
        "corrupt_fraction": corrupt_fraction,
        "n_corrupted": n_corrupt,
        "detectors": detectors,
    }


# ── Wilson score CI ───────────────────────────────────────────────────────────


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 1.0
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * ((p * (1 - p) / n_total + z**2 / (4 * n_total**2)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ── output ────────────────────────────────────────────────────────────────────


def _build_profiles(clean_results: list[dict], corrupt_results: dict[str, dict]) -> list[dict]:
    """Build CalibrationProfile-compatible dicts from measurement results."""
    import calibra

    profiles = []
    for clean in clean_results:
        dataset = clean["dataset"]
        task_family = clean["task_family"]
        n_episodes = clean["n_episodes"]
        corrupt = corrupt_results.get(dataset, {})

        for det in _DETECTORS:
            det_clean = clean["detectors"].get(det, {})
            det_corrupt = corrupt.get("detectors", {}).get(det, {})

            n_flagged = det_clean.get("n_flagged", 0)
            firing_rate = det_clean.get("firing_rate", 0.0)
            ci_lo, ci_hi = _wilson_ci(n_flagged, n_episodes)

            profiles.append(
                {
                    "dataset": dataset,
                    "task_family": task_family,
                    "detector": det,
                    "detector_version": calibra.__version__,
                    "config_hash": None,
                    "n_episodes": n_episodes,
                    "n_flagged": n_flagged,
                    "firing_rate": round(firing_rate, 4),
                    "ci_lower": round(ci_lo, 4),
                    "ci_upper": round(ci_hi, 4),
                    "dataset_version": None,
                    "measured_at": _today(),
                    "provenance": "benchmark_run",
                    # Episode-level: detector fired anywhere in the corrupted episode.
                    # Does not guarantee the corrupted region was the trigger.
                    "corrupted_episode_detection_rate": round(
                        det_corrupt.get("corrupted_episode_detection_rate", 0.0), 4
                    )
                    if det_corrupt
                    else None,
                    "notes": "Measured by benign_firing_rate_benchmark.py",
                }
            )
    return profiles


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _write_json(profiles: list[dict], path: pathlib.Path) -> None:
    path.write_text(json.dumps({"profiles": profiles}, indent=2), encoding="utf-8")
    print(f"  -> {path}")


def _write_csv(profiles: list[dict], path: pathlib.Path) -> None:
    import csv

    fieldnames = [
        "dataset",
        "task_family",
        "detector",
        "detector_version",
        "n_episodes",
        "n_flagged",
        "firing_rate",
        "ci_lower",
        "ci_upper",
        "corrupted_episode_detection_rate",
        "measured_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(profiles)
    print(f"  -> {path}")


def _write_markdown(
    clean_results: list[dict],
    corrupt_results: dict[str, dict],
    path: pathlib.Path,
) -> None:
    lines = [
        "# Calibra Detector Calibration Benchmark",
        "",
        "Measured by `experiments/benign_firing_rate_benchmark.py`.",
        "",
        "## Two-Axis Calibration Table",
        "",
        "| Detector | Clean flag rate | Detection rate | Signal ratio | Concentration |",
        "|---|---|---|---|---|",
    ]

    # Aggregate across datasets
    det_clean: dict[str, list[float]] = defaultdict(list)
    det_corrupt: dict[str, list[float]] = defaultdict(list)
    det_conc: dict[str, list[float]] = defaultdict(list)

    for clean in clean_results:
        for det in _DETECTORS:
            r = clean["detectors"].get(det, {}).get("firing_rate", 0.0)
            det_clean[det].append(r)
        # concentration per dataset: top5_flag_fraction proxy
        for det in _DETECTORS:
            det_conc[det].append(clean.get("top5_flag_fraction", 0.0))

    for cr in corrupt_results.values():
        for det in _DETECTORS:
            r = cr.get("detectors", {}).get(det, {}).get("corrupted_episode_detection_rate", None)
            if r is not None:
                det_corrupt[det].append(r)

    for det in _DETECTORS:
        clean_rate = np.mean(det_clean[det]) if det_clean[det] else 0.0
        corrupt_rate = np.mean(det_corrupt[det]) if det_corrupt[det] else None
        conc = np.mean(det_conc[det]) if det_conc[det] else None

        clean_str = f"{clean_rate:.1%}"
        corrupt_str = f"{corrupt_rate:.1%}" if corrupt_rate is not None else "—"
        ratio = f"{corrupt_rate / clean_rate:.0f}×" if corrupt_rate and clean_rate > 0 else "—"
        conc_str = f"{conc:.0%} in top 5 eps" if conc is not None else "—"

        lines.append(f"| {det} | {clean_str} | {corrupt_str} | {ratio} | {conc_str} |")

    lines += [
        "",
        "## Per-Dataset Results",
        "",
    ]

    for clean in clean_results:
        dataset = clean["dataset"]
        task_family = clean["task_family"]
        n = clean["n_episodes"]
        n_flag = clean["n_flagged_episodes"]
        conc = clean.get("top5_flag_fraction", 0.0)
        pos = clean.get("position_note", "")

        lines += [
            f"### {dataset} (task: {task_family}, n={n})",
            "",
            f"Episode flag rate: {n_flag}/{n} = {n_flag / n:.1%}",
            f"Top-5 episode concentration: {conc:.0%} of flags",
        ]
        if pos:
            lines.append(f"Position note: {pos}")
        lines += [
            "",
            "| Detector | Flagged | Rate | 95% CI |",
            "|---|---|---|---|",
        ]
        for det in _DETECTORS:
            d = clean["detectors"].get(det, {})
            nf = d.get("n_flagged", 0)
            rate = d.get("firing_rate", 0.0)
            ci_lo, ci_hi = _wilson_ci(nf, n)
            lines.append(f"| {det} | {nf}/{n} | {rate:.1%} | ({ci_lo:.1%}, {ci_hi:.1%}) |")

        corrupt = corrupt_results.get(dataset)
        if corrupt:
            lines += [
                "",
                f"**Corruption detection** (corrupt fraction: {corrupt['corrupt_fraction']:.0%})",
                "",
                "| Detector | Detected / Corrupted | Episode detection rate† |",
                "|---|---|---|",
            ]
            for det in _DETECTORS:
                d = corrupt.get("detectors", {}).get(det, {})
                n_det = d.get("n_detected", 0)
                n_corr = d.get("n_corrupted_episodes", 0)
                rate = d.get("corrupted_episode_detection_rate", 0.0)
                lines.append(f"| {det} | {n_det}/{n_corr} | {rate:.1%} |")
        lines.append("")

    lines += [
        "## Interpretation Note",
        "",
        "These rates use within-dataset MAD-based outlier detection.",
        "Even clean datasets have episodes at the tail of their own quality",
        "distribution — the benign firing rate captures how often those tails",
        "exceed the MAD threshold in practice.",
        "",
        "**A detected anomaly is not the same as confirmed corruption.**",
        "Review flagged episodes using `calibra review` before deciding to drop,",
        "downweight, or annotate them.",
        "",
        "† Episode detection rate is episode-level: an episode is counted as detected",
        "if the detector fired anywhere in it. It does not distinguish whether the",
        "detector fired at the corrupted region or elsewhere in the episode.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {path}")


def _print_summary_table(clean_results: list[dict], corrupt_results: dict[str, dict]) -> None:
    header = f"\n{'Detector':<18} {'Clean rate':>11} {'Detection rate':>14} {'Ratio':>7}"
    print("=" * 55)
    print("Calibra Detector Calibration - Two-Axis Summary")
    print("=" * 55)
    print(header)
    print("-" * 55)

    # Aggregate across datasets
    det_clean: dict[str, list[float]] = defaultdict(list)
    det_corrupt: dict[str, list[float]] = defaultdict(list)
    for clean in clean_results:
        for det in _DETECTORS:
            det_clean[det].append(clean["detectors"].get(det, {}).get("firing_rate", 0.0))
    for cr in corrupt_results.values():
        for det in _DETECTORS:
            r = cr.get("detectors", {}).get(det, {}).get("corrupted_episode_detection_rate")
            if r is not None:
                det_corrupt[det].append(r)

    for det in _DETECTORS:
        clean_rate = np.mean(det_clean[det]) if det_clean[det] else 0.0
        corrupt_rate = np.mean(det_corrupt[det]) if det_corrupt[det] else None
        clean_str = f"{clean_rate:.1%}"
        corrupt_str = f"{corrupt_rate:.1%}" if corrupt_rate is not None else "--"
        ratio = (
            f"{corrupt_rate / clean_rate:.0f}x"
            if corrupt_rate is not None and clean_rate > 1e-9
            else "--"
        )
        print(f"  {det:<16} {clean_str:>11} {corrupt_str:>14} {ratio:>7}")

    print("-" * 55)
    print(
        "\nNote: A detected anomaly != confirmed corruption."
        "\nReview flagged episodes before deciding to drop them."
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Calibra detector calibration on known-clean LeRobot datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset IDs to benchmark (default: pusht + aloha_sim_insertion_scripted).",
    )
    parser.add_argument(
        "--task-families",
        nargs="+",
        default=None,
        help="Task family for each dataset (e.g. pusht aloha). Must match --datasets length.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit episodes per dataset (for fast iteration).",
    )
    parser.add_argument(
        "--corrupt-rate",
        type=float,
        default=0.15,
        help="Fraction of episodes to corrupt for detection rate measurement (default: 0.15).",
    )
    parser.add_argument(
        "--no-corruption",
        action="store_true",
        help="Skip corruption detection measurement (only measure clean firing rates).",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR),
        help="Output directory for results files (default: experiments/results/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for corruption injection (default: 42).",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset list
    if args.datasets:
        families = args.task_families or (["unknown"] * len(args.datasets))
        if len(families) != len(args.datasets):
            print("Error: --task-families must have the same length as --datasets")
            return 1
        datasets = list(zip(args.datasets, families))
    else:
        datasets = _DEFAULT_DATASETS

    clean_results: list[dict] = []
    corrupt_results: dict[str, dict] = {}

    for dataset_id, task_family in datasets:
        print(f"\n-- {dataset_id} (task: {task_family}) --")

        try:
            batch = _load_lerobot_batch(dataset_id, max_episodes=args.max_episodes)
        except Exception as exc:
            print(f"  SKIP: could not load dataset — {exc}")
            continue

        n = len(batch.episodes)
        if n < 5:
            print(f"  SKIP: too few episodes ({n}) for reliable measurement")
            continue

        print(f"  Running pipeline on {n} episodes (clean)...")
        t0 = time.time()
        clean_report = _run_pipeline(batch)
        print(f"  Pipeline done in {time.time() - t0:.1f}s")

        cr = measure_clean_firing_rates(dataset_id, task_family, batch, clean_report)
        clean_results.append(cr)
        print(
            f"  Episode flag rate: {cr['n_flagged_episodes']}/{n} = {cr['episode_flag_rate']:.1%}"
        )
        print(f"  Top-5 episode concentration: {cr['top5_flag_fraction']:.0%} of flags")
        if cr.get("position_note"):
            print(f"  Position: {cr['position_note']}")

        if not args.no_corruption:
            print(f"\n  Applying {args.corrupt_rate:.0%} corruption...")
            corrupt_res = measure_corruption_detection(
                dataset_id,
                task_family,
                batch,
                corrupt_fraction=args.corrupt_rate,
                seed=args.seed,
            )
            corrupt_results[dataset_id] = corrupt_res
            print("  Detection rates:")
            for det in _DETECTORS:
                d = corrupt_res["detectors"].get(det, {})
                rate = d.get("corrupted_episode_detection_rate", 0.0)
                n_det = d.get("n_detected", 0)
                n_corr = d.get("n_corrupted_episodes", 0)
                print(f"    {det:<18} {n_det}/{n_corr} = {rate:.1%}")

    if not clean_results:
        print("\nNo datasets could be loaded. Exiting.")
        return 1

    # Build profiles and write outputs
    print("\nWriting results...")
    profiles = _build_profiles(clean_results, corrupt_results)
    _write_json(profiles, out_dir / "benign_firing_rates.json")
    _write_csv(profiles, out_dir / "benign_firing_rates.csv")
    _write_markdown(clean_results, corrupt_results, out_dir / "benign_firing_rates.md")

    # Print summary table
    _print_summary_table(clean_results, corrupt_results)

    print("\nTo update the built-in calibration table, copy the profiles from:")
    print(f"  {out_dir / 'benign_firing_rates.json'}")
    print("into calibra/calibration.py (_BUILTIN_PROFILES).")
    print("\nOr load them at runtime:")
    print("  from calibra.calibration import CalibrationRegistry")
    print(f"  registry = CalibrationRegistry.load_json('{out_dir / 'benign_firing_rates.json'}')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
