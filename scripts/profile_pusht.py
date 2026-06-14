#!/usr/bin/env python3
"""
Profile lerobot/pusht through Calibra and emit raw metric distributions.

This is Calibra's first run against real robotics data. Output is
observation-only: no thresholds applied, no pass/fail. The distributions here
become the first real-data reference point for baseline calibration — future
datasets are compared against these numbers, not against values engineered into
synthetic fixtures.

Usage:
    pip install 'calibra[lerobot]'
    python scripts/profile_pusht.py
    python scripts/profile_pusht.py --dataset lerobot/pusht --out profile_pusht.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibra.analyzers.coverage import CoverageEntropyAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.ingestion.adapters.lerobot import LeRobotReader
from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport

PUSHT_HUB_ID = "lerobot/pusht"
PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


# ── distribution helpers ──────────────────────────────────────────────────────

def distribution(values: list) -> dict:
    """Percentile profile of a per-episode float list. None values are dropped."""
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "note": "no valid values"}
    return {
        "n": int(len(arr)),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        **{f"p{p}": round(float(np.percentile(arr, p)), 6) for p in PERCENTILES},
    }


# ── extraction ────────────────────────────────────────────────────────────────

def per_episode_distributions(report: DiagnosticReport) -> dict:
    out = {}
    for result in report.analyzer_results:
        for key, val in result.raw_metrics.items():
            if not key.startswith("per_episode_"):
                continue
            if not isinstance(val, (list, np.ndarray)):
                continue
            label = f"{result.analyzer_name}/{key}"
            out[label] = distribution(val if isinstance(val, list) else val.tolist())
    return out


def aggregate_metrics(report: DiagnosticReport) -> dict:
    """Flatten non-per-episode scalar metrics from each analyzer's raw_metrics."""
    out: dict[str, dict] = {}
    for result in report.analyzer_results:
        flat: dict[str, float] = {}
        for key, val in result.raw_metrics.items():
            if key.startswith("per_episode_"):
                continue
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if isinstance(subval, (int, float)) and subkey != "episode_values":
                        flat[f"{key}.{subkey}"] = round(float(subval), 6)
            elif isinstance(val, (int, float)):
                flat[key] = round(float(val), 6)
        if flat:
            out[result.analyzer_name] = flat
    return out


def flag_summary(report: DiagnosticReport) -> list[dict]:
    rows = []
    for result in report.analyzer_results:
        for flag in result.flags:
            rows.append({
                "analyzer": result.analyzer_name,
                "metric": flag.metric,
                "level": flag.level.value,
                "observed": flag.observed.value,
                "ci_lower": flag.observed.ci_lower,
                "ci_upper": flag.observed.ci_upper,
                "interpretation": flag.interpretation,
            })
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", default=PUSHT_HUB_ID,
        help=f"HuggingFace Hub ID or local path (default: {PUSHT_HUB_ID})",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write JSON output to this file (default: stdout)",
    )
    args = parser.parse_args()

    log = lambda msg: print(msg, file=sys.stderr, flush=True)

    log(f"Loading {args.dataset!r} ...")
    batch = LeRobotReader().read(args.dataset)
    log(f"  {batch.n_episodes} episodes, {batch.n_samples} steps total")
    if batch.episodes:
        ep0 = batch.episodes[0]
        log(f"  action_dim={ep0.action_dim}  modalities={sorted(batch.modalities)}")

    log("Running Calibra pipeline ...")
    pipeline = Pipeline(analyzers=[
        TemporalAnalyzer(),
        # pusht actions are (dx, dy) velocity — no discrete gripper dimension.
        # Pass gripper_dims=[] so both action dims are included in smoothness metrics.
        ControlSmoothnessAnalyzer(gripper_dims=[]),
        CoverageEntropyAnalyzer(),
        TaskStructureAnalyzer(),
    ])
    report = pipeline.run(batch)
    log("Pipeline complete.")

    ep_lengths = [ep.n_steps for ep in batch.episodes]
    ep_durations = [ep.duration_s for ep in batch.episodes]

    output = {
        "meta": {
            "dataset": args.dataset,
            "n_episodes": batch.n_episodes,
            "n_steps_total": batch.n_samples,
            "action_dim": batch.episodes[0].action_dim if batch.episodes else None,
            "modalities": sorted(batch.modalities),
            "note": (
                "Observation-only profile. No threshold calibration. "
                "These distributions are Calibra's first real-data reference point."
            ),
        },
        "episode_structure": {
            "length_steps": distribution(ep_lengths),
            "duration_seconds": distribution(ep_durations),
        },
        "per_episode_distributions": per_episode_distributions(report),
        "aggregate_metrics": aggregate_metrics(report),
        "flags": flag_summary(report),
    }

    json_str = json.dumps(output, indent=2)
    if args.out:
        Path(args.out).write_text(json_str)
        log(f"Profile written to {args.out}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
