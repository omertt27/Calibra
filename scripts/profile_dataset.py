#!/usr/bin/env python3
"""
Profile any dataset through Calibra and emit a reference JSON.

This is the general-purpose replacement for format-specific scripts like
profile_pusht.py. Use this to add datasets to calibra/references/ and
grow the evidence base for the claim registry.

Usage
-----
    # LeRobot Hub dataset
    python scripts/profile_dataset.py lerobot/pusht
    python scripts/profile_dataset.py lerobot/droid_100 --control-mode position

    # HF URI
    python scripts/profile_dataset.py hf://lerobot/aloha_mobile_cabinet \\
        --control-mode position --gripper-dims 6,13 \\
        --out calibra/references/aloha_mobile_cabinet.json

    # Local HDF5
    python scripts/profile_dataset.py /data/robot_demos.h5 \\
        --format hdf5 --control-mode position \\
        --out calibra/references/my_robot.json

    # After profiling, run the ratio check:
    python scripts/generate_claims_doc.py --check

Output format
-------------
The JSON file emitted by this script is the canonical reference format used by
`calibra compare`. It records:

  meta/              dataset ID, episode count, action_dim, control_mode
  episode_structure/ length and duration distributions
  per_episode_distributions/   percentile profiles of per-episode metrics
  aggregate_metrics/           flat dict of mean metrics and CIs
  flags/             raw flag output from the pipeline

Once the file is placed in calibra/references/ it is immediately available as a
comparison target:

    calibra compare my_dataset <name>

After profiling, open calibra/claims/*.json and check whether the observed
values support or falsify any claim whose pending_tests list includes this
dataset or a dataset of this class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from calibra.analyzers.coverage import CoverageEntropyAnalyzer  # noqa: E402
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer  # noqa: E402
from calibra.analyzers.task_structure import TaskStructureAnalyzer  # noqa: E402
from calibra.analyzers.temporal import TemporalAnalyzer  # noqa: E402
from calibra.ingestion.registry import load  # noqa: E402
from calibra.pipeline import Pipeline  # noqa: E402
from calibra.schema.report import DiagnosticReport  # noqa: E402

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


# ── helpers ───────────────────────────────────────────────────────────────────


def distribution(values: list) -> dict:
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
            rows.append(
                {
                    "analyzer": result.analyzer_name,
                    "metric": flag.metric,
                    "level": flag.level.value,
                    "observed": flag.observed.value,
                    "ci_lower": flag.observed.ci_lower,
                    "ci_upper": flag.observed.ci_upper,
                    "interpretation": flag.interpretation,
                }
            )
    return rows


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Hub ID, hf:// URI, or local path to profile",
    )
    parser.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    parser.add_argument(
        "--control-mode",
        metavar="MODE",
        choices=["position", "velocity", "torque", "unknown"],
        default="unknown",
        help="Action space control mode — recorded in meta and used by compare. "
        "Required for accurate cross-dataset interpretation. (default: unknown)",
    )
    parser.add_argument(
        "--gripper-dims",
        metavar="DIMS",
        default=None,
        help="Comma-separated indices of gripper dimensions to exclude from "
        "smoothness metrics. Use '' to include all. Default: last dim (-1).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Write JSON to this path. Default: calibra/references/<dataset_name>.json",
    )
    parser.add_argument(
        "--note",
        metavar="TEXT",
        default="",
        help="Free-text note recorded in meta.note (e.g. hardware type, task, Hz).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but print JSON to stdout instead of writing to disk.",
    )
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # Strip hf:// prefix
    dataset = args.dataset
    if dataset.startswith("hf://"):
        dataset = dataset[len("hf://") :]

    dataset_name = dataset.split("/")[-1]

    # Resolve output path
    if args.dry_run:
        out_path = None
    elif args.out:
        out_path = Path(args.out)
    else:
        out_path = _REPO / "calibra" / "references" / f"{dataset_name}.json"
        log(f"Output path: {out_path}")

    # Resolve reader
    reader = None
    if args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    # Resolve gripper dims
    gripper_dims: list[int] = [-1]
    if args.gripper_dims is not None:
        raw = args.gripper_dims.strip()
        gripper_dims = [int(x) for x in raw.split(",") if x.strip()] if raw else []

    log(f"Loading {dataset!r} ...")
    try:
        batch = load(dataset, reader=reader)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    log(f"  {batch.n_episodes} episodes  ·  {batch.n_samples} steps total")
    if batch.episodes:
        ep0 = batch.episodes[0]
        log(f"  action_dim={ep0.action_dim}  modalities={sorted(batch.modalities)}")

    log("Running Calibra pipeline ...")
    pipeline = Pipeline(
        analyzers=[
            TemporalAnalyzer(),
            ControlSmoothnessAnalyzer(gripper_dims=gripper_dims),
            CoverageEntropyAnalyzer(),
            TaskStructureAnalyzer(),
        ]
    )
    report = pipeline.run(batch)
    log("Pipeline complete.")

    ep_lengths = [ep.n_steps for ep in batch.episodes]
    ep_durations = [ep.duration_s for ep in batch.episodes]
    action_dim = batch.episodes[0].action_dim if batch.episodes else None

    note = args.note or (
        f"Profiled by scripts/profile_dataset.py. control_mode={args.control_mode}."
    )

    output = {
        "meta": {
            "dataset": args.dataset,  # preserve original (including hf://)
            "n_episodes": batch.n_episodes,
            "n_steps_total": batch.n_samples,
            "action_dim": action_dim,
            "modalities": sorted(batch.modalities),
            "control_mode": args.control_mode,
            "gripper_dims": gripper_dims,
            "note": note,
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

    if args.dry_run or out_path is None:
        print(json_str)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_str)
    log(f"\n✅ Reference profile written to {out_path}")
    log("")
    log("Next steps:")
    log("  1. Open calibra/claims/*.json and check whether observed values")
    log("     support or falsify claims with this dataset in pending_tests.")
    log("  2. Run: python scripts/generate_claims_doc.py")
    log(f"  3. Run: calibra compare <your_dataset> {dataset_name}")


if __name__ == "__main__":
    main()
