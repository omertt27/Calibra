"""
calibra integrity — "Can I trust this dataset?"

Runs a fixed, cheap set of diagnostics that answer the first question
practitioners ask about a new dataset before anything about quality,
diversity, or optimization: are timestamps consistent, are episodes
complete, is the camera stream actually updating, are the recorded
motions themselves jittery/jerky. Groups findings into Critical /
Warnings / Passed rather than leading with a single score — the score
is still computed, but demoted to a summary line, not the headline
result.

Usage:
    calibra integrity /data/my_demos.h5
    calibra integrity /data/my_demos.h5 --format hdf5
    calibra integrity /data/my_demos.h5 --json
    calibra integrity lerobot/pusht --decode-images   # v1 LeRobot only, see below

Exit codes:
    0  No CRITICAL findings.
    1  One or more CRITICAL findings.
"""

from __future__ import annotations

import argparse
import json
import sys

from calibra.analyzers.blur import BlurAnalyzer
from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport, RiskFlag, RiskLevel

# Metric-name whitelist, not analyzer selection — TaskStructureAnalyzer also
# emits trajectory_diversity/contact_density/grasp_events_per_episode, and
# ControlSmoothnessAnalyzer also emits action_state_divergence/
# motion_collection_signature, which belong to the Quality layer, not
# Integrity. Only the three raw jerkiness metrics below ("are the recorded
# motions physically jittery") count as an Integrity-layer trust question —
# tracking error and collection-method signature are Quality-layer questions.
_INTEGRITY_METRICS = frozenset(
    {
        "timestamp_jitter_cv",
        "timestamp_dropout_rate",
        "action_obs_misalignment",
        "camera_physics_drift",
        "action_dropout_rate",
        "short_episode_fraction",
        "duplicate_frame_rate",
        "camera_freeze_events",
        "blurry_episode_fraction",
        "ldlj",
        "jerk_spike_rate",
        "velocity_discontinuity_rate",
    }
)

_ICONS = {
    RiskLevel.CRITICAL: "❌",
    RiskLevel.WARNING: "⚠️ ",
    RiskLevel.OK: "✅",
}

# Credit given per level toward the demoted Integrity Score.
_LEVEL_CREDIT = {RiskLevel.OK: 1.0, RiskLevel.WARNING: 0.5, RiskLevel.CRITICAL: 0.0}


def _is_integrity_flag(f: RiskFlag) -> bool:
    return f.metric in _INTEGRITY_METRICS or f.metric.startswith("camera_lag_std[")


def _integrity_flags(report: DiagnosticReport) -> list[RiskFlag]:
    return [f for f in report.flags if _is_integrity_flag(f)]


def _integrity_score(flags: list[RiskFlag]) -> tuple[int, str]:
    scored = [f for f in flags if f.level in _LEVEL_CREDIT]
    if not scored:
        return 100, "Healthy"
    score = round(100 * sum(_LEVEL_CREDIT[f.level] for f in scored) / len(scored))
    status = "Healthy" if score >= 90 else "Warning" if score >= 70 else "Critical"
    return score, status


def run_integrity(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra integrity",
        description='Check "can I trust this dataset?" before anything else.',
    )
    p.add_argument("path", help="Path or Hub ID of the dataset to check")
    p.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Print the report as JSON instead of human-readable text",
    )
    p.add_argument(
        "--decode-images",
        action="store_true",
        help=(
            "Decode camera frames for LeRobot v1 (HuggingFace Image-feature) "
            "datasets so duplicate_frame_rate/camera_freeze_events/blurry_episode_fraction "
            "can run. LeRobot-specific; increases load time and memory use. "
            "Not yet supported for v2/v3 (video-encoded) LeRobot datasets."
        ),
    )
    args = p.parse_args(argv)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://") :]

    reader = None
    if args.decode_images:
        from calibra.ingestion.adapters.lerobot import LeRobotReader

        reader = LeRobotReader(decode_images=True)
    elif args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {dataset_path!r} ...")
    try:
        from calibra.ingestion.registry import load

        batch = load(dataset_path, reader=reader)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {batch.n_episodes} episodes  ·  {batch.n_samples} steps")
    log("Running integrity checks ...")
    try:
        analyzers = [
            TemporalAnalyzer(),
            TaskStructureAnalyzer(),
            DuplicateFrameAnalyzer(),
            CameraFreezeAnalyzer(),
            BlurAnalyzer(),
            ControlSmoothnessAnalyzer(),
        ]
        report = Pipeline(analyzers=analyzers).run(batch)
    except Exception as exc:
        print(f"error running pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    flags = _integrity_flags(report)
    critical = [f for f in flags if f.level == RiskLevel.CRITICAL]
    warnings = [f for f in flags if f.level == RiskLevel.WARNING]
    passed = [f for f in flags if f.level == RiskLevel.OK]
    score, status = _integrity_score(flags)

    if args.json:
        result = {
            "dataset_name": report.dataset_name,
            "source_path": report.source_path,
            "format": report.format,
            "n_episodes": report.n_episodes,
            "critical": [_flag_to_dict(f) for f in critical],
            "warnings": [_flag_to_dict(f) for f in warnings],
            "passed": [_flag_to_dict(f) for f in passed],
            "integrity_score": score,
            "status": status,
        }
        print(json.dumps(result, indent=2))
    else:
        print(render(report, critical, warnings, passed, score, status))

    sys.exit(1 if critical else 0)


def _flag_to_dict(f: RiskFlag) -> dict:
    return {
        "metric": f.metric,
        "interpretation": f.interpretation,
        "implication": f.implication,
        "observed_value": f.observed.value,
    }


def render(
    report: DiagnosticReport,
    critical: list[RiskFlag],
    warnings: list[RiskFlag],
    passed: list[RiskFlag],
    score: int,
    status: str,
) -> str:
    lines = [
        "─── Dataset Integrity " + "─" * 36,
        f"{report.dataset_name} · {report.n_episodes} episodes",
        "",
    ]

    for label, group in (("Critical", critical), ("Warnings", warnings), ("Passed", passed)):
        lines.append(f"{label} ({len(group)})")
        for f in group:
            icon = _ICONS[f.level]
            lines.append(f"  {icon} {f.metric}: {f.interpretation}")
            if f.level != RiskLevel.OK:
                lines.append(f"      {f.implication}")
        lines.append("")

    lines.append(f"Integrity Score: {score}/100  ·  Status: {status}")
    lines.append("─" * 58)
    return "\n".join(lines)
