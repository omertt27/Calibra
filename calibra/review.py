"""
calibra review — ranked episode review queue.

Runs the diagnostic pipeline, computes a three-axis EpisodeAssessment per
episode (calibra.assessment), and surfaces the episodes most worth a human
look — never a delete recommendation. Suggested action is always "Inspect":
Calibra ranks and explains, a human decides what (if anything) to exclude.

Usage:
    calibra review /data/my_demos.h5
    calibra review lerobot/pusht --format lerobot --top 15
    calibra review /data/my_ds --mode fast
    calibra review /data/my_ds --group-by task
    calibra review /data/my_ds --output episode_ids.json
    calibra review /data/my_ds --json
"""

from __future__ import annotations

import argparse
import json
import sys

from calibra.assessment import EpisodeAssessment, compute_episode_assessments, rank_for_review
from calibra.pipeline import Pipeline


def run_review(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra review",
        description="Rank episodes for human review by anomaly, quality risk, and coverage value.",
    )
    p.add_argument("path", help="Path or Hub ID of the dataset to review")
    p.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="full",
        help=(
            "'full' (default) runs every analyzer, including coverage_value. "
            "'fast' restricts to cheap action/timestamp-only diagnostics — "
            "quicker on very large datasets, but coverage_value will be "
            "unavailable and fewer anomaly reasons will be found."
        ),
    )
    p.add_argument(
        "--group-by",
        metavar="FIELDS",
        default=None,
        help=(
            "Comma-separated metadata fields to rank within (e.g. 'task', or "
            "'task,robot' once populated). Prevents a harder task's episodes "
            "from all looking like outliers just for being compared against "
            "an easier one. Default: rank across the whole dataset."
        ),
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of episodes to show/export (default: 20)",
    )
    p.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        default=None,
        help="Write the top-N episode IDs (LeRobot-compatible) plus full assessments to a JSON file",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Print the ranked queue as JSON instead of human-readable text",
    )
    args = p.parse_args(argv)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://") :]

    reader = None
    if args.format:
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
    log(f"Running diagnostic pipeline (mode={args.mode}) ...")
    try:
        report = Pipeline(mode=args.mode).run(batch)
    except Exception as exc:
        print(f"error running pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    group_by = args.group_by.split(",") if args.group_by else None
    assessments = compute_episode_assessments(report, batch=batch, group_by=group_by)
    ranked = rank_for_review(assessments)
    top = ranked[: args.top]

    result = {
        "dataset_name": report.dataset_name,
        "source_path": report.source_path,
        "format": report.format,
        "n_episodes": report.n_episodes,
        "top_n": len(top),
        "group_by": group_by,
        "episode_ids": [a.episode_id for a in top],
        "assessments": [_assessment_to_dict(a) for a in top],
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        log(f"Wrote {len(top)} episode IDs to {args.output}")

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(render(top, report.n_episodes, group_by))


def _assessment_to_dict(a: EpisodeAssessment) -> dict:
    return {
        "episode_id": a.episode_id,
        "review_priority": a.review_priority,
        "anomaly_score": a.anomaly_score,
        "quality_risk": a.quality_risk,
        "coverage_value": a.coverage_value,
        "reasons": [{"metric": r.metric, "percentile": r.percentile} for r in a.reasons],
        "suggested_action": "Inspect",
    }


def render(top: list[EpisodeAssessment], n_episodes: int, group_by: list[str] | None) -> str:
    if not top:
        return "No episodes ranked — the pipeline produced no per-episode metrics to compare."

    lines = [
        "─── Episode Review Queue " + "─" * 32,
        f"Top {len(top)} of {n_episodes} episodes recommended for review"
        + (f", grouped by {'/'.join(group_by)}" if group_by else "")
        + ":",
        "",
    ]
    for i, a in enumerate(top, 1):
        header = (
            f"{i}. ep_{a.episode_id}   review_priority={a.review_priority:.2f}  "
            f"anomaly={a.anomaly_score:.2f}  quality_risk={a.quality_risk:.2f}"
        )
        if a.coverage_value is not None:
            header += f"  coverage_value={a.coverage_value:.2f}"
        lines.append(header)
        if a.reasons:
            lines.append("   Reasons:")
            for r in a.reasons:
                lines.append(f"     - {r.metric}: {r.percentile:.1f} percentile")
        lines.append("   Suggested action: Inspect")
        lines.append("")

    lines.append(
        "This is a ranked queue, not a verdict — an unusual episode may be a "
        "recovery behavior or a valuable rare state, not a defect."
    )
    lines.append("─" * 58)
    return "\n".join(lines)
