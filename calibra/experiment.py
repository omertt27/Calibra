"""
calibra.experiment — CLI handler for `calibra experiment`.

Records and reports on design-partner training experiments (see
calibra.experiment_log for the storage schema). This command does not run
training itself — training happens in the partner's own pipeline
(lerobot-train, a custom loop, whatever they already use). This command logs
the *results* of that training so the full/random/Calibra comparison the
design-partner protocol requires can be tracked and reported consistently.

    calibra experiment record --experiment-id partner-a-pusht \\
        --dataset partner-a/pusht_v3 --condition calibra --retention 25 \\
        --n-episodes 300 --policy act --eval-success-rate 0.84 \\
        --gpu-hours 19.8 --seed 0

    # or read the measured numbers straight from the training run's output
    calibra experiment record --experiment-id partner-a-pusht \\
        --condition calibra --retention 25 --policy act \\
        --from-metrics wandb/latest-run/files/wandb-summary.json \\
        --from-review review.json

    calibra experiment list --experiment-id partner-a-pusht
    calibra experiment report --experiment-id partner-a-pusht
    calibra experiment coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from calibra.experiment_log import CONDITIONS, ExperimentLog


def _load_metrics_bundle(args):
    """Return a MetricsBundle from --from-metrics, or None if not requested.

    Exits(1) on a bad path / unparseable file / bad --map, so the caller can
    assume a usable bundle when this returns non-None.
    """
    if not args.from_metrics:
        return None
    from calibra.metrics_ingest import load_metrics, parse_field_map

    try:
        field_map = parse_field_map(args.metrics_map)
        return load_metrics(args.from_metrics, fmt=args.metrics_format, extra_map=field_map)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: --from-metrics: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_review_rollup(args):
    """Return a list[EpisodeAssessment] from --from-review, or None if not requested.

    Rejects a partial-coverage review file: the mean_* fields are meant to be a
    dataset-level mean, not a mean over just the review queue's top-N.
    """
    if not args.from_review:
        return None
    from calibra.assessment import EpisodeAssessment

    try:
        rev = json.loads(Path(args.from_review).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: --from-review: {exc}", file=sys.stderr)
        sys.exit(1)

    rows = rev.get("assessments") if isinstance(rev, dict) else None
    if not rows:
        print("error: --from-review: no 'assessments' array in file", file=sys.stderr)
        sys.exit(1)

    n_eps = rev.get("n_episodes")
    if isinstance(n_eps, int) and len(rows) < n_eps:
        print(
            f"error: --from-review: file covers {len(rows)} of {n_eps} episodes; "
            f"re-run `calibra review <dataset> --top {n_eps} -o {args.from_review}` "
            f"so the rollup is a true dataset-level mean",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return [
            EpisodeAssessment(
                episode_id=str(r.get("episode_id", i)),
                anomaly_score=r["anomaly_score"],
                quality_risk=r["quality_risk"],
                coverage_value=r.get("coverage_value"),
            )
            for i, r in enumerate(rows)
        ]
    except (KeyError, TypeError) as exc:
        print(f"error: --from-review: malformed assessment row ({exc})", file=sys.stderr)
        sys.exit(1)


def _print_dry_run(args, resolved, sources, bundle, assessments):
    print("[dry-run] would record:")
    print(f"  experiment_id      = {args.experiment_id}")
    print(f"  condition          = {args.condition}")
    print(f"  retention_pct      = {args.retention:g}")
    for field in (
        "gpu_hours",
        "wall_clock_seconds",
        "energy_kwh",
        "training_loss",
        "eval_success_rate",
    ):
        val = resolved[field]
        shown = "unset" if val is None else f"{val:g}"
        src = f"  ({sources[field]})" if sources[field] else ""
        print(f"  {field:<18} = {shown}{src}")
    if assessments is not None:
        from calibra.assessment import summarize_assessments

        s = summarize_assessments(assessments)
        print(
            f"  mean_anomaly_score  = {s['mean_anomaly_score']}  (rollup of {len(assessments)} episodes)"
        )
        print(f"  mean_quality_risk   = {s['mean_quality_risk']}")
        print(f"  mean_coverage_value = {s['mean_coverage_value']}")
    if bundle is not None:
        print(f"  metrics_source      = {bundle.source}")
    print("  (nothing written)")


def run_experiment(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra experiment",
        description=(
            "Record and report design-partner training experiment results "
            "(full dataset vs. random subset vs. Calibra coreset, at multiple "
            "retention levels). Stored locally only — never synced to any "
            "network endpoint."
        ),
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    record_p = sub.add_parser("record", help="Log the result of one training run")
    record_p.add_argument("--experiment-id", required=True, help="Groups related runs together")
    record_p.add_argument(
        "--condition",
        required=True,
        choices=CONDITIONS,
        help="Which arm of the comparison this run is",
    )
    record_p.add_argument(
        "--retention",
        type=float,
        required=True,
        metavar="PCT",
        help="Retention percentage, e.g. 25",
    )
    record_p.add_argument("--dataset", default="unknown", help="Dataset name/repo id")
    record_p.add_argument(
        "--partner", default="", help="Design-partner label (free text, local only)"
    )
    record_p.add_argument("--embodiment", default="", help="e.g. 'so-100', 'aloha', 'humanoid'")
    record_p.add_argument("--task", default="")
    record_p.add_argument("--policy", dest="policy_family", default="generic")
    record_p.add_argument("--model-size", default="", help="e.g. '80M params'")
    record_p.add_argument("--n-episodes", type=int, default=0)
    record_p.add_argument("--gpu-hours", type=float, default=None)
    record_p.add_argument("--wall-clock-seconds", type=float, default=None)
    record_p.add_argument("--energy-kwh", type=float, default=None)
    record_p.add_argument("--training-loss", type=float, default=None)
    record_p.add_argument(
        "--eval-success-rate", type=float, default=None, help="0.0-1.0 measured policy success rate"
    )
    record_p.add_argument("--seed", type=int, default=None)
    record_p.add_argument("--notes", default="")
    record_p.add_argument(
        "--mean-anomaly-score",
        type=float,
        default=None,
        help="0.0-1.0 dataset-level mean anomaly score (see `calibra review`)",
    )
    record_p.add_argument(
        "--mean-quality-risk",
        type=float,
        default=None,
        help="0.0-1.0 dataset-level mean quality risk (see `calibra review`)",
    )
    record_p.add_argument(
        "--mean-coverage-value",
        type=float,
        default=None,
        help="0.0-1.0 dataset-level mean coverage value (see `calibra review`)",
    )
    record_p.add_argument(
        "--path", default=None, help="Override the default ~/.calibra/experiments.jsonl"
    )
    record_p.add_argument(
        "--from-metrics",
        metavar="PATH",
        default=None,
        help=(
            "Read gpu-hours / wall-clock / eval success / loss / energy from a "
            "finished run's metrics file or directory (flat JSON, or a "
            "wandb-summary.json from an offline run). Explicit flags above "
            "override anything found here. No network access."
        ),
    )
    record_p.add_argument(
        "--metrics-format",
        choices=["auto", "json", "wandb"],
        default="auto",
        help="Format of --from-metrics (default: auto-detect).",
    )
    record_p.add_argument(
        "--map",
        action="append",
        default=[],
        dest="metrics_map",
        metavar="FIELD=PATH",
        help=(
            "Map a record field to a key path in the metrics file when the "
            "built-in aliases miss it, e.g. "
            "--map eval_success_rate=results.eval.success . Repeatable."
        ),
    )
    record_p.add_argument(
        "--from-review",
        metavar="PATH",
        default=None,
        help=(
            "Roll a `calibra review --json` file's per-episode assessments up "
            "into mean anomaly / quality-risk / coverage-value. The file must "
            "cover every episode — run "
            "`calibra review <dataset> --top <n_episodes> -o review.json`."
        ),
    )
    record_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse --from-metrics / --from-review and print what would be recorded, writing nothing.",
    )

    list_p = sub.add_parser("list", help="List recorded runs")
    list_p.add_argument("--experiment-id", default=None)
    list_p.add_argument("--json", action="store_true")
    list_p.add_argument("--path", default=None)

    report_p = sub.add_parser(
        "report", help="Print the retention-curve comparison for one experiment"
    )
    report_p.add_argument("--experiment-id", required=True)
    report_p.add_argument("--json", action="store_true")
    report_p.add_argument("--path", default=None)

    coverage_p = sub.add_parser(
        "coverage",
        help="Print embodiment/task/policy matrix coverage across all recorded experiments",
    )
    coverage_p.add_argument("--json", action="store_true")
    coverage_p.add_argument("--path", default=None)

    args = p.parse_args(argv)
    path = Path(args.path) if getattr(args, "path", None) else None
    log = ExperimentLog(path=path)

    if args.subcommand == "record":
        bundle = _load_metrics_bundle(args)
        assessments = _load_review_rollup(args)

        # Resolve each measured field: an explicit flag wins; otherwise take
        # whatever --from-metrics found; otherwise leave it unset.
        resolved: dict = {}
        sources: dict = {}
        for field in (
            "gpu_hours",
            "wall_clock_seconds",
            "energy_kwh",
            "training_loss",
            "eval_success_rate",
        ):
            flag_val = getattr(args, field)
            mb_val = getattr(bundle, field) if bundle is not None else None
            if flag_val is not None:
                resolved[field] = flag_val
                sources[field] = (
                    f"--{field.replace('_', '-')} (overrides metrics {mb_val:g})"
                    if mb_val is not None
                    else f"--{field.replace('_', '-')}"
                )
            elif mb_val is not None:
                resolved[field] = mb_val
                sources[field] = f"metrics[{bundle.matched.get(field, '?')}]"
            else:
                resolved[field] = None
                sources[field] = None

        if args.dry_run:
            _print_dry_run(args, resolved, sources, bundle, assessments)
            return

        try:
            rec = log.record(
                experiment_id=args.experiment_id,
                condition=args.condition,
                retention_pct=args.retention,
                dataset_name=args.dataset,
                partner=args.partner,
                embodiment=args.embodiment,
                task=args.task,
                policy_family=args.policy_family,
                model_size=args.model_size,
                n_episodes=args.n_episodes,
                gpu_hours=resolved["gpu_hours"],
                wall_clock_seconds=resolved["wall_clock_seconds"],
                energy_kwh=resolved["energy_kwh"],
                training_loss=resolved["training_loss"],
                eval_success_rate=resolved["eval_success_rate"],
                seed=args.seed,
                notes=args.notes,
                metrics_source=bundle.source if bundle is not None else "",
                mean_anomaly_score=args.mean_anomaly_score,
                mean_quality_risk=args.mean_quality_risk,
                mean_coverage_value=args.mean_coverage_value,
                assessments=assessments,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Recorded {rec.record_id}: {rec.experiment_id} / {rec.condition} @ {rec.retention_pct:.0f}%"
        )
        if bundle is not None:
            print(f"  metrics from {bundle.source}:")
            for field in sorted(resolved):
                if sources[field] is not None:
                    print(f"    {field}: {resolved[field]:g}  ({sources[field]})")
        if assessments is not None:
            print(f"  dataset rollup from {args.from_review}: {len(assessments)} episodes")
        print(f"  -> {log.path}")
        return

    if args.subcommand == "list":
        records = log.list_records(args.experiment_id)
        if args.json:
            print(json.dumps([r.to_dict() for r in records], indent=2))
            return
        if not records:
            scope = f" for experiment_id={args.experiment_id!r}" if args.experiment_id else ""
            print(f"No records{scope}.")
            return
        for r in records:
            success = f"{r.eval_success_rate:.1%}" if r.eval_success_rate is not None else "n/a"
            print(
                f"{r.record_id}  {r.experiment_id:<24} {r.condition:<8} "
                f"{r.retention_pct:>5.0f}%  eps={r.n_episodes:<6} success={success}"
            )
        return

    if args.subcommand == "report":
        if args.json:
            table = log.retention_table(args.experiment_id)
            payload = {
                str(level): {cond: rec.to_dict() for cond, rec in conds.items()}
                for level, conds in table.items()
            }
            print(
                json.dumps(
                    {
                        "experiment_id": args.experiment_id,
                        "retention_table": payload,
                        "calibra_vs_random": log.calibra_vs_random(args.experiment_id),
                        "missing_conditions": log.missing_conditions(args.experiment_id),
                    },
                    indent=2,
                )
            )
            return
        print(log.report(args.experiment_id))
        return

    if args.subcommand == "coverage":
        if args.json:
            coverage = log.matrix_coverage()
            payload = {
                f"{embodiment}|{task}|{policy}": info
                for (embodiment, task, policy), info in coverage.items()
            }
            print(json.dumps(payload, indent=2))
            return
        print(log.coverage_report())
        return
