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

    calibra experiment list --experiment-id partner-a-pusht
    calibra experiment report --experiment-id partner-a-pusht
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from calibra.experiment_log import CONDITIONS, ExperimentLog


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
        "--condition", required=True, choices=CONDITIONS, help="Which arm of the comparison this run is"
    )
    record_p.add_argument(
        "--retention", type=float, required=True, metavar="PCT", help="Retention percentage, e.g. 25"
    )
    record_p.add_argument("--dataset", default="unknown", help="Dataset name/repo id")
    record_p.add_argument("--partner", default="", help="Design-partner label (free text, local only)")
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
    record_p.add_argument("--path", default=None, help="Override the default ~/.calibra/experiments.jsonl")

    list_p = sub.add_parser("list", help="List recorded runs")
    list_p.add_argument("--experiment-id", default=None)
    list_p.add_argument("--json", action="store_true")
    list_p.add_argument("--path", default=None)

    report_p = sub.add_parser("report", help="Print the retention-curve comparison for one experiment")
    report_p.add_argument("--experiment-id", required=True)
    report_p.add_argument("--json", action="store_true")
    report_p.add_argument("--path", default=None)

    args = p.parse_args(argv)
    path = Path(args.path) if getattr(args, "path", None) else None
    log = ExperimentLog(path=path)

    if args.subcommand == "record":
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
                gpu_hours=args.gpu_hours,
                wall_clock_seconds=args.wall_clock_seconds,
                energy_kwh=args.energy_kwh,
                training_loss=args.training_loss,
                eval_success_rate=args.eval_success_rate,
                seed=args.seed,
                notes=args.notes,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Recorded {rec.record_id}: {rec.experiment_id} / {rec.condition} @ {rec.retention_pct:.0f}%")
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
