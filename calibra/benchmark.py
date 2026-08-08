"""
calibra benchmark — Closed-loop policy training benchmark and simulation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import List, Optional

from calibra.pipeline import Pipeline
from calibra.predict import predict_outcome
from calibra.pruning import CoresetSelector
from calibra.schema.episode import EpisodeBatch

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN = "─" * _WIDTH


def _lookup_measured(table: dict, condition: str, target_pct: float, tol: float = 1.0):
    """Return the ExperimentRecord for `condition` at the retention level in
    `table` closest to `target_pct`, within `tol` percentage points, or None.
    """
    best = None
    best_dist = tol
    for level, conditions in table.items():
        rec = conditions.get(condition)
        if rec is None:
            continue
        dist = abs(level - target_pct)
        if dist <= best_dist:
            best, best_dist = rec, dist
    return best


def _diagnose_subset(pipeline: Pipeline, batch, episode_ids, suffix: str, policy_family: str):
    """Run diagnostics + heuristic prediction on the subset of `batch` whose
    episode ids are in `episode_ids`. Returns (n_episodes, simulated_score)."""
    episode_ids = set(episode_ids)
    episodes = [ep for ep in batch.episodes if ep.metadata.episode_id in episode_ids]
    subset = EpisodeBatch(
        episodes=episodes,
        dataset_name=f"{batch.dataset_name}_{suffix}",
        format=batch.format,
        source_path=batch.source_path,
    )
    report = pipeline.run(subset, policy_family=policy_family)
    pred = predict_outcome(report, policy_family=policy_family)
    score = max(0.0, pred.get("predicted_score", 100.0))
    return len(episodes), score


def _condition_result(
    n_episodes: int,
    n_total: int,
    base_gpu_hours: float,
    sim_score: float,
    measured_rec: Optional[object],
) -> dict:
    """Build a {n_episodes, gpu_hours, gpu_hours_source, predicted_success_rate,
    success_rate_source} dict, preferring a measured ExperimentRecord's values
    field-by-field over the simulated ones (linear GPU-hour scaling + heuristic
    predicted score)."""
    sim_gpu_hours = base_gpu_hours * (n_episodes / n_total) if n_total else 0.0
    gpu_hours, gpu_hours_source = sim_gpu_hours, "simulated"
    success, success_source = sim_score, "simulated"

    if measured_rec is not None:
        if measured_rec.gpu_hours is not None:
            gpu_hours, gpu_hours_source = measured_rec.gpu_hours, "measured"
        if measured_rec.eval_success_rate is not None:
            success, success_source = measured_rec.eval_success_rate * 100.0, "measured"

    return {
        "n_episodes": n_episodes,
        "gpu_hours": gpu_hours,
        "gpu_hours_source": gpu_hours_source,
        "predicted_success_rate": success,
        "success_rate_source": success_source,
    }


def _case_study_status(conditions: List[dict]) -> str:
    """
    Classify a set of condition dicts (full/random/calibra, one or more
    retention levels) by how much of it is real measured data vs. simulation.

    CASE STUDY / VALIDATED — every condition has both gpu_hours and
        predicted_success_rate sourced from real measured training.
    PARTIAL MEASUREMENT     — some measured values exist, but not all. Not
        safe to present as a validated result — a skeptical reviewer will
        find the simulated numbers immediately.
    SIMULATED               — nothing measured yet; this is a prediction.
    """

    def _fully_measured(c: dict) -> bool:
        return c["gpu_hours_source"] == "measured" and c["success_rate_source"] == "measured"

    if all(_fully_measured(c) for c in conditions):
        return "CASE STUDY / VALIDATED"
    if any(
        c["gpu_hours_source"] == "measured" or c["success_rate_source"] == "measured"
        for c in conditions
    ):
        return "PARTIAL MEASUREMENT"
    return "SIMULATED"


def run_benchmark(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra benchmark",
        description=(
            "Simulates and benchmarks training metrics for the raw dataset, "
            "a randomly-pruned baseline, and the Calibra coreset. "
            "Outputs expected compute savings and predicted success rates."
        ),
    )
    p.add_argument("path", help="Path of the source dataset to benchmark")
    p.add_argument(
        "--keep",
        "-k",
        type=float,
        default=0.3,
        help="Fraction of episodes to retain in the pruned coresets (default: 0.3). Ignored with --sweep.",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run the full design-partner retention curve (see --fractions) instead of a "
            "single --keep fraction: full baseline, then random vs. Calibra at every "
            "retention level."
        ),
    )
    p.add_argument(
        "--fractions",
        default="0.10,0.25,0.50,0.75,1.00",
        help=(
            "Comma-separated retention fractions for --sweep, matching the design-partner "
            "protocol (default: 0.10,0.25,0.50,0.75,1.00)."
        ),
    )
    p.add_argument(
        "--policy",
        metavar="FAMILY",
        default="diffusion",
        help="Target policy family for success prediction (default: diffusion)",
    )
    p.add_argument(
        "--format",
        "-f",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Print raw metrics in JSON format to stdout.",
    )
    p.add_argument(
        "--base-gpu-hours",
        type=float,
        default=24.0,
        help="GPU-hours required to train on the full (100%) dataset (default: 24.0)",
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "If set, substitute real measured GPU-hours / eval success rate from "
            "`calibra experiment record` (calibra.experiment_log) wherever a matching "
            "condition and retention level has been logged, in place of the simulation. "
            "Falls back to simulated values for anything not yet measured."
        ),
    )
    args = p.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading dataset from {args.path!r} ...")

    # 1. Load dataset
    try:
        from calibra.ingestion.registry import load

        batch = load(args.path, reader=args.format)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    n_total = batch.n_episodes
    if n_total < 5:
        print(
            f"error: Dataset has only {n_total} episodes. Need at least 5 to run benchmarks.",
            file=sys.stderr,
        )
        sys.exit(1)

    log(f"Dataset loaded: {n_total} episodes, {batch.n_samples} steps.")
    log("Running diagnostics on full (Raw) dataset ...")

    # 2. Raw dataset diagnostics and prediction (shared baseline for both modes)
    pipeline = Pipeline()
    raw_report = pipeline.run(batch, policy_family=args.policy)
    raw_pred = predict_outcome(raw_report, policy_family=args.policy)
    raw_score = max(0.0, raw_pred.get("predicted_score", 100.0))

    if args.sweep:
        _run_sweep(args, batch, pipeline, raw_report, raw_score, n_total, log)
        return

    k_size = max(1, round(n_total * args.keep))

    # 3. Calibra coreset curation
    log(f"Running Calibra coreset selection (keep fraction: {args.keep:.2f}) ...")
    selector = CoresetSelector(keep_fraction=args.keep)
    prune_res = selector.select(batch, raw_report)
    log("Running diagnostics on Calibra coreset ...")
    calibra_n, calibra_score = _diagnose_subset(
        pipeline, batch, prune_res.keep_episode_ids, "calibra_coreset", args.policy
    )

    # 4. Random pruned baseline
    log("Running diagnostics on Randomly pruned baseline ...")
    random.seed(42)
    random_ids = random.sample([ep.metadata.episode_id for ep in batch.episodes], k_size)
    random_n, random_score = _diagnose_subset(pipeline, batch, random_ids, "random_pruned", args.policy)

    # 5. Substitute real measured numbers wherever `calibra experiment record`
    # has logged them for a matching condition and retention level.
    elog_table: dict = {}
    if args.experiment_id:
        from calibra.experiment_log import ExperimentLog

        elog_table = ExperimentLog().retention_table(args.experiment_id)

    full_rec = _lookup_measured(elog_table, "full", 100.0)
    random_rec = _lookup_measured(elog_table, "random", args.keep * 100.0)
    calibra_rec = _lookup_measured(elog_table, "calibra", args.keep * 100.0)

    raw_cond = _condition_result(n_total, n_total, args.base_gpu_hours, raw_score, full_rec)
    random_cond = _condition_result(random_n, n_total, args.base_gpu_hours, random_score, random_rec)
    calibra_cond = _condition_result(calibra_n, n_total, args.base_gpu_hours, calibra_score, calibra_rec)

    # Compute savings from GPU-hours directly, not from episode-count reduction —
    # data reduction and compute reduction are only equal under the simulated
    # linear-scaling assumption; once real numbers are mixed in they can diverge
    # (dataloader/decode/I/O overhead doesn't shrink proportionally with data).
    compute_savings = (
        100.0 * (1.0 - (calibra_cond["gpu_hours"] / raw_cond["gpu_hours"]))
        if raw_cond["gpu_hours"] > 0
        else 0.0
    )

    status = _case_study_status([raw_cond, random_cond, calibra_cond])
    any_measured = status != "SIMULATED"

    # Compile result summary
    summary = {
        "dataset_name": batch.dataset_name,
        "policy_family": args.policy,
        "n_original": n_total,
        "keep_fraction": args.keep,
        "experiment_id": args.experiment_id,
        "any_measured": any_measured,
        "status": status,
        "results": {
            "raw": {**raw_cond, "gpu_hours": round(raw_cond["gpu_hours"], 1), "predicted_success_rate": round(raw_cond["predicted_success_rate"], 1)},
            "random": {**random_cond, "gpu_hours": round(random_cond["gpu_hours"], 1), "predicted_success_rate": round(random_cond["predicted_success_rate"], 1)},
            "calibra": {**calibra_cond, "gpu_hours": round(calibra_cond["gpu_hours"], 1), "predicted_success_rate": round(calibra_cond["predicted_success_rate"], 1)},
        },
        "compute_savings_pct": round(compute_savings, 1),
    }

    # Render training integration code block recommendations
    kept_indices_str = ",".join(
        str(i)
        for i, ep in enumerate(batch.episodes)
        if ep.metadata.episode_id in prune_res.keep_episode_ids
    )
    if batch.format == "lerobot":
        lerobot_cmd = (
            f"lerobot-train --dataset.repo_id {batch.dataset_name} "
            f'--dataset.episodes "[{kept_indices_str}]"'
        )
    else:
        lerobot_cmd = (
            f'python train.py --dataset {args.path} --coreset-indices "{kept_indices_str}"'
        )

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    def _tag(source: str) -> str:
        return "(measured)" if source == "measured" else "(simulated)"

    header = (
        "CALIBRA CLOSED-LOOP TRAINING BENCHMARK — MIXED MEASURED/SIMULATED"
        if any_measured
        else "CALIBRA CLOSED-LOOP TRAINING BENCHMARK SIMULATION"
    )

    if status == "CASE STUDY / VALIDATED":
        status_lines = (
            "  Full + random + calibra are all measured at this retention level.\n"
            "  Safe to report as a validated case study.\n"
        )
    elif status == "PARTIAL MEASUREMENT":
        status_lines = (
            "  Some figures above are measured, others are still simulated.\n"
            "  Do not report this as a validated case study yet — run "
            "`calibra experiment record`\n"
            "  for the remaining (simulated) conditions first.\n"
        )
    else:
        status_lines = (
            "  No measured results yet — these are predictions, not a case study.\n"
            "  Run `calibra experiment record` after real training to upgrade this.\n"
        )

    # Render comparison report
    print(
        f"\n{_THICK}\n"
        f"  {header}\n"
        f"{_THICK}\n"
        f"  Dataset: {batch.dataset_name}  ({n_total} episodes)\n"
        f"  Policy : {args.policy.upper()}\n"
        f"{_THIN}\n"
        f"  CURATION STRATEGY COMPARISON:\n"
        f"\n"
        f"  1. RAW DATASET (100%)\n"
        f"     - Size: {n_total} episodes\n"
        f"     - Training Time: {raw_cond['gpu_hours']:.1f} GPU-hours {_tag(raw_cond['gpu_hours_source'])}\n"
        f"     - Success Rate: {raw_cond['predicted_success_rate']:.1f}% {_tag(raw_cond['success_rate_source'])}\n"
        f"\n"
        f"  2. RANDOM PRUNED ({(random_cond['n_episodes'] / n_total):.0%})\n"
        f"     - Size: {random_cond['n_episodes']} episodes\n"
        f"     - Training Time: {random_cond['gpu_hours']:.1f} GPU-hours {_tag(random_cond['gpu_hours_source'])}\n"
        f"     - Success Rate: {random_cond['predicted_success_rate']:.1f}% {_tag(random_cond['success_rate_source'])}\n"
        f"\n"
        f"  3. CALIBRA CORESET ({(calibra_cond['n_episodes'] / n_total):.0%})\n"
        f"     - Size: {calibra_cond['n_episodes']} episodes\n"
        f"     - Training Time: {calibra_cond['gpu_hours']:.1f} GPU-hours {_tag(calibra_cond['gpu_hours_source'])}\n"
        f"     - Success Rate: {calibra_cond['predicted_success_rate']:.1f}% {_tag(calibra_cond['success_rate_source'])}\n"
        f"{_THIN}\n"
        f"  🚀  Compute Cost Savings: {compute_savings:.1f}% saved (from GPU-hours, not episode count)\n"
        f"  🎯  Performance Delta: {calibra_cond['predicted_success_rate'] - random_cond['predicted_success_rate']:+.1f}% vs. Random\n"
        f"{_THIN}\n"
        f"  STATUS: {status}\n"
        f"{status_lines}"
        f"{_THIN}\n"
        f"  RECOMMENDED TRAINING COMMAND BRIDGES:\n"
        f"  Copy and run this command to train using Calibra's coreset:\n"
        f"\n"
        f"  $ {lerobot_cmd}\n"
        f"{_THICK}"
    )


def _run_sweep(args, batch, pipeline, raw_report, raw_score, n_total, log) -> None:
    try:
        fractions = sorted({round(float(x), 4) for x in args.fractions.split(",") if x.strip()})
    except ValueError:
        print(
            f"error: --fractions must be a comma-separated list of numbers, got {args.fractions!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not fractions or any(f <= 0.0 or f > 1.0 for f in fractions):
        print("error: --fractions values must each be in the range (0, 1]", file=sys.stderr)
        sys.exit(1)

    elog_table: dict = {}
    if args.experiment_id:
        from calibra.experiment_log import ExperimentLog

        elog_table = ExperimentLog().retention_table(args.experiment_id)

    full_rec = _lookup_measured(elog_table, "full", 100.0)
    full_cond = _condition_result(n_total, n_total, args.base_gpu_hours, raw_score, full_rec)

    rows = []
    for frac in (f for f in fractions if f < 1.0):
        pct = frac * 100.0
        k_size = max(1, round(n_total * frac))

        log(f"[sweep] retention {pct:.0f}%: Calibra coreset selection ...")
        selector = CoresetSelector(keep_fraction=frac)
        prune_res = selector.select(batch, raw_report)
        calibra_n, calibra_score = _diagnose_subset(
            pipeline, batch, prune_res.keep_episode_ids, f"calibra_{pct:.0f}pct", args.policy
        )

        log(f"[sweep] retention {pct:.0f}%: random baseline ...")
        random.seed(42)
        random_ids = random.sample([ep.metadata.episode_id for ep in batch.episodes], k_size)
        random_n, random_score = _diagnose_subset(
            pipeline, batch, random_ids, f"random_{pct:.0f}pct", args.policy
        )

        random_rec = _lookup_measured(elog_table, "random", pct)
        calibra_rec = _lookup_measured(elog_table, "calibra", pct)
        random_cond = _condition_result(random_n, n_total, args.base_gpu_hours, random_score, random_rec)
        calibra_cond = _condition_result(calibra_n, n_total, args.base_gpu_hours, calibra_score, calibra_rec)

        delta = calibra_cond["predicted_success_rate"] - random_cond["predicted_success_rate"]
        rows.append(
            {
                "retention_pct": pct,
                "random": random_cond,
                "calibra": calibra_cond,
                "delta_success_pct": round(delta, 1),
                # Row status covers only this row's own random/calibra measurements —
                # NOT the shared `full` baseline, which would otherwise make every row
                # look "PARTIAL MEASUREMENT" as soon as full is measured once, even if
                # that row's own random/calibra numbers are still entirely simulated.
                "status": _case_study_status([random_cond, calibra_cond]),
            }
        )

    overall_status = _case_study_status(
        [full_cond] + [c for row in rows for c in (row["random"], row["calibra"])]
    )

    summary = {
        "dataset_name": batch.dataset_name,
        "policy_family": args.policy,
        "n_original": n_total,
        "experiment_id": args.experiment_id,
        "fractions": fractions,
        "full": full_cond,
        "rows": rows,
        "overall_status": overall_status,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(_render_sweep_report(batch.dataset_name, args.policy, full_cond, rows, overall_status))


def _render_sweep_report(dataset_name, policy, full_cond, rows, overall_status) -> str:
    def tag(source: str) -> str:
        return "🟢" if source == "measured" else "🟡"

    lines = [
        "",
        _THICK,
        "  CALIBRA RETENTION SWEEP",
        _THICK,
        f"  Dataset: {dataset_name}",
        f"  Policy : {policy.upper()}",
        "  Legend : 🟢 measured from real training   🟡 simulated (heuristic / linear scaling)",
        _THIN,
        f"  {'Retention':>9}  {'Cond':<8} {'Eps':>7} {'Success':>10} {'GPU-hours':>12}",
        f"  {'100%':>9}  {'full':<8} {full_cond['n_episodes']:>7} "
        f"{full_cond['predicted_success_rate']:>7.1f}% {tag(full_cond['success_rate_source'])} "
        f"{full_cond['gpu_hours']:>8.1f}h {tag(full_cond['gpu_hours_source'])}",
    ]
    for row in sorted(rows, key=lambda r: -r["retention_pct"]):
        for cond_name in ("random", "calibra"):
            c = row[cond_name]
            lines.append(
                f"  {row['retention_pct']:>8.0f}%  {cond_name:<8} {c['n_episodes']:>7} "
                f"{c['predicted_success_rate']:>7.1f}% {tag(c['success_rate_source'])} "
                f"{c['gpu_hours']:>8.1f}h {tag(c['gpu_hours_source'])}"
            )

    lines.append(_THIN)
    lines.append("  CALIBRA vs RANDOM (Δ success):")
    for row in sorted(rows, key=lambda r: -r["retention_pct"]):
        d = row["delta_success_pct"]
        verb = "beats" if d > 0 else "trails" if d < 0 else "ties"
        lines.append(
            f"    {row['retention_pct']:>3.0f}%: Calibra {verb} random by {abs(d):.1f}pp "
            f"[{row['status']}]"
        )

    lines.append(_THIN)
    lines.append(f"  OVERALL STATUS: {overall_status}")
    if overall_status == "SIMULATED":
        lines.append("  No measured results yet — this is a prediction, not a case study.")
        lines.append("  Run `calibra experiment record` per condition/retention level after")
        lines.append("  real training, then re-run this sweep with --experiment-id.")
    elif overall_status == "PARTIAL MEASUREMENT":
        lines.append("  Some rows are measured (🟢), others are still simulated (🟡).")
        lines.append("  Do not report this as a validated case study until every row is 🟢.")
    else:
        lines.append("  Full + random + calibra are all measured at every retention level.")
        lines.append("  Safe to report as a validated case study.")
    lines.append(_THICK)
    return "\n".join(lines)
