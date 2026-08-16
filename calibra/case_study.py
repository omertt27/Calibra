"""
calibra.case_study — Case-study report generator for design-partner pilots.

Turns a completed (or in-progress) `calibra experiment record` history for one
experiment_id into the partner-facing report described in the design-partner
protocol: a headline number, a retention curve, a Calibra-vs-random
comparison, and a cost estimate.

Deliberately reads only real measured ExperimentLog data — this is NOT
`calibra benchmark --sweep`, which blends measured and simulated numbers for
internal planning. A report handed to a partner or used in outreach must not
contain heuristic predictions dressed up as evidence, so any protocol gap is
surfaced as an "Open items" section and the report is stamped DRAFT rather
than silently upgraded to a clean validated one.

    calibra case-study --experiment-id partner-a-pusht \\
        --partner "Partner A" --gpu-cost-per-hour 2.50 --out case_study.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from calibra.experiment_log import (
    CONDITIONS,
    PROTOCOL_RETENTION_LEVELS,
    ExperimentLog,
    ExperimentRecord,
)


def _fully_measured(rec: Optional[ExperimentRecord]) -> bool:
    return rec is not None and rec.gpu_hours is not None and rec.eval_success_rate is not None


def _protocol_gaps(log: ExperimentLog, experiment_id: str) -> List[str]:
    """Human-readable list of everything standing between this experiment and
    a fully measured protocol: unrecorded (level, condition) slots, plus
    recorded slots missing gpu_hours or eval_success_rate."""
    table = log.retention_table(experiment_id)
    gaps = [
        f"{level:.0f}% / {cond} — not recorded"
        for level, cond in log.missing_conditions(experiment_id)
    ]
    for level in PROTOCOL_RETENTION_LEVELS:
        for cond_name, rec in table.get(level, {}).items():
            if _fully_measured(rec):
                continue
            missing_fields = []
            if rec.gpu_hours is None:
                missing_fields.append("gpu_hours")
            if rec.eval_success_rate is None:
                missing_fields.append("eval_success_rate")
            gaps.append(
                f"{level:.0f}% / {cond_name} — recorded but missing {', '.join(missing_fields)}"
            )
    return gaps


def _cost(gpu_hours: Optional[float], cost_per_hour: float) -> Optional[float]:
    return None if gpu_hours is None else gpu_hours * cost_per_hour


def generate_case_study(
    log: ExperimentLog,
    experiment_id: str,
    partner_label: str = "",
    gpu_cost_per_hour: float = 2.50,
) -> str:
    table = log.retention_table(experiment_id)
    if not table:
        return f"No records for experiment_id={experiment_id!r}. Nothing to generate."

    records = log.list_records(experiment_id)
    meta = records[0]
    partner = partner_label or meta.partner or experiment_id

    gaps = _protocol_gaps(log, experiment_id)
    status_badge = (
        "VALIDATED — full protocol measured"
        if not gaps
        else f"DRAFT — {len(gaps)} gap(s) open, see Open Items below"
    )

    full_rec = table.get(100.0, {}).get("full")
    full_success = full_rec.eval_success_rate if full_rec and _fully_measured(full_rec) else None
    full_cost = _cost(full_rec.gpu_hours, gpu_cost_per_hour) if full_rec else None

    # Headline: the most aggressive (lowest) retention level with a fully
    # measured Calibra/random pair — this is where Calibra's advantage over
    # random subsampling is expected to be largest, per the protocol.
    headline_level = None
    for level in sorted((lvl for lvl in table if lvl < 100.0)):
        c = table[level].get("calibra")
        r = table[level].get("random")
        if _fully_measured(c) and _fully_measured(r):
            headline_level = level
            break

    lines: List[str] = [f"# Calibra Case Study — {partner}", "", f"**Status:** {status_badge}", ""]
    lines.append(f"- **Experiment ID:** `{experiment_id}`")
    if meta.dataset_name and meta.dataset_name != "unknown":
        lines.append(f"- **Dataset:** {meta.dataset_name}")
    if meta.embodiment:
        lines.append(f"- **Embodiment:** {meta.embodiment}")
    if meta.task:
        lines.append(f"- **Task:** {meta.task}")
    if meta.policy_family:
        lines.append(f"- **Policy family:** {meta.policy_family}")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    if headline_level is not None:
        c = table[headline_level]["calibra"]
        r = table[headline_level]["random"]
        c_cost = _cost(c.gpu_hours, gpu_cost_per_hour)

        headline = (
            f"At **{headline_level:.0f}% retention**, Calibra's selected subset reached "
            f"**{c.eval_success_rate:.1%} eval success**"
        )
        if full_success:
            headline += (
                f" (**{c.eval_success_rate / full_success:.0%}** of the full-dataset baseline)"
            )
        headline += f", vs. **{r.eval_success_rate:.1%}** for a random subset of the same size."
        lines.append(headline)

        if c_cost is not None and full_cost:
            savings = 100.0 * (1.0 - c_cost / full_cost)
            lines.append("")
            lines.append(
                f"Estimated compute cost at this retention level: **${c_cost:,.0f}** vs. "
                f"**${full_cost:,.0f}** for the full dataset — **{savings:.0f}%** lower, "
                f"at ${gpu_cost_per_hour:.2f}/GPU-hour (an assumed rate, not a partner-billed figure)."
            )
    else:
        lines.append(
            "_No retention level yet has a fully measured Calibra + random pair — "
            "the headline number is withheld until at least one level is complete._"
        )
    lines.append("")

    lines.append("## Retention curve")
    lines.append("")
    lines.append("| Retention | Condition | Episodes | Eval success | GPU-hours | Est. cost |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for level in sorted(table.keys(), reverse=True):
        for cond in CONDITIONS:
            rec = table[level].get(cond)
            if rec is None:
                continue
            success = f"{rec.eval_success_rate:.1%}" if rec.eval_success_rate is not None else "—"
            gpu = f"{rec.gpu_hours:.1f}" if rec.gpu_hours is not None else "—"
            cost = _cost(rec.gpu_hours, gpu_cost_per_hour)
            cost_str = f"${cost:,.0f}" if cost is not None else "—"
            mark = "" if _fully_measured(rec) else " *(partial)*"
            lines.append(
                f"| {level:.0f}% | {cond}{mark} | {rec.n_episodes} | {success} | {gpu} | {cost_str} |"
            )
    lines.append("")

    deltas = log.calibra_vs_random(experiment_id)
    measured_deltas = {k: v for k, v in deltas.items() if v is not None}
    if measured_deltas:
        lines.append("## Calibra vs. random (same retention, same size)")
        lines.append("")
        lines.append(
            "This is the comparison that matters: not whether a smaller dataset trains "
            "faster (any subset does that), but whether Calibra's *selection* beats a "
            "random subset of identical size."
        )
        lines.append("")
        lines.append("| Retention | Δ eval success (Calibra − random) |")
        lines.append("|---:|---:|")
        for level in sorted(measured_deltas, reverse=True):
            lines.append(f"| {level:.0f}% | {measured_deltas[level]:+.1%} |")
        lines.append("")

    if gaps:
        lines.append("## Open items before this is a validated case study")
        lines.append("")
        lines.extend(f"- {g}" for g in gaps)
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Three conditions per retention level: **Full** (100% data, baseline), "
        "**Random** (uniform random subset, tests whether *any* subset works), "
        "**Calibra** (Calibra-selected subset, tests whether Calibra's selection beats "
        "random). Policy architecture, hyperparameters, training steps, seeds, and eval "
        "procedure held identical across conditions. GPU-hours and eval success rates are "
        "measured from the partner's own training runs via `calibra experiment record` — "
        "not simulated. Cost figures are GPU-hours times an assumed $/hour rate, not a "
        "partner billing figure, and are labeled as such above."
    )
    lines.append("")
    lines.append(
        "_Generated by `calibra case-study`. Source data never leaves local storage "
        f"(`{log.path}`) — nothing here is uploaded or synced automatically._"
    )

    return "\n".join(lines)


def run_case_study(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra case-study",
        description=(
            "Render a completed (or in-progress) design-partner experiment into a "
            "partner-facing case-study report: headline number, retention curve, "
            "Calibra-vs-random comparison, and cost estimate. Reads only real measured "
            "data recorded via `calibra experiment record` — never the simulated numbers "
            "from `calibra benchmark`."
        ),
    )
    p.add_argument("--experiment-id", required=True)
    p.add_argument(
        "--partner",
        default="",
        help="Display name for the partner (defaults to the recorded partner label, or the experiment id)",
    )
    p.add_argument(
        "--gpu-cost-per-hour",
        type=float,
        default=2.50,
        help="Assumed $/GPU-hour for the cost estimate columns (default: 2.50)",
    )
    p.add_argument(
        "--out", default=None, help="Write the markdown report to this path instead of stdout"
    )
    p.add_argument("--path", default=None, help="Override the default ~/.calibra/experiments.jsonl")
    args = p.parse_args(argv)

    log = ExperimentLog(path=Path(args.path) if args.path else None)
    report = generate_case_study(
        log=log,
        experiment_id=args.experiment_id,
        partner_label=args.partner,
        gpu_cost_per_hour=args.gpu_cost_per_hour,
    )

    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote case study to {args.out}", file=sys.stderr)
    else:
        print(report)
