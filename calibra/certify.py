"""
calibra certify — structured pass/fail certification for robot datasets.

Runs the full diagnostic pipeline and grades the dataset on a three-level scale:

  CERTIFIED             — no CRITICAL or WARNING flags
  PROVISIONALLY CERTIFIED — no CRITICAL flags; WARNING flags present
  NOT CERTIFIED           — one or more CRITICAL flags

If --reference is given, also runs calibra compare and incorporates the
reference-relative findings into the remediation list.

Usage:
    calibra certify /path/to/dataset
    calibra certify /path/to/dataset --reference aloha --policy diffusion
    calibra certify lerobot/pusht --format lerobot
    calibra certify /data/my_ds --reference pusht --strict

Exit codes:
    0  CERTIFIED
    1  PROVISIONALLY CERTIFIED (warnings only)
    2  NOT CERTIFIED (critical failures)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport, RiskLevel


_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN  = "─" * _WIDTH


# ── grading ──────────────────────────────────────────────────────────────────

def _grade(report: DiagnosticReport) -> tuple[str, int]:
    """Return (grade_label, exit_code)."""
    criticals = report.flags_at_level(RiskLevel.CRITICAL)
    warnings  = report.flags_at_level(RiskLevel.WARNING)
    if criticals:
        return "NOT CERTIFIED", 2
    if warnings:
        return "PROVISIONALLY CERTIFIED", 1
    return "CERTIFIED", 0


def _grade_banner(grade: str) -> str:
    if grade == "CERTIFIED":
        return f"  ✓  {grade}"
    if grade == "PROVISIONALLY CERTIFIED":
        return f"  ⚠  {grade}"
    return f"  ✗  {grade}"


# ── remediation ───────────────────────────────────────────────────────────────

def _remediation_steps(report: DiagnosticReport) -> list[str]:
    steps: list[str] = []

    for result in report.analyzer_results:
        for flag in result.flags:
            if flag.level in (RiskLevel.CRITICAL, RiskLevel.WARNING):
                metric = flag.metric
                impl = flag.implication or ""
                steps.append(f"[{flag.level.value}] {metric}: {impl.strip()}")

    return steps


# ── certificate rendering ─────────────────────────────────────────────────────

def render_certificate(
    report: DiagnosticReport,
    grade: str,
    reference_name: Optional[str],
    extra_steps: Optional[list[str]] = None,
) -> str:
    lines = [
        _THICK,
        "  CALIBRA CERTIFICATION REPORT",
        _THICK,
        "",
        f"  Dataset  : {Path(report.source_path).name}",
        f"  Episodes : {report.n_episodes}",
        f"  Steps    : {report.n_samples}",
    ]
    if report.policy_family:
        lines.append(f"  Policy   : {report.policy_family}")
    if reference_name:
        lines.append(f"  Reference: {reference_name}")
    lines += ["", _THIN, ""]

    lines.append(_grade_banner(grade))
    lines.append("")

    # Summary of flags
    criticals = report.flags_at_level(RiskLevel.CRITICAL)
    warnings  = report.flags_at_level(RiskLevel.WARNING)

    if criticals:
        lines.append("  CRITICAL issues:")
        for f in criticals:
            lines.append(f"    • {f.metric}: {f.interpretation}")
        lines.append("")
    if warnings:
        lines.append("  Warnings:")
        for f in warnings:
            lines.append(f"    • {f.metric}: {f.interpretation}")
        lines.append("")

    # Remediation
    steps = _remediation_steps(report)
    if extra_steps:
        steps.extend(extra_steps)

    if steps:
        lines.append(_THIN)
        lines.append("  REMEDIATION CHECKLIST")
        lines.append(_THIN)
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
        lines.append("")

    lines.append(_THICK)
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_certify(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra certify",
        description="Grade a dataset and produce a certification report.",
    )
    p.add_argument("path", help="Path or Hub ID of the dataset to certify")
    p.add_argument(
        "--reference", "-r",
        metavar="REF",
        help="Optional reference profile to compare against (e.g. 'pusht', 'aloha')",
    )
    p.add_argument(
        "--policy", metavar="FAMILY",
        help="Target policy family for conditioned hints (e.g. 'diffusion', 'act')",
    )
    p.add_argument(
        "--format", "-f",
        metavar="FMT",
        choices=["hdf5", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat WARNING flags as certification failures (exit code 2)",
    )
    p.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output report as JSON instead of human-readable text",
    )
    args = p.parse_args(argv)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://"):]

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    log = lambda msg: print(msg, file=sys.stderr, flush=True)
    log(f"Certifying {dataset_path!r} ...")

    try:
        report: DiagnosticReport = Pipeline().analyze_path(
            dataset_path,
            policy_family=args.policy,
            reader=reader,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    log(f"  {report.n_episodes} episodes  ·  {report.n_samples} steps")

    grade, exit_code = _grade(report)
    if args.strict and exit_code == 1:
        exit_code = 2

    # Optional comparison steps
    extra_steps: list[str] = []
    if args.reference:
        try:
            from calibra.compare import load_reference, metrics_from_report, metrics_from_reference
            ref_data    = load_reference(args.reference)
            your_m      = metrics_from_report(report)
            ref_m       = metrics_from_reference(ref_data)
            vd_delta    = ((your_m["vel_disc_rate"] or 0) - (ref_m["vel_disc_rate"] or 0))
            if vd_delta > 0.05:
                extra_steps.append(
                    f"Velocity discontinuity {vd_delta:+.1%} above {args.reference} — "
                    "inspect hardware communication loop."
                )
            entropy = your_m.get("action_entropy")
            if entropy is not None and entropy < 3.0:
                extra_steps.append(
                    f"Action entropy {entropy:.2f} bits/dim < 3.0 — "
                    "collect more diverse demonstrations."
                )
        except Exception as e:
            log(f"  (reference comparison skipped: {e})")

    if args.json:
        import json
        out = {
            "grade": grade,
            "exit_code": exit_code,
            "report": report.model_dump(),
        }
        print(json.dumps(out, indent=2))
    else:
        print(render_certificate(report, grade, args.reference, extra_steps))

    sys.exit(exit_code)