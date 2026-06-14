"""
CLI entry point.

    python -m calibra <path> [--policy FAMILY] [--format FORMAT] [--json] [--strict]
    python -m calibra compare <path> <reference> [--format FORMAT]

Examples:
    python -m calibra /data/robot_demos.h5
    python -m calibra /data/lerobot_ds --policy diffusion
    python -m calibra /data/demo.h5 --policy act --json
    python -m calibra compare /data/my_demos pusht
    python -m calibra compare lerobot/my_dataset aloha --format lerobot

Exit codes:
    0  No critical issues found.
    1  One or more CRITICAL flags or severe episode outliers detected.
       (With --strict: exits 1 on WARNING too.)
"""
from __future__ import annotations

import argparse
import sys

from calibra.pipeline import Pipeline
from calibra.schema.report import RiskLevel


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        from calibra.compare import run_compare
        run_compare(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        prog="calibra",
        description="Calibra — dataset reliability diagnostics for robotics IL",
        epilog="Run 'calibra compare <path> <reference>' to compare against a reference profile.",
    )
    parser.add_argument("path", help="Path to dataset (file or directory)")
    parser.add_argument(
        "--policy", "-p",
        metavar="FAMILY",
        help="Target policy family for conditioned hints "
             "(e.g. 'diffusion', 'act', 'transformer')",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output DiagnosticReport as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--format", "-f",
        metavar="FORMAT",
        choices=["hdf5", "lerobot", "rlds", "mcap"],
        help="Force a specific format adapter (default: auto-detect)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on WARNING in addition to CRITICAL",
    )
    parser.add_argument(
        "--no-anomalies",
        action="store_true",
        help="Skip per-episode outlier detection (faster, aggregate flags only)",
    )

    args = parser.parse_args()

    reader = None
    if args.format:
        reader = _get_reader(args.format)

    try:
        report = Pipeline().analyze_path(
            args.path,
            policy_family=args.policy,
            reader=reader,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(report.model_dump_json(indent=2))
        sys.exit(_exit_code(report, strict=args.strict))

    print(report.summary())

    if not args.no_anomalies:
        from calibra.anomalies import find_outliers, render
        outliers = find_outliers(report)
        if outliers:
            print()
            print(render(outliers, report.n_episodes))

    sys.exit(_exit_code(report, strict=args.strict))


def _exit_code(report, strict: bool = False) -> int:
    if report.flags_at_level(RiskLevel.CRITICAL):
        return 1
    if strict and report.flags_at_level(RiskLevel.WARNING):
        return 1
    return 0


def _get_reader(format_name: str):
    from calibra.ingestion import hdf5, lerobot, rlds, mcap  # noqa: F401 trigger register
    from calibra.ingestion.adapters.hdf5    import HDF5Reader
    from calibra.ingestion.adapters.lerobot import LeRobotReader
    from calibra.ingestion.adapters.rlds    import RLDSReader
    from calibra.ingestion.adapters.mcap    import MCAPReader

    mapping = {
        "hdf5":    HDF5Reader,
        "lerobot": LeRobotReader,
        "rlds":    RLDSReader,
        "mcap":    MCAPReader,
    }
    return mapping[format_name]()


if __name__ == "__main__":
    main()
