"""
CLI entry point.

    python -m calibra <path> [--policy FAMILY] [--format FORMAT] [--json] [--strict]
    python -m calibra compare <path> <reference> [--format FORMAT]
    python -m calibra certify <path> [--reference REF] [--policy FAMILY]
    python -m calibra prune <path> --keep FRACTION [--out coreset_index.json]
    python -m calibra corrupt <path> --drop-frames RATE [--add-jitter-ms STD] ...
    python -m calibra retarget <path> [--out DIR] [--obs-key-pos KEY] [--obs-key-quat KEY]

Examples:
    python -m calibra /data/robot_demos.h5
    python -m calibra /data/lerobot_ds --policy diffusion
    python -m calibra /data/demo.h5 --policy act --json
    python -m calibra compare /data/my_demos pusht
    python -m calibra compare hf://lerobot/my_dataset aloha --format lerobot
    python -m calibra certify /data/my_demos
    python -m calibra certify /data/my_demos --reference aloha --policy diffusion
    python -m calibra prune /data/100k_episodes --keep 0.3 --out coreset.json
    python -m calibra prune /data/my_ds --keep 0.5 --quality-only
    python -m calibra prune /data/my_ds --keep 0.3 --entropy-weight 0.4 --policy gr00t
    python -m calibra corrupt lerobot/pusht --drop-frames 0.10
    python -m calibra corrupt /data/robot.h5 --inject-spikes 0.05 --add-jitter-ms 50
    python -m calibra retarget /data/isaac_lab.h5 --out retargeted/ --pad
    python -m calibra retarget /data/demos.h5 --obs-key-pos robot0_eef_pos \\
                                               --obs-key-quat robot0_eef_quat

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

    if len(sys.argv) > 1 and sys.argv[1] == "certify":
        from calibra.certify import run_certify
        run_certify(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "prune":
        from calibra.prune import run_prune
        run_prune(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "corrupt":
        from calibra.corrupt import run_corrupt
        run_corrupt(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "retarget":
        from calibra.retarget import run_retarget
        run_retarget(sys.argv[2:])
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
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
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
    from calibra.ingestion import isaac_lab, hdf5, lerobot, rlds, mcap  # noqa: F401
    from calibra.ingestion.adapters.isaac_lab import IsaacLabReader
    from calibra.ingestion.adapters.hdf5      import HDF5Reader
    from calibra.ingestion.adapters.lerobot   import LeRobotReader
    from calibra.ingestion.adapters.rlds      import RLDSReader
    from calibra.ingestion.adapters.mcap      import MCAPReader

    mapping = {
        "isaac_lab": IsaacLabReader,
        "hdf5":      HDF5Reader,
        "lerobot":   LeRobotReader,
        "rlds":      RLDSReader,
        "mcap":      MCAPReader,
    }
    return mapping[format_name]()


if __name__ == "__main__":
    main()
