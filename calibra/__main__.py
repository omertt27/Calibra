"""
CLI entry point.

    python -m calibra <path> [--policy FAMILY] [--format FORMAT] [--json]

Examples:
    python -m calibra /data/robot_demos.h5
    python -m calibra /data/lerobot_ds --policy diffusion
    python -m calibra /data/demo.h5 --policy act --json
"""
from __future__ import annotations

import argparse
import sys

from calibra.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="calibra",
        description="Calibra — dataset reliability diagnostics for robotics IL",
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
    else:
        print(report.summary())


def _get_reader(format_name: str):
    from calibra.ingestion import hdf5, lerobot, rlds, mcap  # noqa: F401 trigger register
    from calibra.ingestion.adapters.hdf5   import HDF5Reader
    from calibra.ingestion.adapters.lerobot import LeRobotReader
    from calibra.ingestion.adapters.rlds   import RLDSReader
    from calibra.ingestion.adapters.mcap   import MCAPReader

    mapping = {
        "hdf5":    HDF5Reader,
        "lerobot": LeRobotReader,
        "rlds":    RLDSReader,
        "mcap":    MCAPReader,
    }
    return mapping[format_name]()


if __name__ == "__main__":
    main()
