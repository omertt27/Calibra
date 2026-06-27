"""calibra.cloud.push — CLI handler for `calibra cloud push`."""

from __future__ import annotations

import argparse
import sys


def run_cloud(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra cloud",
        description="Calibra Cloud commands. Requires `calibra login`.",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    push_p = sub.add_parser(
        "push",
        help="Run diagnostics on a dataset and upload the report to Calibra Cloud",
    )
    push_p.add_argument("path", help="Dataset path or HuggingFace Hub ID")
    push_p.add_argument(
        "--policy", "-p",
        metavar="FAMILY",
        default="generic",
        help="Target policy family (e.g. 'diffusion', 'act', 'gr00t')",
    )
    push_p.add_argument(
        "--format", "-f",
        metavar="FORMAT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )

    args = p.parse_args(argv)

    if args.subcommand == "push":
        _push(args)


def _push(args: argparse.Namespace) -> None:
    from calibra.cloud.client import CalibraCloudClient, CalibraCloudError
    from calibra.pipeline import Pipeline

    try:
        client = CalibraCloudClient()
    except CalibraCloudError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    print(f"Analyzing {args.path!r} ...", file=sys.stderr)
    try:
        report = Pipeline().analyze_path(args.path, policy_family=args.policy, reader=reader)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"  {report.n_episodes} episodes · {report.n_samples} steps",
        file=sys.stderr,
    )

    try:
        url = client.push_report(report.model_dump(), dataset_name=report.dataset_name)
        print(f"✓ Report uploaded: {url}")
    except CalibraCloudError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
