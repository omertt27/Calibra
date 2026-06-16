#!/usr/bin/env python3
"""
Batch dataset profiler — profiles multiple datasets from a YAML manifest.

Reads a YAML file listing datasets to profile and calls the profiling pipeline
for each one, writing reference JSON files to calibra/references/.

Usage:
    python scripts/profile_batch.py datasets_to_profile.yaml
    python scripts/profile_batch.py datasets_to_profile.yaml --dry-run
    python scripts/profile_batch.py datasets_to_profile.yaml --skip-existing

Manifest format (YAML):
    datasets:
      - id: lerobot/bridge
        control_mode: velocity
        note: "BridgeData V2, real hardware, varied tasks"
        out: calibra/references/bridge.json   # optional, auto-derived if omitted

      - id: lerobot/so100_pick_place
        control_mode: position
        gripper_dims: "-1"
        note: "Low-cost SO-100 arm, real hardware"

      - id: /data/local_dataset.h5
        format: hdf5
        control_mode: position
        note: "Local HDF5 collection"
        out: calibra/references/local_robot.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    print(
        "error: PyYAML is required. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)

_REPO = Path(__file__).parent.parent
_PROFILE_SCRIPT = _REPO / "scripts" / "profile_dataset.py"
_REFS_DIR = _REPO / "calibra" / "references"


def _derive_out(dataset_id: str) -> Path:
    """Derive a reference JSON output path from the dataset ID."""
    safe_name = dataset_id.replace("/", "_").replace("hf://", "").replace(":", "_")
    return _REFS_DIR / f"{safe_name}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", help="Path to YAML manifest file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets whose reference JSON already exists",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"error: could not parse YAML manifest: {e}", file=sys.stderr)
        sys.exit(1)

    datasets = manifest.get("datasets", [])
    if not datasets:
        print("error: manifest has no 'datasets' entries", file=sys.stderr)
        sys.exit(1)

    print(f"Manifest: {manifest_path}  ({len(datasets)} datasets)")
    print()

    n_skipped = 0
    n_ok = 0
    n_fail = 0

    for entry in datasets:
        ds_id = entry.get("id", "")
        if not ds_id:
            print(f"  [skip] Entry missing 'id': {entry}", file=sys.stderr)
            continue

        out_path = Path(entry["out"]) if "out" in entry else _derive_out(ds_id)

        if args.skip_existing and out_path.exists():
            print(f"  [skip] {ds_id} → {out_path.name} (already exists)")
            n_skipped += 1
            continue

        cmd = [args.python, str(_PROFILE_SCRIPT), ds_id]

        if "control_mode" in entry:
            cmd += ["--control-mode", entry["control_mode"]]

        if "format" in entry:
            cmd += ["--format", entry["format"]]

        if "gripper_dims" in entry:
            cmd += ["--gripper-dims", str(entry["gripper_dims"])]

        if "note" in entry:
            cmd += ["--note", entry["note"]]

        cmd += ["--out", str(out_path)]

        print(f"  → {ds_id}")
        print(f"    out:  {out_path}")
        print(f"    cmd:  {' '.join(cmd)}")

        if args.dry_run:
            print()
            continue

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            print(f"    ✅ done\n")
            n_ok += 1
        else:
            print(f"    ❌ failed (exit {result.returncode})\n", file=sys.stderr)
            n_fail += 1

    if not args.dry_run:
        print("─" * 50)
        print(f"  Done: {n_ok} succeeded  ·  {n_fail} failed  ·  {n_skipped} skipped")
        if n_fail:
            print(
                "\nNext: add evidence entries to calibra/claims/*.json for each "
                "successful profile, then run:\n"
                "  python scripts/generate_claims_doc.py"
            )
            sys.exit(1)
    else:
        print(f"\n[dry-run] Would profile {len(datasets) - n_skipped} dataset(s).")


if __name__ == "__main__":
    main()
