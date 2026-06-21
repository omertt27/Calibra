"""
calibra retarget — Convert absolute EEF actions to relative deltas for GR00T N1.7+.

NVIDIA GR00T N1.7+ uses a Relative End-Effector (EEF) action space: each action
is a 6-DoF delta transformation expressed in the robot's *current* EEF frame, not
in world space. Isaac Lab / robomimic HDF5 datasets record actions in absolute
world-frame Cartesian coordinates.

This command converts a dataset's per-episode EEF observations into relative 6-DoF
action deltas and writes them to a directory of .npz files (one per episode).

Usage:
    calibra retarget /data/isaac_lab_demos.h5
    calibra retarget /data/demos.h5 --out /data/retargeted/ --format isaac_lab
    calibra retarget /data/demos.h5 --obs-key-pos robot0_eef_pos \\
                                     --obs-key-quat robot0_eef_quat \\
                                     --pad

Output (per-episode .npz files):
    relative_actions : (T−1, 6) or (T, 6) if --pad
        Each row is [dx, dy, dz, droll, dpitch, dyaw] in the current EEF frame.
        Position units match the input; rotation is in radians.
    episode_id : str
    source_path : str

Exit codes:
    0  All episodes converted successfully.
    1  Error loading dataset.
    2  EEF observation keys not found in any episode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Candidate observation key names for EEF position and quaternion.
# Searched in order — first match wins.
_POS_CANDIDATES  = [
    "eef_pos", "robot0_eef_pos", "ee_pos", "end_effector_pos",
    "obs/robot0_eef_pos", "observation.eef_pos",
]
_QUAT_CANDIDATES = [
    "eef_quat", "robot0_eef_quat", "ee_quat", "end_effector_quat",
    "obs/robot0_eef_quat", "observation.eef_quat",
]


def run_retarget(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra retarget",
        description=(
            "Convert absolute EEF actions to relative 6-DoF deltas for GR00T N1.7+. "
            "Reads EEF position + quaternion from episode observations, writes one "
            ".npz per episode containing 'relative_actions' of shape (T−1, 6)."
        ),
    )
    p.add_argument("path", help="Path or Hub ID of the source dataset")
    p.add_argument(
        "--out", "-o",
        metavar="DIR",
        default="retargeted",
        help="Output directory for .npz files (default: ./retargeted/)",
    )
    p.add_argument(
        "--obs-key-pos",
        metavar="KEY",
        default=None,
        help=(
            "Observation key for EEF world-frame position, shape (T, 3). "
            f"Auto-detected if not set (tries: {', '.join(_POS_CANDIDATES[:3])}, …)."
        ),
    )
    p.add_argument(
        "--obs-key-quat",
        metavar="KEY",
        default=None,
        help=(
            "Observation key for EEF quaternion [qx, qy, qz, qw], shape (T, 4). "
            f"Auto-detected if not set (tries: {', '.join(_QUAT_CANDIDATES[:3])}, …)."
        ),
    )
    p.add_argument(
        "--pad",
        action="store_true",
        help=(
            "Pad the output to T rows by appending a zero row at the end. "
            "Required when the downstream policy expects fixed-length (T, 6) actions."
        ),
    )
    p.add_argument(
        "--format", "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--json", "-j",
        action="store_true",
        help="Print a JSON summary to stdout after conversion.",
    )
    p.add_argument(
        "--urdf",
        metavar="FILE",
        default=None,
        help="Path to robot URDF model to run joint limit auditing.",
    )
    p.add_argument(
        "--joint-key",
        metavar="KEY",
        default=None,
        help="Observation key for joint positions (e.g. 'joint_pos').",
    )
    args = p.parse_args(argv)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://"):]

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {dataset_path!r} ...")

    try:
        from calibra.ingestion.registry import load
        batch = load(dataset_path, reader=reader)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {batch.n_episodes} episodes  ·  {batch.n_samples} steps")

    from calibra.kinematics.retarget import retarget_episode_eef

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped   = 0
    skipped_ids: list[str] = []
    converted_ids: list[str] = []

    checker = None
    if args.urdf:
        from calibra.kinematics.checker import KinematicURDFChecker
        checker = KinematicURDFChecker(args.urdf)
        log(f"Loaded URDF from {args.urdf}. Auditing kinematic joint limits...")

    for ep in batch.episodes:
        ep_id = ep.metadata.episode_id

        # Resolve position key.
        pos_key = args.obs_key_pos or _find_key(ep.observations, _POS_CANDIDATES)
        quat_key = args.obs_key_quat or _find_key(ep.observations, _QUAT_CANDIDATES)

        if pos_key is None or quat_key is None:
            log(
                f"  skip episode {ep_id!r} — EEF obs keys not found "
                f"(pos_key={pos_key!r}, quat_key={quat_key!r})"
            )
            skipped += 1
            skipped_ids.append(ep_id)
            continue

        eef_pos  = ep.observations[pos_key]
        eef_quat = ep.observations[quat_key]

        try:
            rel_actions = retarget_episode_eef(eef_pos, eef_quat)   # (T-1, 6)
        except Exception as exc:
            log(f"  skip episode {ep_id!r} — retarget failed: {exc}")
            skipped += 1
            skipped_ids.append(ep_id)
            continue

        if checker is not None:
            violations = checker.check_episode(ep, joint_key=args.joint_key)
            if violations:
                log(f"  [kinematic audit] Warning: Episode {ep_id!r} violated joint limits:")
                for joint, vios in violations.items():
                    log(f"    - {joint}: {len(vios)} violations (types: {set(v[2] for v in vios)})")

        if args.pad:
            zero_row = np.zeros((1, rel_actions.shape[1]), dtype=rel_actions.dtype)
            rel_actions = np.concatenate([rel_actions, zero_row], axis=0)  # (T, 6)

        safe_id = ep_id.replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{safe_id}.npz"
        np.savez_compressed(
            out_path,
            relative_actions=rel_actions,
            episode_id=np.bytes_(ep_id),
            source_path=np.bytes_(str(dataset_path)),
        )
        converted += 1
        converted_ids.append(ep_id)

    if converted == 0:
        print(
            "error: EEF observation keys not found in any episode. "
            f"Tried pos keys: {_POS_CANDIDATES}. "
            f"Tried quat keys: {_QUAT_CANDIDATES}. "
            "Use --obs-key-pos and --obs-key-quat to specify them explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)

    shape_str = f"(T{'−1' if not args.pad else ''}, 6)"
    print(
        f"\n{'━' * 56}\n"
        f"  calibra retarget — {batch.dataset_name}\n"
        f"{'━' * 56}\n"
        f"  Episodes converted : {converted}\n"
        f"  Episodes skipped   : {skipped}\n"
        f"  Output directory   : {out_dir.resolve()}\n"
        f"  Action shape       : {shape_str}  [dx, dy, dz, droll, dpitch, dyaw]\n"
        f"  Rotation units     : radians (intrinsic XYZ)\n"
        f"{'━' * 56}"
    )

    if args.json:
        summary = {
            "dataset":       batch.dataset_name,
            "n_converted":   converted,
            "n_skipped":     skipped,
            "out_dir":       str(out_dir.resolve()),
            "action_shape":  [None, 6],
            "padded":        args.pad,
            "converted_ids": converted_ids,
            "skipped_ids":   skipped_ids,
        }
        print(json.dumps(summary, indent=2))


def _find_key(obs: dict, candidates: list[str]) -> str | None:
    """Return the first candidate key present in obs, or None."""
    for key in candidates:
        if key in obs:
            return key
    return None
