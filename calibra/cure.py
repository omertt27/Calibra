"""
calibra cure — physical and kinematic remediation for robotics datasets.

Automatically applies data-cleaning filters to raw datasets based on diagnostic flags:
  - Trajectory smoothing via Savitzky-Golay to remove jerk spikes and discontinuities.
  - Uniform temporal interpolation (resampling) to resolve packet drops and timing jitter.
  - Dead-time trimming to cut out trailing/leading static segments where no motion occurs.

Usage:
    calibra cure /path/to/dataset --remedy smooth,interpolate,trim --out cured_dataset/
    calibra cure lerobot/pusht --hz 10 --out cured/ --format lerobot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from calibra.pipeline import Pipeline


def smooth_actions(actions: np.ndarray, window_len: int = 7, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth actions, with a moving average fallback."""
    if len(actions) <= window_len:
        window_len = len(actions)
        if window_len % 2 == 0:
            window_len -= 1
    if window_len < 3:
        return actions

    try:
        from scipy.signal import savgol_filter

        return savgol_filter(actions, window_length=window_len, polyorder=polyorder, axis=0)
    except ImportError:
        # Fallback to simple 1D moving average filter
        kernel = np.ones(window_len) / window_len
        smoothed = np.copy(actions)
        for d in range(actions.shape[1]):
            smoothed[:, d] = np.convolve(actions[:, d], kernel, mode="same")
        return smoothed


def interpolate_episode(
    timestamps: np.ndarray,
    actions: np.ndarray,
    observations: dict[str, np.ndarray],
    target_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Resample the episode to a uniform frequency, interpolating actions and observations."""
    if len(timestamps) < 2:
        return timestamps, actions, observations

    if target_hz is None:
        dt = np.diff(timestamps)
        mean_dt = np.mean(dt)
        target_hz = 1.0 / max(mean_dt, 1e-5)

    t_start, t_end = timestamps[0], timestamps[-1]
    n_steps = int(round((t_end - t_start) * target_hz)) + 1
    if n_steps < 2:
        return timestamps, actions, observations

    t_new = np.linspace(t_start, t_end, n_steps)

    # Actions interpolation
    actions_new = np.zeros((n_steps, actions.shape[1]), dtype=actions.dtype)
    for d in range(actions.shape[1]):
        actions_new[:, d] = np.interp(t_new, timestamps, actions[:, d])

    # Observations interpolation
    obs_new = {}
    for key, val in observations.items():
        if val.size == 0:
            obs_new[key] = val
        elif val.ndim == 1:
            obs_new[key] = np.interp(t_new, timestamps, val)
        elif val.ndim == 2:
            val_new = np.zeros((n_steps, val.shape[1]), dtype=val.dtype)
            for d in range(val.shape[1]):
                val_new[:, d] = np.interp(t_new, timestamps, val[:, d])
            obs_new[key] = val_new
        elif val.ndim >= 3:
            # Multi-dimensional (e.g. camera image frames) - sample closest frame in time
            closest_indices = [np.argmin(np.abs(timestamps - t)) for t in t_new]
            obs_new[key] = val[closest_indices]
        else:
            obs_new[key] = val

    return t_new, actions_new, obs_new


def trim_dead_time(
    timestamps: np.ndarray,
    actions: np.ndarray,
    observations: dict[str, np.ndarray],
    threshold: float = 5e-4,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Remove trailing and leading static segments where action rate of change is zero."""
    if len(timestamps) < 5:
        return timestamps, actions, observations

    diff_actions = np.diff(actions, axis=0)
    diff_t = np.diff(timestamps)[:, None]
    velocities = np.abs(diff_actions / np.maximum(diff_t, 1e-5))
    speed = np.mean(velocities, axis=1)

    moving_indices = np.where(speed > threshold)[0]
    if len(moving_indices) == 0:
        return timestamps, actions, observations

    start_idx = max(0, moving_indices[0] - 2)
    end_idx = min(len(timestamps), moving_indices[-1] + 3)

    if end_idx <= start_idx + 2:
        return timestamps, actions, observations

    trimmed_t = timestamps[start_idx:end_idx]
    trimmed_actions = actions[start_idx:end_idx]
    trimmed_obs = {key: val[start_idx:end_idx] for key, val in observations.items()}

    return trimmed_t, trimmed_actions, trimmed_obs


def run_cure(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra cure",
        description="Remediate kinematic anomalies and timing inconsistencies in a robot dataset.",
    )
    p.add_argument("path", help="Path or Hub ID of the source dataset")
    p.add_argument(
        "--remedy",
        default="smooth,interpolate,trim",
        help="Comma-separated remedies to apply (smooth, interpolate, trim)",
    )
    p.add_argument(
        "--out",
        "-o",
        metavar="DIR",
        default="cured_dataset",
        help="Output directory for cured per-episode .npz files",
    )
    p.add_argument(
        "--hz",
        type=float,
        default=None,
        help="Target control frequency in Hz for interpolation (default: auto-detected)",
    )
    p.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--trim-threshold",
        type=float,
        default=5e-4,
        help="Movement threshold for dead-time trimming",
    )
    args = p.parse_args(argv)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://") :]

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
    log("Running diagnostics to check anomaly states ...")
    try:
        Pipeline().run(batch)
    except Exception as exc:
        print(f"error running diagnostic check: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    remedies = [r.strip().lower() for r in args.remedy.split(",")]
    log(f"Applying remedies: {remedies} ...")

    cured_count = 0
    metadata_log = []

    for ep in batch.episodes:
        ep_id = ep.metadata.episode_id
        t = ep.timestamps
        actions = ep.actions
        obs = ep.observations

        orig_steps = len(t)
        orig_hz = 1.0 / np.mean(np.diff(t)) if len(t) > 1 else 0

        # Apply interpolation
        if "interpolate" in remedies:
            t, actions, obs = interpolate_episode(t, actions, obs, target_hz=args.hz)

        # Apply smoothing
        if "smooth" in remedies:
            actions = smooth_actions(actions)

        # Apply dead-time trimming
        if "trim" in remedies:
            t, actions, obs = trim_dead_time(t, actions, obs, threshold=args.trim_threshold)

        new_steps = len(t)
        new_hz = 1.0 / np.mean(np.diff(t)) if len(t) > 1 else 0

        # Save cured episode
        safe_id = ep_id.replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{safe_id}.npz"

        # Prep variables for saving
        save_data = {
            "timestamps": t,
            "actions": actions,
            "episode_id": np.bytes_(ep_id),
        }
        for k, v in obs.items():
            save_data[f"obs_{k}"] = v

        np.savez_compressed(out_path, **save_data)
        cured_count += 1

        metadata_log.append(
            {
                "episode_id": ep_id,
                "original_steps": orig_steps,
                "original_hz": round(orig_hz, 1),
                "cured_steps": new_steps,
                "cured_hz": round(new_hz, 1),
                "saved_file": str(out_path.name),
            }
        )

    # Write summary manifest file
    with open(out_dir / "cure_manifest.json", "w") as f:
        json.dump(
            {
                "dataset_name": batch.dataset_name,
                "cured_episodes": cured_count,
                "remedies_applied": remedies,
                "episodes": metadata_log,
            },
            f,
            indent=2,
        )

    print(
        f"\n{'━' * 56}\n"
        f"  calibra cure — {batch.dataset_name}\n"
        f"{'━' * 56}\n"
        f"  Episodes cured    : {cured_count}\n"
        f"  Output directory  : {out_dir.resolve()}\n"
        f"  Manifest written  : {out_dir.resolve()}/cure_manifest.json\n"
        f"{'━' * 56}\n"
    )
