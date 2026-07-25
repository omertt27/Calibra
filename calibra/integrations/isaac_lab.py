"""
calibra.integrations.isaac_lab — filter Isaac Lab HDF5 demos with a CalibraReport
                                  and export a GR00T training manifest.

This module closes the NVIDIA workflow:

    isaac sim → record demos (HDF5)
        ↓
    calibra prune /path/to/demos.hdf5 --keep 0.3 --policy gr00t --report report.json
        ↓
    from calibra.integrations.isaac_lab import export_gr00t_manifest
    manifest_path = export_gr00t_manifest("report.json", demos_path="demos.hdf5")
        ↓
    GR00T fine-tune with manifest_path

Python API
----------
    from calibra.integrations.isaac_lab import (
        recommended_demo_indices,
        rejected_demo_indices,
        rejection_reason_codes,
        export_gr00t_manifest,
        filter_hdf5,
    )

    # 1. Get approved demo indices (integer list matching HDF5 group order)
    indices = recommended_demo_indices("results/franka/latest.json")
    # → [0, 3, 7, 11, ...]

    # 2. Export a GR00T training manifest JSON
    manifest = export_gr00t_manifest(
        report_path="results/franka/latest.json",
        demos_path="demos/franka_pick.hdf5",
        out_path="gr00t_manifest.json",
    )

    # 3. Filter an HDF5 file to approved demos (writes a new HDF5)
    filter_hdf5(
        src="demos/franka_pick.hdf5",
        report_path="results/franka/latest.json",
        dst="demos/franka_pick_coreset.hdf5",
    )

GR00T manifest format
---------------------
The manifest is the standard Isaac Lab → GR00T contract:

    {
      "schema_version": "1.0.0",
      "calibra_report": "results/franka/latest.json",
      "dataset_path": "demos/franka_pick.hdf5",
      "n_demos_total": 200,
      "n_demos_selected": 60,
      "keep_fraction": 0.30,
      "method": "calibra-diversity",
      "demo_indices": [0, 3, 7, 11, ...],
      "demo_ids": ["demo_0", "demo_3", "demo_7", ...],
      "reason_codes": {"demo_42": ["jerk_spike"], ...},
      "created_at": "2026-07-23T12:00:00+00:00"
    }

Generating a report with verdicts
----------------------------------
    calibra prune demos.hdf5 --keep 0.3 --policy gr00t --report results/franka/latest.json

The ``episode_verdicts`` field (populated by ``calibra prune --report``) must be
present in the report. Reports from ``calibra certify --report`` do not contain
per-episode verdicts and will raise ``ValueError``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

# ── reading verdicts ──────────────────────────────────────────────────────────


def recommended_demo_indices(report_path: Union[str, Path]) -> list[int]:
    """
    Return approved demo indices (0-based integers) from a CalibraReport JSON.

    Isaac Lab HDF5 files index demos by integer (``data/demo_0``, ``data/demo_1``,
    …). This function converts the string episode IDs in the report to integers
    so they can be used to select HDF5 groups directly.

    Parameters
    ----------
    report_path : path to a CalibraReport JSON produced by
                  ``calibra prune --report <path>``

    Returns
    -------
    Sorted list of integer demo indices to keep.
    """
    verdicts = _load_verdicts(report_path)
    return sorted(int(eid) for eid in verdicts.get("keep_episode_ids", []))


def rejected_demo_indices(report_path: Union[str, Path]) -> list[int]:
    """Return demo indices that were rejected by Calibra."""
    verdicts = _load_verdicts(report_path)
    return sorted(int(eid) for eid in verdicts.get("reject_episode_ids", []))


def rejection_reason_codes(report_path: Union[str, Path]) -> dict[str, list[str]]:
    """
    Return per-demo rejection reason codes.

    Returns ``{demo_id: [reason, ...]}`` where demo_id is the string episode ID
    from the report (e.g. ``"42"`` for ``demo_42`` in the HDF5).

    Reason codes
    ------------
    Stage 1 (quality):
      short_episode, jerk_spike, velocity_discontinuity,
      timestamp_dropout, low_smoothness
    Stage 2 (GR00T diversity):
      diversity_pruned, novelty_pruned, influence_pruned,
      energy_pruned, world_model_pruned
    """
    verdicts = _load_verdicts(report_path)
    return verdicts.get("reason_codes", {})


def verdict_summary(report_path: Union[str, Path]) -> str:
    """Return a human-readable summary of the episode verdicts."""
    verdicts = _load_verdicts(report_path)
    n_orig = verdicts.get("n_original", 0)
    n_kept = verdicts.get("n_kept", 0)
    frac = verdicts.get("keep_fraction_actual", 0.0)
    method = verdicts.get("method", "unknown")
    lines = [
        "━" * 55,
        "  CALIBRA → ISAAC LAB / GR00T VERDICTS",
        "━" * 55,
        f"  Approved : {n_kept} / {n_orig} demos  ({frac:.1%})",
        f"  Rejected : {n_orig - n_kept}",
        f"  Method   : {method}",
        "━" * 55,
    ]
    return "\n".join(lines)


# ── GR00T manifest export ──────────────────────────────────────────────────────


def export_gr00t_manifest(
    report_path: Union[str, Path],
    demos_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Export a GR00T training manifest JSON from a CalibraReport.

    The manifest is the standard contract understood by Isaac Lab's GR00T
    fine-tuning scripts. It lists approved demo indices, rejection reasons,
    and provenance metadata so training runs are fully reproducible.

    Parameters
    ----------
    report_path : path to a CalibraReport JSON produced by
                  ``calibra prune --report <path>``.
    demos_path  : path to the source HDF5 demo file (optional — stored in
                  manifest for provenance but not read by this function).
    out_path    : path to write the manifest JSON. Defaults to
                  ``<report_dir>/gr00t_manifest.json``.

    Returns
    -------
    Path to the written manifest JSON.

    Examples
    --------
    >>> from calibra.integrations.isaac_lab import export_gr00t_manifest
    >>> manifest_path = export_gr00t_manifest(
    ...     "results/franka/latest.json",
    ...     demos_path="demos/franka_pick.hdf5",
    ... )
    >>> print(f"Manifest: {manifest_path}")

    To fine-tune GR00T with the manifest:

        python -m gr00t.train \\
            --manifest gr00t_manifest.json \\
            --demo-file demos/franka_pick.hdf5
    """
    report_path = Path(report_path)
    verdicts = _load_verdicts(report_path)

    keep_ids_str = verdicts.get("keep_episode_ids", [])
    keep_indices = sorted(int(eid) for eid in keep_ids_str)
    demo_ids = [f"demo_{i}" for i in keep_indices]

    if out_path is None:
        out_path = report_path.parent / "gr00t_manifest.json"
    out_path = Path(out_path)

    manifest = {
        "schema_version": "1.0.0",
        "calibra_report": str(report_path.resolve()),
        "dataset_path": str(Path(demos_path).resolve()) if demos_path else None,
        "n_demos_total": verdicts.get("n_original", len(keep_indices)),
        "n_demos_selected": len(keep_indices),
        "keep_fraction": verdicts.get("keep_fraction_actual", 1.0),
        "method": verdicts.get("method", "calibra"),
        "demo_indices": keep_indices,
        "demo_ids": demo_ids,
        "reason_codes": verdicts.get("reason_codes", {}),
        "quality_scores": verdicts.get("quality_scores", {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


# ── HDF5 coreset filter ───────────────────────────────────────────────────────


def filter_hdf5(
    src: Union[str, Path],
    report_path: Union[str, Path],
    dst: Union[str, Path],
) -> Path:
    """
    Write a new HDF5 file containing only Calibra-approved demos.

    Copies ``data/demo_N`` groups for all approved indices from *src* to *dst*.
    All top-level attributes (``data.attrs``, ``mask``, etc.) from the source
    are preserved.

    Parameters
    ----------
    src         : path to the source Isaac Lab HDF5 demo file.
    report_path : path to a CalibraReport JSON produced by
                  ``calibra prune --report <path>``.
    dst         : destination path for the filtered HDF5 file.

    Returns
    -------
    Path to the written HDF5 file.

    Raises
    ------
    ImportError : if ``h5py`` is not installed.

    Examples
    --------
    >>> from calibra.integrations.isaac_lab import filter_hdf5
    >>> filtered = filter_hdf5(
    ...     src="demos/franka_pick.hdf5",
    ...     report_path="results/franka/latest.json",
    ...     dst="demos/franka_pick_coreset.hdf5",
    ... )
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "The 'h5py' package is required for HDF5 filtering.\nInstall it with: pip install h5py"
        ) from None

    keep_indices = recommended_demo_indices(report_path)
    keep_set = set(keep_indices)

    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as f_src, h5py.File(dst, "w") as f_dst:
        # Copy top-level attrs
        for key, val in f_src.attrs.items():
            f_dst.attrs[key] = val

        # Copy approved demo groups into a new contiguous numbering
        new_idx = 0
        for idx in sorted(keep_set):
            old_key = f"data/demo_{idx}"
            if old_key not in f_src:
                continue
            new_key = f"data/demo_{new_idx}"
            f_src.copy(old_key, f_dst, name=new_key)
            new_idx += 1

        # Copy other top-level groups (mask, env, etc.) except data
        for group_name in f_src:
            if group_name != "data" and group_name not in f_dst:
                f_src.copy(group_name, f_dst)

    return dst


# ── internal helpers ──────────────────────────────────────────────────────────


def _load_verdicts(report_path: Union[str, Path]) -> dict:
    """Load and validate episode_verdicts from a CalibraReport JSON."""
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"CalibraReport not found: {report_path}")
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    verdicts = data.get("episode_verdicts")
    if verdicts is None:
        raise ValueError(
            f"No episode_verdicts in {report_path}.\n"
            "Episode verdicts are produced by 'calibra prune --report <path>'.\n"
            "The 'calibra certify --report' command produces a quality-only report "
            "without per-episode selection verdicts."
        )
    return verdicts
