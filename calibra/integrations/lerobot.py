"""
calibra.integrations.lerobot — filter LeRobot datasets with a CalibraReport.

This module is the bridge between Calibra's episode verdicts and LeRobot's
training pipeline. It closes the workflow:

    lerobot-record
        ↓
    calibra prune /path/to/dataset --keep 0.3 --report report.json
        ↓
    from calibra.integrations.lerobot import load_dataset
    ds = load_dataset("lerobot/pusht", report_path="report.json")
        ↓
    lerobot-train (with ds filtered to approved episodes)

Python API
----------
    from calibra.integrations.lerobot import (
        recommended_episode_ids,
        rejected_episode_ids,
        rejection_reason_codes,
        load_dataset,
        filter_by_report,
    )

    # 1. Get approved IDs from a CalibraReport JSON
    ids = recommended_episode_ids("results/lerobot/pusht/latest.json")
    # → ['0', '4', '7', '12', ...]

    # 2. Load and filter a HuggingFace LeRobot dataset in one call
    ds = load_dataset("lerobot/pusht", report_path="results/pusht/latest.json")
    # → datasets.Dataset filtered to approved episodes only

    # 3. Filter an already-loaded dataset
    from datasets import load_dataset as hf_load
    full = hf_load("lerobot/pusht", split="train")
    ds = filter_by_report(full, "results/pusht/latest.json")

    # 4. Inspect why episodes were rejected
    codes = rejection_reason_codes("results/pusht/latest.json")
    # → {"42": ["jerk_spike"], "7": ["diversity_pruned"], ...}

Generating a report with verdicts
----------------------------------
    calibra prune /path/to/dataset --keep 0.3 --report results/pusht/latest.json

The report must have been generated with ``calibra prune --report`` (not just
``calibra certify --report``) for ``episode_verdicts`` to be present.

LeRobot training integration
-----------------------------
To use filtered episodes directly with LeRobot's training:

    # Export the coreset to disk as a valid LeRobot v2 dataset:
    calibra prune /path/to/dataset --keep 0.3 --export-dataset ./pusht_coreset/
    # Then train on it:
    lerobot-train policy=act dataset_repo_id=./pusht_coreset/

    # Or, use the episode IDs as a filter within your own training loop:
    ids = recommended_episode_ids("report.json")
    # Pass ids to your dataset loader's episode_indices argument.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union


# ── reading verdicts from a CalibraReport ────────────────────────────────────


def recommended_episode_ids(report_path: Union[str, Path]) -> list[str]:
    """
    Return approved episode IDs from a CalibraReport JSON.

    Parameters
    ----------
    report_path : path to a CalibraReport JSON file produced by
                  ``calibra prune --report <path>``

    Returns
    -------
    List of episode ID strings to keep. Raises ValueError if the report
    has no episode_verdicts (use ``calibra prune --report`` not ``certify``).

    Examples
    --------
    >>> ids = recommended_episode_ids("results/lerobot/pusht/latest.json")
    >>> print(f"{len(ids)} episodes approved for training")
    """
    verdicts = _load_verdicts(report_path)
    return verdicts.get("keep_episode_ids", [])


def rejected_episode_ids(report_path: Union[str, Path]) -> list[str]:
    """Return episode IDs that were rejected by Calibra."""
    verdicts = _load_verdicts(report_path)
    return verdicts.get("reject_episode_ids", [])


def rejection_reason_codes(report_path: Union[str, Path]) -> dict[str, list[str]]:
    """
    Return per-episode rejection reason codes from a CalibraReport JSON.

    Returns a dict mapping episode_id → list of reason strings, e.g.:
      {"42": ["jerk_spike", "timestamp_dropout"], "7": ["diversity_pruned"]}

    Reason codes
    ------------
    Stage 1 (quality failures):
      short_episode, jerk_spike, velocity_discontinuity,
      timestamp_dropout, low_smoothness
    Stage 2 (diversity / selection):
      diversity_pruned, novelty_pruned, influence_pruned,
      energy_pruned, world_model_pruned
    """
    verdicts = _load_verdicts(report_path)
    return verdicts.get("reason_codes", {})


def episode_quality_scores(report_path: Union[str, Path]) -> dict[str, float]:
    """Return per-episode composite quality scores (lower = cleaner)."""
    verdicts = _load_verdicts(report_path)
    return verdicts.get("quality_scores", {})


def verdict_summary(report_path: Union[str, Path]) -> str:
    """Return a human-readable summary of the episode verdicts."""
    verdicts = _load_verdicts(report_path)
    n_orig = verdicts.get("n_original", 0)
    n_kept = verdicts.get("n_kept", 0)
    frac = verdicts.get("keep_fraction_actual", 0.0)
    method = verdicts.get("method", "unknown")
    lines = [
        "━" * 50,
        "  CALIBRA EPISODE VERDICTS",
        "━" * 50,
        f"  Kept     : {n_kept} / {n_orig}  ({frac:.1%})",
        f"  Rejected : {n_orig - n_kept}",
        f"  Method   : {method}",
        "━" * 50,
    ]
    return "\n".join(lines)


# ── HuggingFace dataset integration ──────────────────────────────────────────


def load_dataset(
    dataset_id_or_path: str,
    report_path: Union[str, Path],
    *,
    split: str = "train",
    episode_index_column: str = "episode_index",
    num_proc: int = 1,
):
    """
    Load a HuggingFace LeRobot dataset filtered to episodes approved by a CalibraReport.

    Parameters
    ----------
    dataset_id_or_path   : HuggingFace repo ID (e.g. "lerobot/pusht") or local path.
    report_path          : path to a CalibraReport JSON produced by
                           ``calibra prune --report <path>``.
    split                : dataset split to load (default: "train").
    episode_index_column : column name for episode index (default: "episode_index").
    num_proc             : number of processes for filtering (default: 1).

    Returns
    -------
    A ``datasets.Dataset`` filtered to approved episodes only.

    Raises
    ------
    ImportError : if the ``datasets`` package is not installed.
    ValueError  : if the report has no episode_verdicts.

    Examples
    --------
    >>> from calibra.integrations.lerobot import load_dataset
    >>> ds = load_dataset("lerobot/pusht", report_path="results/pusht/latest.json")
    >>> print(f"Training on {len(set(ds['episode_index']))} approved episodes")

    Training integration
    --------------------
    The filtered dataset can be saved to disk and used with LeRobot's trainer:

    >>> ds.save_to_disk("./pusht_coreset")
    # then: lerobot-train policy=act dataset_repo_id=./pusht_coreset
    """
    try:
        from datasets import load_dataset as _hf_load
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install it with: pip install datasets"
        ) from None

    keep_ids = recommended_episode_ids(report_path)
    keep_int = {int(eid) for eid in keep_ids}

    full_ds = _hf_load(dataset_id_or_path, split=split)
    filtered = full_ds.filter(
        lambda batch: [ep_idx in keep_int for ep_idx in batch[episode_index_column]],
        batched=True,
        num_proc=num_proc,
        desc=f"Filtering to {len(keep_int)} Calibra-approved episodes",
    )
    return filtered


def filter_by_report(
    dataset,
    report_path: Union[str, Path],
    *,
    episode_index_column: str = "episode_index",
    num_proc: int = 1,
):
    """
    Filter an already-loaded HuggingFace dataset to episodes approved by a CalibraReport.

    Parameters
    ----------
    dataset              : an already-loaded ``datasets.Dataset``.
    report_path          : path to a CalibraReport JSON.
    episode_index_column : column name for episode index (default: "episode_index").
    num_proc             : number of processes for filtering (default: 1).

    Returns
    -------
    Filtered ``datasets.Dataset``.

    Examples
    --------
    >>> from datasets import load_dataset
    >>> from calibra.integrations.lerobot import filter_by_report
    >>> full = load_dataset("lerobot/pusht", split="train")
    >>> filtered = filter_by_report(full, "results/pusht/latest.json")
    """
    keep_ids = recommended_episode_ids(report_path)
    keep_int = {int(eid) for eid in keep_ids}
    return dataset.filter(
        lambda batch: [ep_idx in keep_int for ep_idx in batch[episode_index_column]],
        batched=True,
        num_proc=num_proc,
        desc=f"Filtering to {len(keep_int)} Calibra-approved episodes",
    )


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
