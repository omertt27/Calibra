"""
calibra transfer — cross-embodiment compatibility scoring.

When you want to reuse demonstrations from Robot A to train a policy for
Robot B (cross-embodiment transfer), this command scores how compatible the
two datasets are across action space, morphology, and trajectory structure.

Use cases
----------
  • Mixing ALOHA bimanual data into a single-arm training run.
  • Reusing Open X-Embodiment data for a new robot.
  • Checking whether SO-100 arm data transfers to a Franka Panda.

Scoring dimensions
-------------------
  Action-space alignment     — do the action distributions overlap?
  Frequency compatibility    — are the control rates similar enough?
  Episode structure parity   — similar episode lengths and task structure?
  Trajectory style match     — similar smoothness profiles?

Transfer risk levels
---------------------
  DIRECT     — high compatibility; direct mixing is viable
  ADAPT      — moderate compatibility; normalisation / retargeting recommended
  DIFFICULT  — significant mismatch; targeted domain adaptation required
  INCOMPATIBLE — incompatible morphology or action space

Usage
------
    calibra transfer /data/source_robot.h5 /data/target_robot.h5
    calibra transfer lerobot/aloha_mobile_cabinet lerobot/so100_pickplace
    calibra transfer /data/source.h5 /data/target.h5 --json

Exit codes
----------
    0  DIRECT or ADAPT
    1  DIFFICULT
    2  INCOMPATIBLE
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN  = "─" * _WIDTH


def _raw(report: DiagnosticReport, analyzer: str) -> dict:
    for r in report.analyzer_results:
        if r.analyzer_name == analyzer:
            return r.raw_metrics
    return {}


def _risk_icon(level: str) -> str:
    return {
        "DIRECT":        "✅",
        "ADAPT":         "🟡",
        "DIFFICULT":     "🟠",
        "INCOMPATIBLE":  "🔴",
    }.get(level, "❓")


# ── scoring helpers ───────────────────────────────────────────────────────────

def _action_dim_compatibility(src_dim: int, tgt_dim: int) -> tuple[str, str]:
    """
    Returns (level, note) for action dimensionality compatibility.
    We support many-to-fewer retargeting and exact matches; more complex
    cases are flagged as DIFFICULT.
    """
    if src_dim == tgt_dim:
        return "DIRECT", f"Action dims match ({src_dim}D)."
    if src_dim > tgt_dim:
        return "ADAPT", (
            f"Source has {src_dim}D actions, target has {tgt_dim}D. "
            "Subset retargeting (drop extra dims) may work — "
            "use `calibra retarget` to convert."
        )
    return "DIFFICULT", (
        f"Source has fewer action dims ({src_dim}D) than target ({tgt_dim}D). "
        "You cannot synthesise missing degrees of freedom from source data."
    )


def _frequency_compatibility(src_hz: Optional[float], tgt_hz: Optional[float]) -> tuple[str, str]:
    if src_hz is None or tgt_hz is None:
        return "ADAPT", "Could not determine control frequency for one or both datasets."
    ratio = max(src_hz, tgt_hz) / max(min(src_hz, tgt_hz), 0.1)
    if ratio < 1.5:
        return "DIRECT", f"Control frequencies are similar ({src_hz:.0f} Hz vs {tgt_hz:.0f} Hz)."
    if ratio < 3.0:
        return "ADAPT", (
            f"Frequency mismatch ({src_hz:.0f} Hz vs {tgt_hz:.0f} Hz). "
            "Sub-sampling or interpolation may be needed."
        )
    return "DIFFICULT", (
        f"Large frequency mismatch ({src_hz:.0f} Hz vs {tgt_hz:.0f} Hz, ratio {ratio:.1f}×). "
        "Temporal dynamics will not align without significant resampling."
    )


def _smoothness_compatibility(src_ldlj: Optional[float], tgt_ldlj: Optional[float]) -> tuple[str, str]:
    if src_ldlj is None or tgt_ldlj is None:
        return "ADAPT", "LDLJ not available for one or both datasets."
    delta = abs(src_ldlj - tgt_ldlj)
    if delta < 3.0:
        return "DIRECT", f"Similar smoothness profiles (ΔLDLJ = {delta:.2f})."
    if delta < 8.0:
        return "ADAPT", (
            f"Moderate smoothness gap (ΔLDLJ = {delta:.2f}). "
            "Source and target have different trajectory styles."
        )
    return "DIFFICULT", (
        f"Large smoothness gap (ΔLDLJ = {delta:.2f}). "
        "Very different trajectory styles may confuse the policy."
    )


def _episode_length_compatibility(
    src_mean: Optional[float], tgt_mean: Optional[float]
) -> tuple[str, str]:
    if src_mean is None or tgt_mean is None:
        return "ADAPT", "Episode length statistics not available."
    ratio = max(src_mean, tgt_mean) / max(min(src_mean, tgt_mean), 1.0)
    if ratio < 2.0:
        return "DIRECT", f"Similar episode lengths ({src_mean:.0f} vs {tgt_mean:.0f} steps)."
    if ratio < 5.0:
        return "ADAPT", (
            f"Episode length mismatch ({src_mean:.0f} vs {tgt_mean:.0f} steps). "
            "Source episodes are {:.0f}× {}.".format(
                ratio, "longer" if src_mean > tgt_mean else "shorter"
            )
        )
    return "DIFFICULT", (
        f"Very different episode lengths ({src_mean:.0f} vs {tgt_mean:.0f} steps). "
        "The temporal structure is incompatible."
    )


def _action_range_overlap(src_batch, tgt_batch) -> tuple[str, str]:
    """Estimate how much of the target action range is covered by the source."""
    try:
        src_actions = np.concatenate(
            [ep.actions for ep in src_batch.episodes if ep.actions.ndim > 1], axis=0
        )
        tgt_actions = np.concatenate(
            [ep.actions for ep in tgt_batch.episodes if ep.actions.ndim > 1], axis=0
        )
        dim = min(src_actions.shape[1], tgt_actions.shape[1])
        src_actions = src_actions[:, :dim]
        tgt_actions = tgt_actions[:, :dim]

        overlaps: list[float] = []
        for d in range(dim):
            tgt_lo, tgt_hi = tgt_actions[:, d].min(), tgt_actions[:, d].max()
            src_lo, src_hi = src_actions[:, d].min(), src_actions[:, d].max()
            if tgt_hi <= tgt_lo:
                continue
            inter_lo = max(tgt_lo, src_lo)
            inter_hi = min(tgt_hi, src_hi)
            overlap = max(0.0, inter_hi - inter_lo) / (tgt_hi - tgt_lo)
            overlaps.append(overlap)

        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        if mean_overlap >= 0.8:
            return "DIRECT", f"Source covers {mean_overlap:.0%} of target action range."
        if mean_overlap >= 0.5:
            return "ADAPT", (
                f"Source covers {mean_overlap:.0%} of target action range. "
                "Some target actions have no source demonstrations."
            )
        return "DIFFICULT", (
            f"Source only covers {mean_overlap:.0%} of target action range. "
            "Large parts of the target action space are undemonstrated."
        )
    except Exception:
        return "ADAPT", "Could not compute action range overlap."


# ── aggregate level ───────────────────────────────────────────────────────────

_LEVEL_ORDER = {"DIRECT": 0, "ADAPT": 1, "DIFFICULT": 2, "INCOMPATIBLE": 3}


def _worst(levels: list[str]) -> str:
    return max(levels, key=lambda l: _LEVEL_ORDER.get(l, 0)) if levels else "DIRECT"


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_transfer(
    src_report: DiagnosticReport,
    tgt_report: DiagnosticReport,
    src_batch=None,
    tgt_batch=None,
) -> dict:
    """
    Score the compatibility of transferring data from src_report to tgt_report.

    Returns a structured dict with per-dimension scores and an overall level.
    """
    dimensions: dict = {}
    levels: list[str] = []

    # ── action dimensionality ─────────────────────────────────────────────────
    src_dim = src_report.analyzer_results[0].raw_metrics.get("action_dim") if \
        src_report.analyzer_results else None
    # Try to get from coverage analyzer
    for r in src_report.analyzer_results:
        d = r.raw_metrics.get("action_entropy", {}).get("action_dim")
        if d is not None:
            src_dim = int(d)
            break
    for r in tgt_report.analyzer_results:
        d = r.raw_metrics.get("action_entropy", {}).get("action_dim")
        if d is not None:
            tgt_dim = int(d)
            break
    else:
        tgt_dim = None

    # Fallback: read from batch if available
    if src_dim is None and src_batch and src_batch.episodes:
        src_dim = src_batch.episodes[0].action_dim
    if tgt_dim is None and tgt_batch and tgt_batch.episodes:
        tgt_dim = tgt_batch.episodes[0].action_dim

    if src_dim is not None and tgt_dim is not None:
        level, note = _action_dim_compatibility(src_dim, tgt_dim)
        dimensions["action_dim_compatibility"] = {
            "source_dim": src_dim, "target_dim": tgt_dim,
            "level": level, "note": note,
        }
        if level == "INCOMPATIBLE":
            return {
                "overall_level": "INCOMPATIBLE",
                "source_dataset": src_report.dataset_name,
                "target_dataset": tgt_report.dataset_name,
                "source_episodes": src_report.n_episodes,
                "target_episodes": tgt_report.n_episodes,
                "dimensions": dimensions,
                "note": "Action dimension incompatibility prevents direct transfer.",
            }
        levels.append(level)

    # ── control frequency ─────────────────────────────────────────────────────
    def _get_hz(report: DiagnosticReport) -> Optional[float]:
        t = _raw(report, "temporal_stability")
        jitter = t.get("jitter", {})
        return jitter.get("mean_hz") or t.get("mean_control_hz")

    src_hz = _get_hz(src_report)
    tgt_hz = _get_hz(tgt_report)
    level, note = _frequency_compatibility(src_hz, tgt_hz)
    dimensions["frequency_compatibility"] = {
        "source_hz": src_hz, "target_hz": tgt_hz,
        "level": level, "note": note,
    }
    levels.append(level)

    # ── smoothness match ──────────────────────────────────────────────────────
    src_ldlj = _raw(src_report, "control_smoothness").get("ldlj", {}).get("mean_ldlj")
    tgt_ldlj = _raw(tgt_report, "control_smoothness").get("ldlj", {}).get("mean_ldlj")
    level, note = _smoothness_compatibility(src_ldlj, tgt_ldlj)
    dimensions["smoothness_match"] = {
        "source_ldlj": src_ldlj, "target_ldlj": tgt_ldlj,
        "level": level, "note": note,
    }
    levels.append(level)

    # ── episode length parity ─────────────────────────────────────────────────
    src_ep_len = _raw(src_report, "coverage_entropy").get("episode_lengths", {}).get("mean_steps")
    tgt_ep_len = _raw(tgt_report, "coverage_entropy").get("episode_lengths", {}).get("mean_steps")
    level, note = _episode_length_compatibility(src_ep_len, tgt_ep_len)
    dimensions["episode_length_parity"] = {
        "source_mean_steps": src_ep_len,
        "target_mean_steps": tgt_ep_len,
        "level": level, "note": note,
    }
    levels.append(level)

    # ── action range overlap (requires batches) ───────────────────────────────
    if src_batch is not None and tgt_batch is not None:
        level, note = _action_range_overlap(src_batch, tgt_batch)
        dimensions["action_range_overlap"] = {"level": level, "note": note}
        levels.append(level)

    overall = _worst(levels)

    return {
        "overall_level":   overall,
        "source_dataset":  src_report.dataset_name,
        "target_dataset":  tgt_report.dataset_name,
        "source_episodes": src_report.n_episodes,
        "target_episodes": tgt_report.n_episodes,
        "dimensions":      dimensions,
    }


# ── rendering ─────────────────────────────────────────────────────────────────

def render_transfer(result: dict) -> str:
    overall = result["overall_level"]
    icon    = _risk_icon(overall)
    lines = [
        _THICK,
        "  CALIBRA CROSS-EMBODIMENT TRANSFER SCORE",
        _THICK,
        "",
        f"  Source : {result['source_dataset']}  ({result['source_episodes']} eps)",
        f"  Target : {result['target_dataset']}  ({result['target_episodes']} eps)",
        "",
        _THIN,
        f"  {icon}  Transfer Compatibility: {overall}",
        _THIN,
        "",
    ]

    _labels = {
        "action_dim_compatibility":  "Action Dimensionality",
        "frequency_compatibility":   "Control Frequency",
        "smoothness_match":          "Trajectory Smoothness",
        "episode_length_parity":     "Episode Length",
        "action_range_overlap":      "Action Range Overlap",
    }

    for dim_key, dim_data in result["dimensions"].items():
        label      = _labels.get(dim_key, dim_key.replace("_", " ").title())
        dim_level  = dim_data.get("level", "?")
        dim_icon   = _risk_icon(dim_level)
        lines.append(f"  {dim_icon} {label:<35} [{dim_level}]")
        lines.append(f"     {dim_data.get('note', '')}")
        lines.append("")

    lines.append(_THIN)
    lines.append("  RECOMMENDATIONS")
    lines.append(_THIN)
    if overall == "DIRECT":
        lines.append(
            "  ✓ Source data is highly compatible. Direct mixing is recommended."
        )
        lines.append(
            "  Use `calibra prune` on the source to select the most diverse episodes."
        )
    elif overall == "ADAPT":
        lines += [
            "  • Normalise action spaces before mixing source and target data.",
            "  • Use `calibra retarget` if action dims differ.",
            "  • Sub-sample or interpolate if control frequencies mismatch.",
            "  • Consider weighting source data lower (e.g. 0.3×) than target data.",
        ]
    elif overall == "DIFFICULT":
        lines += [
            "  • Collect at least 100 target-robot demonstrations before mixing.",
            "  • Apply domain adaptation (style transfer on trajectories).",
            "  • Investigate embodiment-aware training (per-robot action heads).",
        ]
    else:
        lines += [
            "  ✗ Direct transfer is not viable without significant engineering.",
            "  • Consider using the source data for pre-training only.",
            "  • Train on target data exclusively for fine-tuning.",
        ]

    lines.append(_THICK)
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_transfer(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra transfer",
        description=(
            "Score cross-embodiment compatibility between a source (donor) "
            "and target robot dataset."
        ),
    )
    p.add_argument("source_path", help="Path or HF Hub ID of the source (donor) dataset")
    p.add_argument("target_path", help="Path or HF Hub ID of the target robot dataset")
    p.add_argument(
        "--source-format",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
    )
    p.add_argument(
        "--target-format",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
    )
    p.add_argument("--policy", "-p", metavar="FAMILY")
    p.add_argument("--json", "-j", action="store_true")
    args = p.parse_args(argv)

    def strip_hf(path: str) -> str:
        return path[len("hf://"):] if path.startswith("hf://") else path

    src_path = strip_hf(args.source_path)
    tgt_path = strip_hf(args.target_path)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    from calibra.__main__ import _get_reader
    from calibra.ingestion.registry import load as _load

    log(f"Loading source dataset: {src_path!r} ...")
    try:
        src_reader = _get_reader(args.source_format) if args.source_format else None
        src_batch  = _load(src_path, reader=src_reader)
    except Exception as exc:
        print(f"error loading source dataset: {exc}", file=sys.stderr)
        sys.exit(2)

    log(f"Loading target dataset: {tgt_path!r} ...")
    try:
        tgt_reader = _get_reader(args.target_format) if args.target_format else None
        tgt_batch  = _load(tgt_path, reader=tgt_reader)
    except Exception as exc:
        print(f"error loading target dataset: {exc}", file=sys.stderr)
        sys.exit(2)

    log("Running diagnostic pipeline ...")
    pipeline   = Pipeline()
    src_report = pipeline.run(src_batch, policy_family=args.policy)
    tgt_report = pipeline.run(tgt_batch, policy_family=args.policy)

    result = analyze_transfer(src_report, tgt_report, src_batch, tgt_batch)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_transfer(result))

    exit_code = _LEVEL_ORDER.get(result["overall_level"], 0)
    sys.exit(min(exit_code, 2))


_LEVEL_ORDER = {"DIRECT": 0, "ADAPT": 1, "DIFFICULT": 2, "INCOMPATIBLE": 3}
