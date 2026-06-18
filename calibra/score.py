"""
calibra score — compute a composite Calibra Score for a robot dataset.

The Calibra Score is a single 0–100 number that summarises dataset quality
across all diagnostic dimensions. It is designed to be:

  • Human-readable — 0 = unusable, 100 = pristine.
  • Comparable     — the same formula applies to every dataset and format.
  • Actionable     — each dimension's sub-score explains the deduction.
  • Badge-friendly — the JSON output can be embedded in dataset cards.

Scoring dimensions and weights
-------------------------------
  Temporal stability     (25 pts)  — jitter CV, dropout rate
  Control smoothness     (35 pts)  — LDLJ, jerk spike rate, velocity discontinuity
  Coverage / diversity   (25 pts)  — action entropy, episode length CV
  Task structure         (15 pts)  — trajectory diversity, short-episode fraction

Score categories
-----------------
  90–100   Excellent  — ready for training
  75–89    Good       — minor issues; training likely fine
  60–74    Fair       — notable issues; consider pruning or re-collection
  40–59    Poor       — significant data quality problems
  0–39     Critical   — major failures; training on this data is risky

Usage
------
    calibra score /data/robot_demos.h5
    calibra score lerobot/my_dataset --format lerobot
    calibra score /data/my_ds --reference aloha --json
    calibra score hf://lerobot/pusht_image --badge          # print markdown badge

Exit codes
----------
    0  Score ≥ 75 (Good or better)
    1  Score 40–74 (Fair or Poor)
    2  Score < 40 (Critical)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport, RiskLevel

# ── scoring constants ─────────────────────────────────────────────────────────

_MAX_TEMPORAL   = 25.0
_MAX_SMOOTHNESS = 35.0
_MAX_COVERAGE   = 25.0
_MAX_STRUCTURE  = 15.0
_MAX_TOTAL      = _MAX_TEMPORAL + _MAX_SMOOTHNESS + _MAX_COVERAGE + _MAX_STRUCTURE

# Metric thresholds used to deduct points
_JITTER_CV_WARNING   = 0.05
_JITTER_CV_CRITICAL  = 0.20
_DROPOUT_WARNING     = 0.01
_DROPOUT_CRITICAL    = 0.05

_LDLJ_GOOD     = -7.0
_LDLJ_WARNING  = -10.0
_LDLJ_CRITICAL = -15.0

_SPIKE_WARNING  = 0.02
_SPIKE_CRITICAL = 0.05

_VEL_DISC_WARNING  = 0.02
_VEL_DISC_CRITICAL = 0.05

_ENTROPY_GOOD     = 4.0   # bits/dim
_ENTROPY_WARNING  = 2.5
_ENTROPY_CRITICAL = 1.5

_TRAJ_DIV_GOOD    = 0.4   # trajectory diversity score
_SHORT_EP_WARNING = 0.05


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _raw(report: DiagnosticReport, analyzer: str) -> dict:
    for r in report.analyzer_results:
        if r.analyzer_name == analyzer:
            return r.raw_metrics
    return {}


# ── dimension scorers ─────────────────────────────────────────────────────────

def _score_temporal(report: DiagnosticReport) -> tuple[float, dict]:
    """Returns (score 0–25, details dict)."""
    t = _raw(report, "temporal_stability")
    jitter  = t.get("jitter",  {}).get("mean_cv",             None)
    dropout = t.get("dropout", {}).get("mean_dropout_fraction", None)

    deduction = 0.0
    details: dict = {}

    if jitter is not None:
        details["jitter_cv"] = jitter
        if jitter >= _JITTER_CV_CRITICAL:
            deduction += 12.5
        elif jitter >= _JITTER_CV_WARNING:
            frac = (jitter - _JITTER_CV_WARNING) / (_JITTER_CV_CRITICAL - _JITTER_CV_WARNING)
            deduction += 12.5 * _clamp(frac)

    if dropout is not None:
        details["dropout_rate"] = dropout
        if dropout >= _DROPOUT_CRITICAL:
            deduction += 12.5
        elif dropout >= _DROPOUT_WARNING:
            frac = (dropout - _DROPOUT_WARNING) / (_DROPOUT_CRITICAL - _DROPOUT_WARNING)
            deduction += 12.5 * _clamp(frac)

    score = max(0.0, _MAX_TEMPORAL - deduction)
    return score, details


def _score_smoothness(report: DiagnosticReport) -> tuple[float, dict]:
    """Returns (score 0–35, details dict)."""
    s = _raw(report, "control_smoothness")
    ldlj      = s.get("ldlj",             {}).get("mean_ldlj",         None)
    spike     = s.get("jerk_spikes",      {}).get("mean_spike_fraction", None)
    vel_disc  = s.get("vel_discontinuities", {}).get("mean_disc_fraction", None)

    deduction = 0.0
    details: dict = {}

    # LDLJ (up to 15 pts)
    if ldlj is not None:
        details["ldlj"] = ldlj
        if ldlj <= _LDLJ_CRITICAL:
            deduction += 15.0
        elif ldlj <= _LDLJ_WARNING:
            frac = (ldlj - _LDLJ_WARNING) / (_LDLJ_CRITICAL - _LDLJ_WARNING)
            deduction += 15.0 * _clamp(frac)
        elif ldlj <= _LDLJ_GOOD:
            frac = (ldlj - _LDLJ_GOOD) / (_LDLJ_WARNING - _LDLJ_GOOD)
            deduction += 7.5 * _clamp(frac)

    # Spike rate (up to 10 pts)
    if spike is not None:
        details["spike_rate"] = spike
        if spike >= _SPIKE_CRITICAL:
            deduction += 10.0
        elif spike >= _SPIKE_WARNING:
            frac = (spike - _SPIKE_WARNING) / (_SPIKE_CRITICAL - _SPIKE_WARNING)
            deduction += 10.0 * _clamp(frac)

    # Velocity discontinuity (up to 10 pts)
    if vel_disc is not None:
        details["vel_disc_rate"] = vel_disc
        if vel_disc >= _VEL_DISC_CRITICAL:
            deduction += 10.0
        elif vel_disc >= _VEL_DISC_WARNING:
            frac = (vel_disc - _VEL_DISC_WARNING) / (_VEL_DISC_CRITICAL - _VEL_DISC_WARNING)
            deduction += 10.0 * _clamp(frac)

    score = max(0.0, _MAX_SMOOTHNESS - deduction)
    return score, details


def _score_coverage(report: DiagnosticReport) -> tuple[float, dict]:
    """Returns (score 0–25, details dict)."""
    c = _raw(report, "coverage_entropy")
    entropy = c.get("action_entropy", {}).get("entropy_bits_per_dim", None)
    ep_cv   = c.get("episode_lengths", {}).get("cv",                 None)

    deduction = 0.0
    details: dict = {}

    # Action entropy (up to 20 pts)
    if entropy is not None:
        details["action_entropy_bits_per_dim"] = entropy
        if entropy <= _ENTROPY_CRITICAL:
            deduction += 20.0
        elif entropy <= _ENTROPY_WARNING:
            frac = (_ENTROPY_WARNING - entropy) / (_ENTROPY_WARNING - _ENTROPY_CRITICAL)
            deduction += 20.0 * _clamp(frac)
        elif entropy <= _ENTROPY_GOOD:
            frac = (_ENTROPY_GOOD - entropy) / (_ENTROPY_GOOD - _ENTROPY_WARNING)
            deduction += 10.0 * _clamp(frac)

    # Episode length consistency (up to 5 pts)
    if ep_cv is not None:
        details["episode_length_cv"] = ep_cv
        if ep_cv > 0.5:
            deduction += 5.0
        elif ep_cv > 0.2:
            frac = (ep_cv - 0.2) / 0.3
            deduction += 5.0 * _clamp(frac)

    score = max(0.0, _MAX_COVERAGE - deduction)
    return score, details


def _score_structure(report: DiagnosticReport) -> tuple[float, dict]:
    """Returns (score 0–15, details dict)."""
    ts = _raw(report, "task_structure")
    traj_div   = ts.get("trajectory_diversity", {}).get("separation_score",        None)
    short_frac = ts.get("short_episodes",       {}).get("short_episode_fraction",  None)

    deduction = 0.0
    details: dict = {}

    # Trajectory diversity (up to 10 pts)
    if traj_div is not None:
        details["trajectory_diversity"] = traj_div
        if traj_div < _TRAJ_DIV_GOOD:
            frac = _clamp(1.0 - traj_div / _TRAJ_DIV_GOOD)
            deduction += 10.0 * frac

    # Short episodes (up to 5 pts)
    if short_frac is not None:
        details["short_episode_fraction"] = short_frac
        if short_frac > _SHORT_EP_WARNING:
            deduction += min(5.0, 5.0 * short_frac / 0.20)

    score = max(0.0, _MAX_STRUCTURE - deduction)
    return score, details


# ── category labelling ────────────────────────────────────────────────────────

def _category(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    if score >= 40:
        return "Poor"
    return "Critical"


def _exit_code(score: float) -> int:
    if score >= 75:
        return 0
    if score >= 40:
        return 1
    return 2


# ── public API ────────────────────────────────────────────────────────────────

def compute_score(report: DiagnosticReport) -> dict:
    """
    Compute the composite Calibra Score from a DiagnosticReport.

    Returns a dict with:
        total_score      : float 0–100
        category         : str label
        dimensions       : per-dimension scores and details
        n_critical_flags : int
        n_warning_flags  : int
    """
    t_score,  t_details  = _score_temporal(report)
    s_score,  s_details  = _score_smoothness(report)
    c_score,  c_details  = _score_coverage(report)
    st_score, st_details = _score_structure(report)

    total = (t_score + s_score + c_score + st_score) / _MAX_TOTAL * 100.0
    total = round(total, 1)

    return {
        "total_score": total,
        "category":    _category(total),
        "dimensions": {
            "temporal_stability": {
                "score": round(t_score, 2),
                "max":   _MAX_TEMPORAL,
                "details": t_details,
            },
            "control_smoothness": {
                "score": round(s_score, 2),
                "max":   _MAX_SMOOTHNESS,
                "details": s_details,
            },
            "coverage_diversity": {
                "score": round(c_score, 2),
                "max":   _MAX_COVERAGE,
                "details": c_details,
            },
            "task_structure": {
                "score": round(st_score, 2),
                "max":   _MAX_STRUCTURE,
                "details": st_details,
            },
        },
        "n_critical_flags": len(report.flags_at_level(RiskLevel.CRITICAL)),
        "n_warning_flags":  len(report.flags_at_level(RiskLevel.WARNING)),
        "n_episodes":       report.n_episodes,
        "n_samples":        report.n_samples,
        "dataset_name":     report.dataset_name,
        "source_path":      report.source_path,
    }


# ── rendering ─────────────────────────────────────────────────────────────────

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN  = "─" * _WIDTH

_CATEGORY_ICON = {
    "Excellent": "✅",
    "Good":      "🟢",
    "Fair":      "🟡",
    "Poor":      "🟠",
    "Critical":  "🔴",
}

_DIM_LABELS = {
    "temporal_stability": "Temporal Stability",
    "control_smoothness": "Control Smoothness",
    "coverage_diversity": "Coverage / Diversity",
    "task_structure":     "Task Structure",
}


def render_score(result: dict) -> str:
    score    = result["total_score"]
    category = result["category"]
    icon     = _CATEGORY_ICON[category]

    lines = [
        _THICK,
        "  CALIBRA SCORE",
        _THICK,
        "",
        f"  Dataset  : {result['dataset_name']}",
        f"  Episodes : {result['n_episodes']}  ·  Steps: {result['n_samples']}",
        "",
        _THIN,
        f"  {icon}  {score:.1f} / 100  —  {category}",
        _THIN,
        "",
    ]

    for dim_key, dim_data in result["dimensions"].items():
        label   = _DIM_LABELS.get(dim_key, dim_key)
        s       = dim_data["score"]
        mx      = dim_data["max"]
        pct     = s / mx * 100 if mx else 0
        bar_len = int(pct / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {label:<22} {s:5.1f}/{mx:.0f}  [{bar}] {pct:4.0f}%")
        for k, v in dim_data["details"].items():
            lines.append(f"     {k}: {v:.4g}")
        lines.append("")

    n_crit = result["n_critical_flags"]
    n_warn = result["n_warning_flags"]
    lines.append(f"  {n_crit} critical flags  ·  {n_warn} warnings")
    lines.append(_THICK)
    return "\n".join(lines)


def render_badge(result: dict) -> str:
    """Render a markdown snippet suitable for a README or HuggingFace dataset card."""
    score    = result["total_score"]
    category = result["category"]
    color_map = {
        "Excellent": "brightgreen",
        "Good":      "green",
        "Fair":      "yellow",
        "Poor":      "orange",
        "Critical":  "red",
    }
    color = color_map[category]
    label = f"Calibra%20Score-{score:.0f}%2F100-{color}"
    badge = f"![Calibra Score](https://img.shields.io/badge/{label})"
    return badge


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_score(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra score",
        description="Compute a composite Calibra Score (0–100) for a robot dataset.",
    )
    p.add_argument("path", help="Path or HF Hub ID of the dataset to score")
    p.add_argument(
        "--format", "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--policy", "-p",
        metavar="FAMILY",
        help="Policy family for conditioned hints (e.g. 'diffusion', 'pi0', 'octo')",
    )
    p.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output full score breakdown as JSON",
    )
    p.add_argument(
        "--badge",
        action="store_true",
        help="Print a markdown badge snippet for README / dataset cards",
    )
    p.add_argument(
        "--reference", "-r",
        metavar="REF",
        help="Optional reference profile for context (e.g. 'aloha', 'pusht')",
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

    log(f"Scoring {dataset_path!r} ...")

    try:
        report: DiagnosticReport = Pipeline().analyze_path(
            dataset_path,
            policy_family=args.policy,
            reader=reader,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    log(f"  {report.n_episodes} episodes  ·  {report.n_samples} steps")

    result = compute_score(report)

    # Attach reference comparison if requested
    if args.reference:
        try:
            from calibra.compare import load_reference, metrics_from_reference
            ref_data = load_reference(args.reference)
            ref_m    = metrics_from_reference(ref_data)
            result["reference"] = {
                "name":    args.reference,
                "metrics": ref_m,
            }
        except Exception as e:
            log(f"  (reference comparison skipped: {e})")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_score(result))
        if args.badge:
            print()
            print(render_badge(result))

    sys.exit(_exit_code(result["total_score"]))
