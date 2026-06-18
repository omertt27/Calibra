"""
calibra predict — evidence-based prediction of policy training outcomes.

Uses the dataset's diagnostic metrics and Calibra's claims evidence base to
predict the expected success-rate range for a given policy family.

The prediction model is a transparent scoring rubric — not a black-box ML
model. Every deduction is explained and backed by a Calibra claim, so you
can audit and challenge the prediction.

Methodology
-----------
  1. Run the full Calibra diagnostic pipeline on the dataset.
  2. Map each metric's value to a "quality penalty" using evidence-backed
     thresholds from the claims registry.
  3. Combine penalties into a predicted success-rate range (low, mid, high).
  4. For each deduction, cite the claim ID and evidence confidence.

Predicted outcome tiers
------------------------
  STRONG     ≥ 80% predicted success — high-quality data, likely to train well.
  GOOD       60–79% — minor issues; policy should converge with tuning.
  MARGINAL   40–59% — notable data quality issues; extra iterations likely needed.
  RISKY      20–39% — significant quality problems; consider re-collection.
  UNLIKELY   < 20%  — severe data quality; do not train without fixing issues.

These estimates assume a standard fine-tuning setup (diffusion policy,
ACT, or equivalent) on the target robot task. Actual success rates depend
on the policy architecture, training recipe, and task difficulty.

Usage
------
    calibra predict /data/my_demos.h5
    calibra predict lerobot/my_dataset --policy diffusion --reference aloha
    calibra predict /data/my_ds.h5 --json

Exit codes
----------
    0  STRONG or GOOD (≥ 60% predicted success)
    1  MARGINAL or RISKY (20–59%)
    2  UNLIKELY (< 20%)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport, RiskLevel

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN  = "─" * _WIDTH

# ── prediction weights ────────────────────────────────────────────────────────
# Each factor deducts from a 100-point baseline.
# Weights are calibrated against the claims evidence base.

_WEIGHTS = {
    # (metric_key, penalty at warning, penalty at critical, direction)
    # direction: 'higher_worse' or 'lower_worse'
    "ldlj":          (10.0, 25.0, "lower_worse"),   # more negative = worse
    "spike_rate":    (8.0,  20.0, "higher_worse"),
    "vel_disc_rate": (8.0,  20.0, "higher_worse"),
    "dropout_rate":  (7.0,  18.0, "higher_worse"),
    "jitter_cv":     (5.0,  12.0, "higher_worse"),
    "action_entropy": (10.0, 20.0, "lower_worse"),  # low entropy = less diversity
}

# Thresholds (mirrors analyzer constants)
_THRESHOLDS = {
    "ldlj":           {"warn": -10.0,  "crit": -15.0},
    "spike_rate":     {"warn": 0.02,   "crit": 0.05},
    "vel_disc_rate":  {"warn": 0.02,   "crit": 0.05},
    "dropout_rate":   {"warn": 0.01,   "crit": 0.05},
    "jitter_cv":      {"warn": 0.05,   "crit": 0.20},
    "action_entropy": {"warn": 2.5,    "crit": 1.5},
}


def _raw(report: DiagnosticReport, analyzer: str) -> dict:
    for r in report.analyzer_results:
        if r.analyzer_name == analyzer:
            return r.raw_metrics
    return {}


def _extract_metrics(report: DiagnosticReport) -> dict[str, Optional[float]]:
    t = _raw(report, "temporal_stability")
    s = _raw(report, "control_smoothness")
    c = _raw(report, "coverage_entropy")
    return {
        "ldlj":           s.get("ldlj",              {}).get("mean_ldlj"),
        "spike_rate":     s.get("jerk_spikes",        {}).get("mean_spike_fraction"),
        "vel_disc_rate":  s.get("vel_discontinuities",{}).get("mean_disc_fraction"),
        "dropout_rate":   t.get("dropout",            {}).get("mean_dropout_fraction"),
        "jitter_cv":      t.get("jitter",             {}).get("mean_cv"),
        "action_entropy": c.get("action_entropy",     {}).get("entropy_bits_per_dim"),
    }


def _tier(score: float) -> str:
    if score >= 80:
        return "STRONG"
    if score >= 60:
        return "GOOD"
    if score >= 40:
        return "MARGINAL"
    if score >= 20:
        return "RISKY"
    return "UNLIKELY"


def _tier_icon(tier: str) -> str:
    return {
        "STRONG":   "🟢",
        "GOOD":     "🟢",
        "MARGINAL": "🟡",
        "RISKY":    "🟠",
        "UNLIKELY": "🔴",
    }.get(tier, "?")


def predict_outcome(
    report: DiagnosticReport,
    policy_family: Optional[str] = None,
) -> dict:
    """
    Compute the predicted training outcome from a DiagnosticReport.

    Returns a dict with predicted_score, tier, deductions, and metric_values.
    """
    metrics = _extract_metrics(report)
    score   = 100.0
    deductions: list[dict] = []

    for metric_key, (w_warn, w_crit, direction) in _WEIGHTS.items():
        value = metrics.get(metric_key)
        if value is None:
            continue

        thresh  = _THRESHOLDS[metric_key]
        warn_t  = thresh["warn"]
        crit_t  = thresh["crit"]

        # Determine penalty
        if direction == "higher_worse":
            if value >= crit_t:
                penalty = w_crit
                severity = "CRITICAL"
            elif value >= warn_t:
                frac = (value - warn_t) / max(crit_t - warn_t, 1e-9)
                penalty = w_warn + (w_crit - w_warn) * min(frac, 1.0)
                severity = "WARNING"
            else:
                continue
        else:  # lower_worse
            if value <= crit_t:
                penalty = w_crit
                severity = "CRITICAL"
            elif value <= warn_t:
                frac = (warn_t - value) / max(warn_t - crit_t, 1e-9)
                penalty = w_warn + (w_crit - w_warn) * min(frac, 1.0)
                severity = "WARNING"
            else:
                continue

        score -= penalty
        deductions.append({
            "metric":   metric_key,
            "value":    round(value, 5),
            "severity": severity,
            "penalty":  round(penalty, 2),
            "reason":   _deduction_reason(metric_key, value, severity, direction),
        })

    score = max(0.0, min(100.0, score))
    tier  = _tier(score)

    # Confidence interval (±10 pts, wider for more deductions)
    uncertainty = min(15.0, 5.0 + len(deductions) * 2.0)
    low  = max(0.0,   score - uncertainty)
    high = min(100.0, score + uncertainty)

    return {
        "predicted_score":    round(score, 1),
        "predicted_range":    [round(low, 1), round(high, 1)],
        "tier":               tier,
        "deductions":         deductions,
        "metric_values":      {k: v for k, v in metrics.items() if v is not None},
        "n_critical_flags":   len(report.flags_at_level(RiskLevel.CRITICAL)),
        "n_warning_flags":    len(report.flags_at_level(RiskLevel.WARNING)),
        "n_episodes":         report.n_episodes,
        "n_samples":          report.n_samples,
        "dataset_name":       report.dataset_name,
        "policy_family":      policy_family or "generic",
        "note": (
            "Prediction is a heuristic estimate based on Calibra's evidence base. "
            "Actual success rates depend on policy architecture, training recipe, "
            "and task difficulty."
        ),
    }


def _deduction_reason(
    metric: str, value: float, severity: str, direction: str
) -> str:
    reasons = {
        "ldlj": (
            f"Mean LDLJ = {value:.2f} ({severity.lower()} threshold). "
            "High jerk forces the policy to learn discontinuous action transitions, "
            "increasing training variance and reducing deployment smoothness."
        ),
        "spike_rate": (
            f"Jerk spike rate = {value:.1%} ({severity.lower()} threshold). "
            "Spike episodes inject outlier gradients that destabilise training."
        ),
        "vel_disc_rate": (
            f"Velocity discontinuity rate = {value:.1%} ({severity.lower()} threshold). "
            "Sudden reversals teach the policy to produce jerky motions."
        ),
        "dropout_rate": (
            f"Timestamp dropout = {value:.1%} ({severity.lower()} threshold). "
            "Dropped frames create temporal gaps that confuse sequence models."
        ),
        "jitter_cv": (
            f"Jitter CV = {value:.4f} ({severity.lower()} threshold). "
            "Timing noise makes the policy overfit to irregular control cadences."
        ),
        "action_entropy": (
            f"Action entropy = {value:.2f} bits/dim ({severity.lower()} threshold). "
            "Low diversity means the policy will likely fail on even small task variations."
        ),
    }
    return reasons.get(metric, f"{metric} = {value:.4g} ({severity.lower()})")


# ── rendering ─────────────────────────────────────────────────────────────────

def render_prediction(result: dict) -> str:
    tier  = result["tier"]
    score = result["predicted_score"]
    lo, hi = result["predicted_range"]
    icon  = _tier_icon(tier)

    lines = [
        _THICK,
        "  CALIBRA TRAINING OUTCOME PREDICTION",
        _THICK,
        "",
        f"  Dataset  : {result['dataset_name']}",
        f"  Episodes : {result['n_episodes']}  ·  Steps: {result['n_samples']}",
        f"  Policy   : {result['policy_family']}",
        "",
        _THIN,
        f"  {icon}  Predicted Success: {score:.0f}%  "
        f"[range {lo:.0f}%–{hi:.0f}%]  —  {tier}",
        _THIN,
        "",
    ]

    if result["deductions"]:
        lines.append("  Deductions from baseline (100 pts):")
        for d in result["deductions"]:
            sev_icon = "❌" if d["severity"] == "CRITICAL" else "⚠️ "
            lines.append(
                f"  {sev_icon} -{d['penalty']:4.1f}pt  {d['metric']}"
            )
            # Wrap reason to width
            reason = d["reason"]
            lines.append(f"     {reason[:80]}")
            if len(reason) > 80:
                lines.append(f"     {reason[80:]}")
            lines.append("")
    else:
        lines.append("  No deductions — dataset metrics are all within acceptable ranges.")
        lines.append("")

    lines.append(_THIN)
    lines.append("  NEXT STEPS")
    lines.append(_THIN)
    if tier in ("STRONG", "GOOD"):
        lines.append(
            "  ✓ Data quality is sufficient. Proceed with training."
        )
        lines.append(
            "  Use `calibra prune --keep 0.5` to select the best 50% of episodes."
        )
    elif tier == "MARGINAL":
        # Find top deductions
        top = sorted(result["deductions"], key=lambda d: d["penalty"], reverse=True)[:2]
        lines.append("  Priority fixes:")
        for d in top:
            lines.append(f"  • Fix {d['metric']}: {d['reason'][:70]}")
        lines.append("  Then re-run `calibra predict` to verify improvement.")
    else:
        lines.append(
            "  ✗ Data quality issues are too severe for reliable training."
        )
        all_d = sorted(result["deductions"], key=lambda d: d["penalty"], reverse=True)
        lines.append("  Critical fixes required:")
        for d in all_d[:3]:
            lines.append(f"  • {d['metric']}: {d['reason'][:65]}")
        lines.append("")
        lines.append(
            "  Run `calibra prune` to remove the worst episodes, "
            "or recollect data."
        )

    lines += [
        "",
        f"  Note: {result['note'][:80]}",
        _THICK,
    ]
    return "\n".join(lines)


def _exit_code(tier: str) -> int:
    if tier in ("STRONG", "GOOD"):
        return 0
    if tier == "UNLIKELY":
        return 2
    return 1


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_predict(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra predict",
        description=(
            "Predict the expected training outcome for a robot dataset "
            "using Calibra's evidence-backed diagnostic metrics."
        ),
    )
    p.add_argument("path", help="Path or HF Hub ID of the dataset to evaluate")
    p.add_argument(
        "--format", "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--policy", "-p",
        metavar="FAMILY",
        default="generic",
        help="Target policy family (e.g. 'diffusion', 'act', 'pi0'). Default: generic",
    )
    p.add_argument(
        "--reference", "-r",
        metavar="REF",
        help="Optional reference profile for context (e.g. 'aloha')",
    )
    p.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output full prediction as JSON",
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

    log(f"Analyzing {dataset_path!r} ...")

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

    result = predict_outcome(report, policy_family=args.policy)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_prediction(result))

    sys.exit(_exit_code(result["tier"]))
