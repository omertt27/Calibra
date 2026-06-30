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
from pathlib import Path
from typing import Optional

from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport, RiskLevel

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN = "─" * _WIDTH

# ── prediction weights ────────────────────────────────────────────────────────
# Each factor deducts from a 100-point baseline.
# Weights are calibrated against the claims evidence base.

_WEIGHTS = {
    # (metric_key, penalty at warning, penalty at critical, direction)
    # direction: 'higher_worse' or 'lower_worse'
    "ldlj": (10.0, 25.0, "lower_worse"),  # more negative = worse
    "spike_rate": (8.0, 20.0, "higher_worse"),
    "vel_disc_rate": (8.0, 20.0, "higher_worse"),
    "dropout_rate": (7.0, 18.0, "higher_worse"),
    "jitter_cv": (5.0, 12.0, "higher_worse"),
    "action_entropy": (10.0, 20.0, "lower_worse"),  # low entropy = less diversity
    "contact_phase_fraction": (10.0, 20.0, "lower_worse"),
    # World-model learnability (from WorldModelConsistencyAnalyzer / RobotJEPA)
    # High mean JEPA surprise → world model can't predict the data → poor generalisation
    "jepa_surprise": (8.0, 18.0, "higher_worse"),
}

# Thresholds (mirrors analyzer constants)
_THRESHOLDS = {
    "ldlj": {"warn": -10.0, "crit": -15.0},
    "spike_rate": {"warn": 0.02, "crit": 0.05},
    "vel_disc_rate": {"warn": 0.02, "crit": 0.05},
    "dropout_rate": {"warn": 0.01, "crit": 0.05},
    "jitter_cv": {"warn": 0.05, "crit": 0.20},
    "action_entropy": {"warn": 2.5, "crit": 1.5},
    "contact_phase_fraction": {"warn": 0.10, "crit": 0.05},
    # JEPA surprise is normalised to [0, 1]; above 0.4 signals poor learnability
    "jepa_surprise": {"warn": 0.40, "crit": 0.70},
}


def _load_weights() -> dict[str, tuple[float, float, str]]:
    """Return _WEIGHTS, overriding warn penalties from ~/.calibra/weights.json if present."""
    weights_path = Path.home() / ".calibra" / "weights.json"
    if not weights_path.exists():
        return dict(_WEIGHTS)
    try:
        with open(weights_path) as f:
            data = json.load(f)
        custom: dict[str, float] = data.get("weights", {})
        if not custom:
            return dict(_WEIGHTS)
        result = {}
        for key, (w_warn, w_crit, direction) in _WEIGHTS.items():
            if key in custom:
                new_warn = float(custom[key])
                # Scale crit proportionally to preserve the warn:crit ratio
                ratio = w_crit / w_warn if w_warn > 0 else 2.0
                result[key] = (new_warn, round(new_warn * ratio, 2), direction)
            else:
                result[key] = (w_warn, w_crit, direction)
        return result
    except Exception:
        return dict(_WEIGHTS)


def get_weights_and_thresholds(
    policy_family: Optional[str] = None,
) -> tuple[dict[str, tuple[float, float, str]], dict[str, dict[str, float]]]:
    """Retrieve weights and thresholds customized for a specific policy family."""
    weights = _load_weights()
    thresholds = dict(_THRESHOLDS)
    if not policy_family:
        return weights, thresholds

    pf = policy_family.lower()

    # Universal calibrations from grid search
    thresholds["action_entropy"] = {"warn": 2.0, "crit": 1.0}
    thresholds["contact_phase_fraction"] = {"warn": 0.04, "crit": 0.02}

    if "act" in pf or "pi0" in pf:
        # ACT/pi0 use position/chunking and are highly sensitive to velocity discontinuities and jerk spikes
        thresholds["ldlj"] = {"warn": -10.0, "crit": -15.0}
        thresholds["spike_rate"] = {"warn": 0.005, "crit": 0.01}
        thresholds["vel_disc_rate"] = {"warn": 0.04, "crit": 0.08}
    elif "diffusion" in pf:
        # Diffusion Policy handles multi-modal action trajectories and jerk, but is slightly sensitive
        thresholds["ldlj"] = {"warn": -20.0, "crit": -25.0}
        thresholds["spike_rate"] = {"warn": 0.05, "crit": 0.10}
        thresholds["vel_disc_rate"] = {"warn": 0.15, "crit": 0.30}
    else:
        # Other policy families (e.g. Octo/VLA)
        thresholds["ldlj"] = {"warn": -15.0, "crit": -25.0}
        thresholds["spike_rate"] = {"warn": 0.05, "crit": 0.10}
        thresholds["vel_disc_rate"] = {"warn": 0.05, "crit": 0.15}

    return weights, thresholds


def _raw(report: DiagnosticReport, analyzer: str) -> dict:
    for r in report.analyzer_results:
        if r.analyzer_name == analyzer:
            return r.raw_metrics
    return {}


def _extract_metrics(report: DiagnosticReport) -> dict[str, Optional[float]]:
    t = _raw(report, "temporal_stability")
    s = _raw(report, "control_smoothness")
    c = _raw(report, "coverage_entropy")
    pb = _raw(report, "phase_balance")
    wm = _raw(report, "world_model_consistency")
    return {
        "ldlj": s.get("ldlj", {}).get("mean_ldlj"),
        "spike_rate": s.get("jerk_spikes", {}).get("mean_spike_fraction"),
        "vel_disc_rate": s.get("vel_discontinuities", {}).get("mean_disc_fraction"),
        "dropout_rate": t.get("dropout", {}).get("mean_dropout_fraction"),
        "jitter_cv": t.get("jitter", {}).get("mean_cv"),
        "action_entropy": c.get("action_entropy", {}).get("entropy_bits_per_dim"),
        "contact_phase_fraction": pb.get("mean_contact_fraction"),
        # JEPA world-model surprise (only present if WorldModelConsistencyAnalyzer ran)
        "jepa_surprise": wm.get("mean_surprise"),
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
        "STRONG": "🟢",
        "GOOD": "🟢",
        "MARGINAL": "🟡",
        "RISKY": "🟠",
        "UNLIKELY": "🔴",
    }.get(tier, "?")


def predict_outcome(
    report: DiagnosticReport,
    policy_family: Optional[str] = None,
    use_outcome_db: bool = True,
) -> dict:
    """
    Compute the predicted training outcome from a DiagnosticReport.

    When use_outcome_db=True (default), blends the heuristic score with
    empirically observed outcomes from similar past datasets stored in
    ~/.calibra/outcomes.jsonl.

    Returns a dict with predicted_score, tier, deductions, and metric_values.
    """
    metrics = _extract_metrics(report)
    score = 100.0
    deductions: list[dict] = []

    weights, thresholds = get_weights_and_thresholds(policy_family)

    is_scripted = any(
        f.metric == "motion_collection_signature" for r in report.analyzer_results for f in r.flags
    )

    for metric_key, (w_warn, w_crit, direction) in weights.items():
        value = metrics.get(metric_key)
        if value is None:
            continue

        thresh = thresholds[metric_key]
        warn_t = thresh["warn"]
        crit_t = thresh["crit"]

        w_w = w_warn
        w_c = w_crit
        if is_scripted and metric_key in ["ldlj", "spike_rate", "vel_disc_rate"]:
            w_w *= 0.4
            w_c *= 0.4

        # Determine penalty
        if direction == "higher_worse":
            if value >= crit_t:
                penalty = w_c
                severity = "CRITICAL"
            elif value >= warn_t:
                frac = (value - warn_t) / max(crit_t - warn_t, 1e-9)
                penalty = w_w + (w_c - w_w) * min(frac, 1.0)
                severity = "WARNING"
            else:
                continue
        else:  # lower_worse
            if value <= crit_t:
                penalty = w_c
                severity = "CRITICAL"
            elif value <= warn_t:
                frac = (warn_t - value) / max(warn_t - crit_t, 1e-9)
                penalty = w_w + (w_c - w_w) * min(frac, 1.0)
                severity = "WARNING"
            else:
                continue

        score -= penalty
        deductions.append(
            {
                "metric": metric_key,
                "value": round(value, 5),
                "severity": severity,
                "penalty": round(penalty, 2),
                "reason": _deduction_reason(metric_key, value, severity, direction),
            }
        )

    heuristic_score = max(0.0, min(100.0, score))

    # ── empirical blending from outcome DB ───────────────────────────────────
    empirical_weight = 0.0
    similar_outcomes: list[dict] = []
    blended_score = heuristic_score

    if use_outcome_db:
        try:
            from calibra.outcome_db import OutcomeDatabase

            db = OutcomeDatabase()
            fp = {k: v for k, v in metrics.items() if v is not None}
            similar = db.find_similar(fp, policy_family=policy_family)
            if similar:
                blended_score, empirical_weight = db.blend_prediction(heuristic_score, similar)
                similar_outcomes = [
                    {
                        "dataset": rec.dataset_name,
                        "actual_rate": rec.actual_success_rate,
                        "distance": round(dist, 3),
                        "policy": rec.policy_family,
                    }
                    for rec, dist in similar
                ]
        except Exception:
            pass

    final_score = blended_score
    tier = _tier(final_score)

    uncertainty = min(15.0, 5.0 + len(deductions) * 2.0)
    if empirical_weight > 0:
        uncertainty *= 1.0 - empirical_weight * 0.5
    low = max(0.0, final_score - uncertainty)
    high = min(100.0, final_score + uncertainty)

    if empirical_weight > 0:
        note = (
            f"Score blended: {round(empirical_weight * 100)}% empirical weight from "
            f"{len(similar_outcomes)} similar past dataset(s). "
            "Record outcomes with `calibra predict --record-outcome RATE` to improve future predictions."
        )
    else:
        note = (
            "Prediction is a heuristic estimate based on Calibra's evidence base. "
            "Actual success rates depend on policy architecture, training recipe, "
            "and task difficulty."
        )

    return {
        "predicted_score": round(final_score, 1),
        "heuristic_score": round(heuristic_score, 1),
        "empirical_weight": round(empirical_weight, 3),
        "similar_outcomes": similar_outcomes,
        "predicted_success_rate": round(final_score / 100.0, 3),
        "predicted_range": [round(low, 1), round(high, 1)],
        "tier": tier,
        "deductions": deductions,
        "metric_values": {k: v for k, v in metrics.items() if v is not None},
        "n_critical_flags": len(report.flags_at_level(RiskLevel.CRITICAL)),
        "n_warning_flags": len(report.flags_at_level(RiskLevel.WARNING)),
        "n_episodes": report.n_episodes,
        "n_samples": report.n_samples,
        "dataset_name": report.dataset_name,
        "policy_family": policy_family or "generic",
        "note": note,
    }


def _deduction_reason(metric: str, value: float, severity: str, direction: str) -> str:
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
        "contact_phase_fraction": (
            f"Contact fraction = {value:.1%} ({severity.lower()} threshold). "
            "Underrepresented contact/grasp phase means policy struggles with contact/manipulation precision."
        ),
        "jepa_surprise": (
            f"Mean JEPA surprise = {value:.3f} ({severity.lower()} threshold). "
            "The RobotJEPA world model cannot reliably predict state transitions in this dataset. "
            "High surprise correlates with corrupted or inconsistent dynamics that cause IL policies "
            "to fail on novel start states outside the training distribution."
        ),
    }
    return reasons.get(metric, f"{metric} = {value:.4g} ({severity.lower()})")


# ── rendering ─────────────────────────────────────────────────────────────────


def render_prediction(result: dict) -> str:
    tier = result["tier"]
    score = result["predicted_score"]
    lo, hi = result["predicted_range"]
    icon = _tier_icon(tier)
    emp_w = result.get("empirical_weight", 0.0)

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
        f"  {icon}  Predicted Success: {score:.0f}%  [range {lo:.0f}%–{hi:.0f}%]  —  {tier}",
    ]

    if emp_w > 0:
        heuristic = result.get("heuristic_score", score)
        lines.append(
            f"  📊  Empirical blend: {round(emp_w * 100)}% observed / "
            f"{round((1 - emp_w) * 100)}% heuristic  (heuristic alone: {heuristic:.0f}%)"
        )
        for s in result.get("similar_outcomes", [])[:3]:
            lines.append(
                f"    ↳ {s['dataset']} [{s['policy']}]  "
                f"actual={s['actual_rate']:.0%}  dist={s['distance']:.2f}"
            )

    lines += [_THIN, ""]

    if result["deductions"]:
        lines.append("  Deductions from baseline (100 pts):")
        for d in result["deductions"]:
            sev_icon = "❌" if d["severity"] == "CRITICAL" else "⚠️ "
            lines.append(f"  {sev_icon} -{d['penalty']:4.1f}pt  {d['metric']}")
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
        lines.append("  ✓ Data quality is sufficient. Proceed with training.")
        lines.append("  Use `calibra prune --keep 0.5` to select the best 50% of episodes.")
        lines.append("  After training, close the loop:")
        lines.append("    calibra predict <dataset> --record-outcome <actual_success_rate>")
    elif tier == "MARGINAL":
        top = sorted(result["deductions"], key=lambda d: d["penalty"], reverse=True)[:2]
        lines.append("  Priority fixes:")
        for d in top:
            lines.append(f"  • Fix {d['metric']}: {d['reason'][:70]}")
        lines.append("  Then re-run `calibra predict` to verify improvement.")
    else:
        lines.append("  ✗ Data quality issues are too severe for reliable training.")
        all_d = sorted(result["deductions"], key=lambda d: d["penalty"], reverse=True)
        lines.append("  Critical fixes required:")
        for d in all_d[:3]:
            lines.append(f"  • {d['metric']}: {d['reason'][:65]}")
        lines.append("")
        lines.append("  Run `calibra prune` to remove the worst episodes, or recollect data.")

    lines += [
        "",
        f"  Note: {result['note'][:90]}",
        _THICK,
    ]
    return "\n".join(lines)


def _exit_code(tier: str) -> int:
    if tier in ("STRONG", "GOOD"):
        return 0
    if tier == "UNLIKELY":
        return 2
    return 1


# ── community context ─────────────────────────────────────────────────────────


def _estimate_pct(val: float, p: dict) -> Optional[float]:
    """Linear interpolation of percentile rank from p25/p50/p75/p90 landmarks."""
    p25, p50, p75, p90 = p.get("p25"), p.get("p50"), p.get("p75"), p.get("p90")
    if None in (p25, p50, p75, p90):
        return None
    if val <= p25:
        return max(0.0, 25.0 * val / p25) if p25 != 0 else 25.0
    if val <= p50:
        return 25.0 + 25.0 * (val - p25) / max(p50 - p25, 1e-9)
    if val <= p75:
        return 50.0 + 25.0 * (val - p50) / max(p75 - p50, 1e-9)
    if val <= p90:
        return 75.0 + 15.0 * (val - p75) / max(p90 - p75, 1e-9)
    return 90.0


def _print_community_context(metrics: dict[str, float], policy_family: str) -> None:
    """Fetch community percentiles and print a 1-2 line community comparison to stderr."""
    import os
    import urllib.request

    base = os.environ.get("CALIBRA_CLOUD_URL", "https://app.calibra.io")
    url = f"{base}/v1/percentiles?policy_family={policy_family}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception:
        return

    n_community = data.get("n", 0)
    if n_community < 5:
        return

    percentiles = data.get("percentiles", {})
    # For each metric, estimate percentile rank (direction-aware)
    _lower_better = {"spike_rate", "vel_disc_rate", "dropout_rate", "jitter_cv", "jepa_surprise"}
    _higher_better = {"action_entropy", "contact_phase_fraction"}
    # ldlj: higher (less negative) = better

    highlights: list[str] = []
    for key, val in metrics.items():
        p = percentiles.get(key)
        if p is None:
            continue
        pct = _estimate_pct(val, p)
        if pct is None:
            continue
        # Convert raw percentile to "better than X%" framing
        if key in _lower_better:
            better_pct = 100.0 - pct  # lower value = better = higher rank
        else:
            better_pct = pct  # higher value = better
        highlights.append((key, better_pct))

    if not highlights:
        return

    highlights.sort(key=lambda x: abs(x[1] - 50), reverse=True)
    print(
        f"  Community: {n_community} datasets in Calibra Cloud for '{policy_family}'.",
        file=sys.stderr,
    )
    for key, better_pct in highlights[:2]:
        rank_str = f"better than {better_pct:.0f}%" if better_pct >= 50 else f"worse than {100 - better_pct:.0f}%"
        print(f"    {key}: {rank_str} of community", file=sys.stderr)


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
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--policy",
        "-p",
        metavar="FAMILY",
        default="generic",
        help="Target policy family (e.g. 'diffusion', 'act', 'pi0'). Default: generic",
    )
    p.add_argument(
        "--reference",
        "-r",
        metavar="REF",
        help="Optional reference profile for context (e.g. 'aloha')",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output full prediction as JSON",
    )
    p.add_argument(
        "--record-outcome",
        metavar="RATE",
        type=float,
        help=(
            "After training, record the observed success rate (0.0–1.0) to improve "
            "future predictions on similar datasets. Example: --record-outcome 0.82"
        ),
    )
    p.add_argument(
        "--notes",
        metavar="TEXT",
        default="",
        help="Optional annotation to attach to the recorded outcome (use with --record-outcome).",
    )
    p.add_argument(
        "--no-empirical",
        action="store_true",
        help="Disable empirical blending from outcome database (pure heuristic).",
    )
    p.add_argument(
        "--world-model",
        action="store_true",
        help=(
            "Train a RobotJEPA world model on the dataset and include "
            "the JEPA surprise score in the prediction. Requires PyTorch. "
            "Adds ~1-2 min runtime on M2 Pro."
        ),
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

    log(f"Analyzing {dataset_path!r} ...")

    try:
        report: DiagnosticReport = Pipeline(
            world_model=getattr(args, "world_model", False)
        ).analyze_path(
            dataset_path,
            policy_family=args.policy,
            reader=reader,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    log(f"  {report.n_episodes} episodes  ·  {report.n_samples} steps")

    use_db = not args.no_empirical
    result = predict_outcome(report, policy_family=args.policy, use_outcome_db=use_db)

    if args.record_outcome is not None:
        actual_rate = float(args.record_outcome)
        if not (0.0 <= actual_rate <= 1.0):
            print("error: --record-outcome must be between 0.0 and 1.0", file=sys.stderr)
            sys.exit(2)
        try:
            from calibra.outcome_db import OutcomeDatabase

            db = OutcomeDatabase()
            rec = db.record(
                fingerprint=result["metric_values"],
                predicted_score=result["heuristic_score"],
                actual_success_rate=actual_rate,
                policy_family=result["policy_family"],
                n_episodes=result["n_episodes"],
                dataset_name=result["dataset_name"],
                notes=getattr(args, "notes", ""),
            )
            log(
                f"  Outcome recorded (id={rec.record_id}): "
                f"predicted={result['heuristic_score']:.0f}%  actual={actual_rate:.0%}  "
                f"error={abs(result['heuristic_score'] / 100.0 - actual_rate) * 100:.1f}%"
            )
            log(f"  {db.summary()}")
            log(
                "  Outcome recorded locally."
                " Run `calibra login` to sync outcomes to the global prediction model."
            )
            _print_community_context(result["metric_values"], result["policy_family"])
        except Exception as exc:
            log(f"  Warning: could not record outcome: {exc}")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_prediction(result))

    sys.exit(_exit_code(result["tier"]))
