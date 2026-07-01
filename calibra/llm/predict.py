"""
calibra.llm.predict — evidence-based prediction of SFT training outcomes.

The `llm_sft` domain analogue of `calibra/predict.py`: a transparent scoring
rubric over the aggregate fingerprint produced by
`calibra.llm.select.SFTCoresetSelector` (mean_coherence, repetition_rate,
template_ratio, mean_response_length, diversity_nn_dist), blended with
empirically observed outcomes from `calibra.outcome_db.OutcomeDatabase`
(domain="llm_sft") the same way the robotics predictor blends in
`calibra/predict.py::predict_outcome`.

Usage
-----
    from calibra.llm.predict import predict_sft_outcome

    result = predict_sft_outcome(selection_result.aggregate_fingerprint)
    print(result["tier"], result["predicted_score"])
"""

from __future__ import annotations

from typing import Optional

# (metric_key, penalty at warning, penalty at critical, direction)
_SFT_WEIGHTS: dict[str, tuple[float, float, str]] = {
    "mean_coherence": (15.0, 30.0, "lower_worse"),
    "repetition_rate": (10.0, 20.0, "higher_worse"),
    "template_ratio": (8.0, 18.0, "higher_worse"),
    "mean_response_length": (10.0, 20.0, "lower_worse"),
    "diversity_nn_dist": (12.0, 22.0, "lower_worse"),
}

_SFT_THRESHOLDS: dict[str, dict[str, float]] = {
    "mean_coherence": {"warn": 0.15, "crit": 0.05},
    "repetition_rate": {"warn": 0.25, "crit": 0.40},
    "template_ratio": {"warn": 0.10, "crit": 0.30},
    "mean_response_length": {"warn": 15.0, "crit": 5.0},
    "diversity_nn_dist": {"warn": 0.15, "crit": 0.05},
}

_REASONS = {
    "mean_coherence": (
        "Mean coherence = {value:.3f} ({severity} threshold). "
        "Low instruction/response similarity means many examples don't actually "
        "address their prompt, teaching the model to ignore instructions."
    ),
    "repetition_rate": (
        "Repetition rate = {value:.1%} ({severity} threshold). "
        "Self-repetitive responses reinforce degenerate output patterns."
    ),
    "template_ratio": (
        "Template ratio = {value:.1%} ({severity} threshold). "
        "Boilerplate openers dominate responses, teaching stock phrasing over content."
    ),
    "mean_response_length": (
        "Mean response length = {value:.1f} words ({severity} threshold). "
        "Very short responses under-specify the target behavior."
    ),
    "diversity_nn_dist": (
        "Diversity (avg NN distance) = {value:.3f} ({severity} threshold). "
        "Low embedding-space spread means the coreset is redundant and under-covers "
        "the instruction distribution."
    ),
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


def predict_sft_outcome(
    aggregate_fingerprint: dict[str, float],
    policy_family: str = "generic",
    use_outcome_db: bool = True,
    n_examples: int = 0,
    dataset_name: str = "unknown",
) -> dict:
    """
    Compute the predicted SFT training outcome from an aggregate fingerprint.

    When use_outcome_db=True (default), blends the heuristic score with
    empirically observed llm_sft outcomes from ~/.calibra/outcomes.jsonl.

    Returns a dict with the same shape as `calibra.predict.predict_outcome`:
    predicted_score, tier, deductions, similar_outcomes, metric_values, note.
    """
    score = 100.0
    deductions: list[dict] = []

    for metric_key, (w_warn, w_crit, direction) in _SFT_WEIGHTS.items():
        value = aggregate_fingerprint.get(metric_key)
        if value is None:
            continue

        thresh = _SFT_THRESHOLDS[metric_key]
        warn_t, crit_t = thresh["warn"], thresh["crit"]

        if direction == "higher_worse":
            if value >= crit_t:
                penalty, severity = w_crit, "CRITICAL"
            elif value >= warn_t:
                frac = (value - warn_t) / max(crit_t - warn_t, 1e-9)
                penalty, severity = w_warn + (w_crit - w_warn) * min(frac, 1.0), "WARNING"
            else:
                continue
        else:  # lower_worse
            if value <= crit_t:
                penalty, severity = w_crit, "CRITICAL"
            elif value <= warn_t:
                frac = (warn_t - value) / max(warn_t - crit_t, 1e-9)
                penalty, severity = w_warn + (w_crit - w_warn) * min(frac, 1.0), "WARNING"
            else:
                continue

        score -= penalty
        deductions.append(
            {
                "metric": metric_key,
                "value": round(value, 5),
                "severity": severity,
                "penalty": round(penalty, 2),
                "reason": _REASONS[metric_key].format(value=value, severity=severity.lower()),
            }
        )

    heuristic_score = max(0.0, min(100.0, score))

    empirical_weight = 0.0
    similar_outcomes: list[dict] = []
    blended_score = heuristic_score

    if use_outcome_db:
        try:
            from calibra.outcome_db import OutcomeDatabase

            db = OutcomeDatabase()
            similar = db.find_similar(
                aggregate_fingerprint, policy_family=policy_family, domain="llm_sft"
            )
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
            f"{len(similar_outcomes)} similar past SFT run(s). "
            "Record outcomes with `calibra sft-outcome --record-outcome RATE` to improve "
            "future predictions."
        )
    else:
        note = (
            "Prediction is a heuristic estimate over the coreset's aggregate fingerprint. "
            "Actual eval scores depend on model architecture, training recipe, and eval suite."
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
        "metric_values": {k: v for k, v in aggregate_fingerprint.items() if v is not None},
        "n_examples": n_examples,
        "dataset_name": dataset_name,
        "policy_family": policy_family,
        "domain": "llm_sft",
        "note": note,
    }


def render_prediction(result: dict) -> str:
    """Render a predict_sft_outcome() result as human-readable text (mirrors calibra.predict)."""
    width = 60
    thick, thin = "━" * width, "─" * width
    tier = result["tier"]
    score = result["predicted_score"]
    lo, hi = result["predicted_range"]
    emp_w = result.get("empirical_weight", 0.0)

    lines = [
        thick,
        "  CALIBRA SFT OUTCOME PREDICTION",
        thick,
        "",
        f"  Dataset  : {result['dataset_name']}",
        f"  Examples : {result['n_examples']}",
        f"  Model    : {result['policy_family']}",
        "",
        thin,
        f"  Predicted Eval Score: {score:.0f}%  [range {lo:.0f}%–{hi:.0f}%]  —  {tier}",
    ]

    if emp_w > 0:
        heuristic = result.get("heuristic_score", score)
        lines.append(
            f"  Empirical blend: {round(emp_w * 100)}% observed / "
            f"{round((1 - emp_w) * 100)}% heuristic  (heuristic alone: {heuristic:.0f}%)"
        )
        for s in result.get("similar_outcomes", [])[:3]:
            lines.append(
                f"    - {s['dataset']} [{s['policy']}]  "
                f"actual={s['actual_rate']:.0%}  dist={s['distance']:.2f}"
            )

    lines += [thin, ""]

    if result["deductions"]:
        lines.append("  Deductions from baseline (100 pts):")
        for d in result["deductions"]:
            sev = "CRIT" if d["severity"] == "CRITICAL" else "WARN"
            lines.append(f"  [{sev}] -{d['penalty']:4.1f}pt  {d['metric']}")
            lines.append(f"     {d['reason']}")
    else:
        lines.append("  No deductions — aggregate fingerprint is within acceptable ranges.")

    lines += ["", thin, f"  Note: {result['note']}", thick]
    return "\n".join(lines)
