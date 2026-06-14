"""
Claim registry — structured hypotheses about metric behavior across dataset classes.

Each claim records:
  - what it asserts
  - what data currently supports it
  - what would falsify it
  - its current confidence level

Confidence levels reflect evidence count, not subjective belief:
  HIGH           >= 5 supporting datasets, no counter-evidence
  MODERATE       2–4 supporting datasets
  LOW-MODERATE   1 supporting dataset
  LOW            no real-data evidence (synthetic fixtures only)
  NOT VALIDATED  claim has been made but zero datasets tested

When a new dataset is profiled, check whether it supports or falsifies active claims.
Update the relevant JSON file accordingly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_CLAIMS_DIR = Path(__file__).parent / "claims"

# Metric key used in compare.py → claim metric field
_METRIC_KEYS = {
    "vel_disc_rate":  "velocity_discontinuity_rate",
    "spike_rate":     "spike_rate",
    "ldlj":           "ldlj",
    "jitter_cv":      "jitter_cv",
    "dropout_rate":   "dropout_rate",
    "action_entropy": "action_entropy",
}


def _derive_confidence(claim: dict) -> str:
    """
    Confidence is derived from evidence count, not asserted.
    See SPEC.md for the derivation scale.
    """
    n = sum(1 for e in claim.get("evidence", []) if e.get("supports", True))
    if n == 0:
        return "NOT VALIDATED"
    if n == 1:
        return "LOW-MODERATE"
    if n <= 4:
        return "MODERATE"
    if n <= 9:
        return "HIGH"
    return "STRONG"


def load_all() -> dict[str, dict]:
    """Load all claim files. Returns dict keyed by claim id."""
    claims: dict[str, dict] = {}
    for path in sorted(_CLAIMS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            for claim in data.get("claims", []):
                claim["confidence"] = _derive_confidence(claim)
                claims[claim["id"]] = claim
        except (json.JSONDecodeError, KeyError):
            pass
    return claims


def get(metric_key: str, class_name: str) -> list[dict]:
    """
    Return active claims for a metric key (as used in compare.py) and class name.
    Class name is matched against the claim's 'class' field; 'any' matches everything.
    """
    metric = _METRIC_KEYS.get(metric_key, metric_key)
    active_statuses = {"active_hypothesis", "validated"}
    return [
        c for c in load_all().values()
        if c.get("metric") == metric
        and c.get("status") in active_statuses
        and (c.get("class") == class_name or c.get("class") == "any")
    ]


def evidence_line(metric_key: str, class_name: str) -> str:
    """
    One-line evidence summary for display in comparison output.
    Shows evidence count, source datasets, and next pending test.
    """
    relevant = get(metric_key, class_name)
    if not relevant:
        return ""

    # Pick the most specific claim (class match over 'any')
    specific = [c for c in relevant if c.get("class") == class_name]
    claim = specific[0] if specific else relevant[0]

    confidence = claim.get("confidence", "UNKNOWN")
    evidence = claim.get("evidence", [])
    n = len(evidence)

    if n == 0:
        base = f"{confidence} · n=0 (no real-data evidence)"
    else:
        names = [_short(e.get("dataset", "?")) for e in evidence]
        base = f"{confidence} · n={n} ({', '.join(names)})"

    pending = claim.get("falsification", {}).get("pending_tests", [])
    if pending:
        next_test = pending[0].get("dataset", "")
        base += f" · pending {next_test}"

    return f"[{base}]"


def _short(dataset_name: str) -> str:
    return dataset_name.split("/")[-1]
