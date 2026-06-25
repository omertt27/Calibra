#!/usr/bin/env python3
"""
Cross-reference all calibra/claims/*.json against calibra/references/*.json.

Prints a status table showing which claims have evidence from profiled datasets,
which are still pending, and the current confidence level of each.

Usage:
    python scripts/validate_claims.py               # print status table
    python scripts/validate_claims.py --check       # CI gate: exit 1 on zero-evidence claims
    python scripts/validate_claims.py --pending     # show only claims needing more evidence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from calibra.claims import _derive_confidence  # noqa: E402


_CONF_RANK = {
    "STRONG": 5,
    "HIGH": 4,
    "MEDIUM": 3,
    "MODERATE": 2,
    "LOW-MODERATE": 2,
    "LOW": 1,
    "NOT VALIDATED": 0,
}

_CONF_COLOR = {
    "STRONG": "\033[92m",  # green
    "HIGH": "\033[92m",
    "MEDIUM": "\033[93m",  # yellow
    "MODERATE": "\033[93m",
    "LOW-MODERATE": "\033[33m",  # orange
    "LOW": "\033[33m",
    "NOT VALIDATED": "\033[91m",  # red
}
_RESET = "\033[0m"


def load_claims(claims_dir: Path) -> list[dict]:
    claims = []
    for path in sorted(claims_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
            continue
        for claim in data.get("claims", []):
            claim["_source_file"] = path.name
            claim["confidence"] = _derive_confidence(claim)
            claims.append(claim)
    return sorted(claims, key=lambda c: c.get("id", ""))


def load_reference_names(refs_dir: Path) -> set[str]:
    """Return the set of dataset names present in reference profiles."""
    names: set[str] = set()
    for path in refs_dir.glob("*.json"):
        try:
            meta = json.loads(path.read_text()).get("meta", {})
            ds = meta.get("dataset", "")
            if ds:
                names.add(ds)
            names.add(path.stem)  # also index by filename stem
        except (json.JSONDecodeError, KeyError):
            names.add(path.stem)
    return names


def _evidence_coverage(claim: dict, ref_names: set[str]) -> tuple[int, int]:
    """Return (n_evidence_in_refs, n_total_evidence)."""
    evidence = claim.get("evidence", [])
    n_total = len(evidence)
    n_in_refs = sum(
        1 for e in evidence if any(part in ref_names for part in _ds_parts(e.get("dataset", "")))
    )
    return n_in_refs, n_total


def _ds_parts(dataset_id: str) -> list[str]:
    """Generate candidate lookup keys from a dataset string (handles composite ids)."""
    parts = [dataset_id]
    if "/" in dataset_id:
        parts.append(dataset_id.split("/")[-1])
    return parts


def _pending_datasets(claim: dict) -> list[str]:
    falsify = claim.get("falsification", {})
    return [pt.get("dataset", "?") for pt in falsify.get("pending_tests", [])]


def print_table(claims: list[dict], ref_names: set[str], pending_only: bool) -> None:
    rows = []
    for claim in claims:
        cid = claim.get("id", "?")
        conf = claim.get("confidence", "NOT VALIDATED")
        n_refs_covered, n_evidence = _evidence_coverage(claim, ref_names)
        pending = _pending_datasets(claim)
        n_pending = len(pending)
        has_zero = n_evidence == 0

        if pending_only and not has_zero and n_pending == 0:
            continue

        rows.append(
            {
                "id": cid,
                "conf": conf,
                "n_evidence": n_evidence,
                "n_refs_covered": n_refs_covered,
                "n_pending": n_pending,
                "has_zero": has_zero,
                "pending_list": pending,
                "assertion": claim.get("assertion", "")[:65],
            }
        )

    if not rows:
        print("All claims have evidence and no pending tests.")
        return

    # Header
    w_id = max(len(r["id"]) for r in rows)
    w_conf = max(len(r["conf"]) for r in rows)
    sep = "─" * (w_id + w_conf + 50)
    print(sep)
    print(f"  {'ID':<{w_id}}  {'Confidence':<{w_conf}}  Evidence  Pending  Status")
    print(sep)

    for r in rows:
        conf_str = r["conf"]
        color = _CONF_COLOR.get(conf_str, "")
        if r["has_zero"]:
            status = "⬜ NO EVIDENCE"
        elif r["n_pending"] > 0:
            status = f"🔬 {r['n_pending']} pending"
        else:
            status = "✅ validated"

        print(
            f"  {r['id']:<{w_id}}  "
            f"{color}{conf_str:<{w_conf}}{_RESET}  "
            f"  {r['n_evidence']:<6}  {r['n_pending']:<7}  {status}"
        )
        if r["n_pending"] > 0:
            for ds in r["pending_list"][:2]:
                print(f"  {'':<{w_id}}  {'':<{w_conf}}    → {ds}")

    print(sep)
    zero_count = sum(1 for r in rows if r["has_zero"])
    total = len(claims)
    fully_validated = sum(
        1 for c in claims if len(c.get("evidence", [])) > 0 and not _pending_datasets(c)
    )
    print(
        f"\n  {total} claims  ·  {zero_count} with zero evidence  ·  {fully_validated} fully validated"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any claim has zero evidence (CI gate)",
    )
    parser.add_argument(
        "--pending",
        action="store_true",
        help="Show only claims that have zero evidence or open pending tests",
    )
    args = parser.parse_args()

    claims_dir = _REPO / "calibra" / "claims"
    refs_dir = _REPO / "calibra" / "references"

    claims = load_claims(claims_dir)
    ref_names = load_reference_names(refs_dir)

    if args.check:
        zero = [c for c in claims if not c.get("evidence")]
        if zero:
            ids = ", ".join(c.get("id", "?") for c in zero)
            print(
                f"❌ {len(zero)} claim(s) with zero evidence: {ids}\n"
                "   Profile more datasets with scripts/profile_batch.py before CI passes.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"✅ All {len(claims)} claims have at least one evidence entry.")
        sys.exit(0)

    print_table(claims, ref_names, pending_only=args.pending)


if __name__ == "__main__":
    main()
