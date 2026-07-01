"""
calibra sft-outcome — predict and record SFT training outcomes.

Reads the `aggregate_fingerprint` written by `calibra sft-select` and either
prints a heuristic + community-blended outcome prediction, or — after you've
actually run SFT on the selected coreset and evaluated it — records the
predicted-vs-actual result into the same outcome database (and Calibra Cloud
sync path) that `calibra predict --record-outcome` uses for robotics, tagged
with domain="llm_sft".

Usage
-----
    calibra sft-outcome --coreset coreset_index.json
    calibra sft-outcome --coreset coreset_index.json --policy llama3-8b-sft

    # After training + eval:
    calibra sft-outcome --coreset coreset_index.json \\
        --record-outcome 0.73 --model-family llama3-8b-sft \\
        --notes "MMLU 5-shot after 3 epochs"

Exit codes
----------
    0  STRONG or GOOD (>= 60% predicted score)
    1  MARGINAL or RISKY (20-59%)
    2  UNLIKELY (< 20%)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calibra.llm.predict import predict_sft_outcome, render_prediction


def _exit_code(tier: str) -> int:
    if tier in ("STRONG", "GOOD"):
        return 0
    if tier == "UNLIKELY":
        return 2
    return 1


def run_sft_outcome(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra sft-outcome",
        description="Predict and record SFT training outcomes from a calibra sft-select coreset.",
    )
    p.add_argument(
        "--coreset", required=True, metavar="PATH",
        help="Path to a coreset_index.json written by `calibra sft-select`",
    )
    p.add_argument(
        "--policy", "-p", metavar="FAMILY", default="generic",
        help="Model/training family for prediction context (e.g. 'llama3-8b-sft'). Default: generic",
    )
    p.add_argument(
        "--record-outcome", metavar="RATE", type=float,
        help="After training+eval, record the observed eval score (0.0-1.0) to improve "
        "future predictions on similar coresets. Example: --record-outcome 0.73",
    )
    p.add_argument(
        "--model-family", metavar="FAMILY", default=None,
        help="Model/training family to attach to the recorded outcome "
        "(defaults to --policy if not given)",
    )
    p.add_argument(
        "--dataset-name", metavar="NAME", default=None,
        help="Dataset name to attach to the recorded outcome (defaults to the coreset file name)",
    )
    p.add_argument("--notes", metavar="TEXT", default="", help="Optional annotation")
    p.add_argument(
        "--no-empirical", action="store_true",
        help="Disable empirical blending from the outcome database (pure heuristic).",
    )
    p.add_argument("--json", "-j", action="store_true", help="Output full prediction as JSON")
    args = p.parse_args(argv)

    coreset_path = Path(args.coreset)
    try:
        with open(coreset_path) as f:
            coreset = json.load(f)
    except Exception as exc:
        print(f"error reading {coreset_path}: {exc}", file=sys.stderr)
        sys.exit(2)

    aggregate_fingerprint = coreset.get("aggregate_fingerprint")
    if not aggregate_fingerprint:
        print(f"error: {coreset_path} has no 'aggregate_fingerprint' field", file=sys.stderr)
        sys.exit(2)

    n_examples = coreset.get("n_kept", len(coreset.get("keep_indices", [])))
    dataset_name = args.dataset_name or coreset_path.stem

    result = predict_sft_outcome(
        aggregate_fingerprint,
        policy_family=args.policy,
        use_outcome_db=not args.no_empirical,
        n_examples=n_examples,
        dataset_name=dataset_name,
    )

    if args.record_outcome is not None:
        actual_rate = float(args.record_outcome)
        if not (0.0 <= actual_rate <= 1.0):
            print("error: --record-outcome must be between 0.0 and 1.0", file=sys.stderr)
            sys.exit(2)
        try:
            from calibra.outcome_db import OutcomeDatabase

            db = OutcomeDatabase()
            rec = db.record(
                fingerprint=aggregate_fingerprint,
                predicted_score=result["heuristic_score"],
                actual_success_rate=actual_rate,
                policy_family=args.model_family or args.policy,
                n_episodes=n_examples,
                dataset_name=dataset_name,
                notes=args.notes,
                domain="llm_sft",
            )
            print(
                f"Outcome recorded (id={rec.record_id}): "
                f"predicted={result['heuristic_score']:.0f}%  actual={actual_rate:.0%}  "
                f"error={abs(result['heuristic_score'] / 100.0 - actual_rate) * 100:.1f}%",
                file=sys.stderr,
            )
            print(f"{db.summary()}", file=sys.stderr)
        except Exception as exc:
            print(f"warning: could not record outcome: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_prediction(result))

    sys.exit(_exit_code(result["tier"]))
