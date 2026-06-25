"""
calibra card — generate a HuggingFace dataset quality card.

Runs the full Calibra diagnostic pipeline on a dataset and produces a
structured quality card that can be inserted into a HuggingFace dataset
README (YAML front-matter + Markdown section).

The card includes:
  • Overall certification status (CERTIFIED / PROVISIONALLY CERTIFIED / NOT CERTIFIED)
  • Per-metric quality summary table
  • Coreset size recommendation
  • Predicted training outcome per policy family
  • Calibra version and profiling date

Usage
-----
    calibra card /data/my_demos.h5
    calibra card lerobot/my_dataset --policy diffusion --push
    calibra card /data/demo.h5 --policy act --json
    calibra card /data/my_ds.h5 --out quality_card.md

Exit codes
----------
    0  CERTIFIED or PROVISIONALLY CERTIFIED
    1  NOT CERTIFIED (critical quality failures)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Optional

from calibra import __version__
from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport, RiskLevel
from calibra.predict import predict_outcome


_WIDTH = 60
_THICK = "━" * _WIDTH


def _metric_row(
    name: str,
    value: Optional[float],
    unit: str,
    warn: float,
    crit: float,
    direction: str = "higher_worse",
) -> str:
    """Return a markdown table row with a status icon."""
    if value is None:
        return f"| {name} | — | {unit} | ✅ |"

    if direction == "higher_worse":
        if value >= crit:
            icon = "❌"
        elif value >= warn:
            icon = "⚠️"
        else:
            icon = "✅"
    else:  # lower_worse
        if value <= crit:
            icon = "❌"
        elif value <= warn:
            icon = "⚠️"
        else:
            icon = "✅"

    return f"| {name} | {value:.4g} | {unit} | {icon} |"


def generate_card(
    report: DiagnosticReport,
    policy_family: Optional[str] = None,
) -> str:
    """
    Generate a HuggingFace dataset card Markdown string.

    The output is designed to be appended to (or embedded in) a dataset's
    README.md on HuggingFace Hub.
    """
    from calibra.predict import _extract_metrics

    metrics = _extract_metrics(report)
    pred = predict_outcome(report, policy_family=policy_family, use_outcome_db=False)

    n_crit = len(report.flags_at_level(RiskLevel.CRITICAL))
    n_warn = len(report.flags_at_level(RiskLevel.WARNING))

    if n_crit == 0 and n_warn == 0:
        status = "CERTIFIED"
        badge = "![Calibra: CERTIFIED](https://img.shields.io/badge/Calibra-CERTIFIED-brightgreen)"
    elif n_crit == 0:
        status = "PROVISIONALLY CERTIFIED"
        badge = "![Calibra: PROVISIONALLY CERTIFIED](https://img.shields.io/badge/Calibra-PROVISIONAL-yellow)"
    else:
        status = "NOT CERTIFIED"
        badge = (
            "![Calibra: NOT CERTIFIED](https://img.shields.io/badge/Calibra-NOT%20CERTIFIED-red)"
        )

    today = date.today().isoformat()
    policy_str = policy_family or "generic"

    rows = [
        _metric_row("Jerk spike rate", metrics.get("spike_rate"), "fraction", 0.02, 0.05),
        _metric_row("Velocity discontinuity", metrics.get("vel_disc_rate"), "fraction", 0.02, 0.05),
        _metric_row(
            "LDLJ smoothness", metrics.get("ldlj"), "score", -10.0, -15.0, direction="lower_worse"
        ),
        _metric_row("Timestamp dropout", metrics.get("dropout_rate"), "fraction", 0.01, 0.05),
        _metric_row("Jitter CV", metrics.get("jitter_cv"), "CV", 0.05, 0.20),
        _metric_row(
            "Action entropy",
            metrics.get("action_entropy"),
            "bits/dim",
            2.5,
            1.5,
            direction="lower_worse",
        ),
        _metric_row(
            "Contact phase fraction",
            metrics.get("contact_phase_fraction"),
            "fraction",
            0.10,
            0.05,
            direction="lower_worse",
        ),
    ]

    pred_score = pred["predicted_score"]
    pred_lo, pred_hi = pred["predicted_range"]

    card = f"""\
## Dataset Quality Report (Calibra v{__version__})

{badge}

> **{report.dataset_name}** · Profiled on {today} · {report.n_episodes} episodes · {report.n_samples:,} steps · policy: `{policy_str}`

**Certification status:** {status}

### Metric Summary

| Metric | Value | Unit | Status |
|--------|-------|------|--------|
{chr(10).join(rows)}

### Training Outcome Prediction

| Policy | Predicted success | Range |
|--------|------------------|-------|
| `{policy_str}` | {pred_score:.0f}% | {pred_lo:.0f}%–{pred_hi:.0f}% |

**Calibra tier:** {pred["tier"]}

"""

    if pred["deductions"]:
        card += "### Quality Issues\n\n"
        for d in pred["deductions"]:
            sev = "**CRITICAL**" if d["severity"] == "CRITICAL" else "_WARNING_"
            card += f"- {sev} `{d['metric']}` — {d['reason'][:120]}\n"
        card += "\n"

    card += (
        f"### Recommended Coreset\n\n"
        f"Run `calibra prune <dataset> --keep 0.3` to select the most diverse "
        f"30% of quality-passing episodes before training.\n\n"
        f"---\n"
        f"_Generated by [Calibra](https://github.com/omerTT/Calibra) v{__version__}_\n"
    )

    return card


def generate_yaml_frontmatter(report: DiagnosticReport) -> str:
    """
    Generate YAML front-matter tags for HuggingFace dataset cards.

    Adds `calibra_certified: true/false` and quality metric tags.
    """
    n_crit = len(report.flags_at_level(RiskLevel.CRITICAL))
    certified = "true" if n_crit == 0 else "false"

    return (
        f"calibra_certified: {certified}\n"
        f'calibra_version: "{__version__}"\n'
        f"calibra_n_episodes: {report.n_episodes}\n"
    )


def push_to_hub(card_text: str, repo_id: str) -> None:
    """
    Append the quality card section to the dataset's README on HuggingFace Hub.

    Requires: `pip install huggingface_hub`
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "error: huggingface_hub is not installed. Run: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi()
    try:
        existing = api.dataset_info(repo_id).card_data
        existing_readme = existing.text if existing and hasattr(existing, "text") else ""
    except Exception:
        existing_readme = ""

    marker = "## Dataset Quality Report (Calibra"
    if marker in existing_readme:
        # Replace existing Calibra section
        start = existing_readme.index(marker)
        readme = existing_readme[:start] + card_text
    else:
        readme = existing_readme.rstrip() + "\n\n" + card_text

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Quality card pushed to https://huggingface.co/datasets/{repo_id}")


# ── CLI entry point ───────────────────────────────────────────────────────────


def run_card(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra card",
        description=(
            "Generate a HuggingFace dataset quality card from Calibra diagnostics. "
            "Output is Markdown that can be embedded in a dataset README."
        ),
    )
    p.add_argument("path", help="Dataset path or HuggingFace Hub ID")
    p.add_argument(
        "--policy",
        "-p",
        metavar="FAMILY",
        default=None,
        help="Target policy family (e.g. 'diffusion', 'act', 'gr00t')",
    )
    p.add_argument(
        "--format",
        "-f",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force format adapter",
    )
    p.add_argument(
        "--out",
        "-o",
        metavar="PATH",
        help="Write card to a file instead of stdout",
    )
    p.add_argument(
        "--push",
        action="store_true",
        help="Push the quality card to the dataset's HuggingFace Hub README. "
        "Requires huggingface_hub and HF_TOKEN environment variable.",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output full diagnostic report as JSON",
    )
    args = p.parse_args(argv)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://") :]

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Profiling {dataset_path!r} ...")

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    try:
        report = Pipeline().analyze_path(
            dataset_path,
            policy_family=args.policy,
            reader=reader,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {report.n_episodes} episodes  ·  {report.n_samples:,} steps")

    if args.json:
        print(report.model_dump_json(indent=2))
        sys.exit(0)

    card = generate_card(report, policy_family=args.policy)

    if args.out:
        from pathlib import Path

        Path(args.out).write_text(card)
        log(f"  Card written to {args.out}")
    else:
        print(card)

    if args.push:
        push_to_hub(card, dataset_path)

    n_crit = len(report.flags_at_level(RiskLevel.CRITICAL))
    sys.exit(1 if n_crit > 0 else 0)
