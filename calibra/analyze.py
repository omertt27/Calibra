"""
calibra analyze — the single-command "is this trustworthy, how good is it,
what should I train on" report.

Cold-outreach demo path: an engineer at a design-partner candidate should be
able to point this at their dataset and see, in one screen, the same story
that today takes three separate commands (`calibra <path>` for flags,
`calibra integrity` for trust, `calibra prune` for a coreset) to assemble by
hand. Nothing here is a new metric — it's the existing analyzers, the
existing Calibra Score, and the existing regime-adaptive coreset selector,
composed into one narrative and one report object.

The training-set recommendation is a heuristic starting point (roughly
1 - measured state redundancy, clamped), not a substitute for the
design-partner three-condition retention-sweep protocol — the report says so
explicitly, and `calibra experiment` / `calibra case-study` are the commands
that turn it into a validated number.

    calibra analyze /data/robot_demos
    calibra analyze lerobot/pusht --format lerobot --policy act
    calibra analyze /data/robot_demos --keep 0.4 --export coreset_index.json
    calibra analyze /data/robot_demos --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional

from calibra.analyzers.base import Analyzer
from calibra.analyzers.blur import BlurAnalyzer
from calibra.analyzers.calibration_drift import CalibrationDriftAnalyzer
from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
from calibra.analyzers.coverage import CoverageEntropyAnalyzer
from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer
from calibra.analyzers.force_torque import ForceTorqueContactAnalyzer
from calibra.analyzers.influence import InfluenceAnalyzer
from calibra.analyzers.latent_dynamics import LatentDynamicsAnalyzer
from calibra.analyzers.phase_balance import PhaseBalanceAnalyzer
from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
from calibra.analyzers.ssl_embed import SSLTrajectoryEmbedderAnalyzer
from calibra.analyzers.task_structure import TaskStructureAnalyzer
from calibra.analyzers.temporal import TemporalAnalyzer
from calibra.analyzers.transition_dynamics import TransitionDynamicsAnalyzer
from calibra.integrity import _integrity_score, _is_integrity_flag
from calibra.pipeline import Pipeline
from calibra.pruning import CoresetSelector, PruningResult
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport, RiskFlag, RiskLevel
from calibra.score import compute_score
from calibra.strategy import _REGIME_LABELS, RegimeDiagnosis, diagnose_regime

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN = "─" * _WIDTH

_MIN_KEEP_FRACTION = 0.15
_MAX_KEEP_FRACTION = 0.85
_DEFAULT_KEEP_FRACTION = 0.5

# Category label -> metric names that roll up into it, mirroring the trust
# questions `calibra integrity` asks. A category with no flags at all means
# its analyzer(s) were skipped (missing capability), not "passed".
_INTEGRITY_CATEGORIES: dict[str, frozenset[str]] = {
    "Timestamps & sync": frozenset(
        {
            "timestamp_jitter_cv",
            "timestamp_dropout_rate",
            "action_obs_misalignment",
            "camera_physics_drift",
        }
    ),
    "Episode structure": frozenset({"action_dropout_rate", "short_episode_fraction"}),
    "Camera feed": frozenset(
        {"duplicate_frame_rate", "camera_freeze_events", "blurry_episode_fraction"}
    ),
    "Motion & control": frozenset(
        {"ldlj", "jerk_spike_rate", "velocity_discontinuity_rate", "joint_offset_max_abs"}
    ),
}


def _combined_analyzers() -> list[Analyzer]:
    """Union of the default quality/coverage analyzer set and the
    integrity-only ones (blur/camera-freeze/duplicate-frame/calibration-drift)
    that `calibra integrity` runs separately — one pass, one report, so every
    number in `calibra analyze` traces back to the same run."""
    return [
        TemporalAnalyzer(),
        ControlSmoothnessAnalyzer(),
        TaskStructureAnalyzer(),
        DuplicateFrameAnalyzer(),
        CameraFreezeAnalyzer(),
        BlurAnalyzer(),
        CalibrationDriftAnalyzer(),
        CoverageEntropyAnalyzer(),
        PhaseBalanceAnalyzer(),
        InfluenceAnalyzer(),
        TransitionDynamicsAnalyzer(),
        LatentDynamicsAnalyzer(),
        SSLTrajectoryEmbedderAnalyzer(),
        ForceTorqueContactAnalyzer(),
    ]


def _category_for(metric: str) -> Optional[str]:
    if metric.startswith("camera_lag_std["):
        return "Timestamps & sync"
    for category, metrics in _INTEGRITY_CATEGORIES.items():
        if metric in metrics:
            return category
    return None


def _integrity_by_category(report: DiagnosticReport) -> dict[str, list[RiskFlag]]:
    by_category: dict[str, list[RiskFlag]] = {name: [] for name in _INTEGRITY_CATEGORIES}
    for flag in report.flags:
        if not _is_integrity_flag(flag):
            continue
        category = _category_for(flag.metric)
        if category is not None:
            by_category[category].append(flag)
    return by_category


def _worst_level(flags: list[RiskFlag]) -> Optional[RiskLevel]:
    if not flags:
        return None
    if any(f.level == RiskLevel.CRITICAL for f in flags):
        return RiskLevel.CRITICAL
    if any(f.level == RiskLevel.WARNING for f in flags):
        return RiskLevel.WARNING
    return RiskLevel.OK


def _recommend_keep_fraction(redundancy: Optional[float]) -> float:
    """
    Heuristic starting point: keep roughly (1 - measured state redundancy),
    clamped to a sane range. A starting point for the design-partner
    retention-curve protocol, not a substitute for it.
    """
    if redundancy is None:
        return _DEFAULT_KEEP_FRACTION
    return round(min(_MAX_KEEP_FRACTION, max(_MIN_KEEP_FRACTION, 1.0 - redundancy)), 2)


@dataclass
class AnalysisResult:
    report: DiagnosticReport
    score_result: dict
    integrity_by_category: dict[str, list[RiskFlag]]
    integrity_score: int
    integrity_status: str
    redundancy: Optional[float]
    keep_fraction: float
    regime_diagnosis: Optional[RegimeDiagnosis]
    prune_result: Optional[PruningResult]
    n_tasks: int
    action_dim: Optional[int]


def run_analysis(
    batch: EpisodeBatch,
    policy_family: Optional[str] = None,
    keep_fraction: Optional[float] = None,
    cache=None,
) -> AnalysisResult:
    """Pure function: batch in, full analysis out. No argv, no printing —
    the CLI wrapper (run_analyze) and any future caller (web UI, notebook)
    both go through this."""
    pipeline = Pipeline(analyzers=_combined_analyzers())
    report = pipeline.run(batch, policy_family=policy_family, cache=cache)

    score_result = compute_score(report)

    by_category = _integrity_by_category(report)
    all_integrity_flags = [f for flags in by_category.values() for f in flags]
    integrity_score, integrity_status = _integrity_score(all_integrity_flags)

    diagnosis: Optional[RegimeDiagnosis] = None
    prune_result: Optional[PruningResult] = None
    redundancy: Optional[float] = None
    resolved_keep_fraction = keep_fraction if keep_fraction is not None else _DEFAULT_KEEP_FRACTION

    if batch.n_episodes >= 5:
        diagnosis = diagnose_regime(report)
        redundancy = diagnosis.evidence.get("state_redundancy")
        resolved_keep_fraction = (
            keep_fraction if keep_fraction is not None else _recommend_keep_fraction(redundancy)
        )
        try:
            selector = CoresetSelector(
                keep_fraction=resolved_keep_fraction, **diagnosis.recommended_config
            )
            prune_result = selector.select(batch, report)
        except Exception:
            prune_result = None

    n_tasks = len(
        {ep.metadata.task_description for ep in batch.episodes if ep.metadata.task_description}
    )
    action_dim = batch.episodes[0].action_dim if batch.episodes else None

    return AnalysisResult(
        report=report,
        score_result=score_result,
        integrity_by_category=by_category,
        integrity_score=integrity_score,
        integrity_status=integrity_status,
        redundancy=redundancy,
        keep_fraction=resolved_keep_fraction,
        regime_diagnosis=diagnosis,
        prune_result=prune_result,
        n_tasks=n_tasks,
        action_dim=action_dim,
    )


# ── rendering ────────────────────────────────────────────────────────────────

_LEVEL_ICON = {RiskLevel.CRITICAL: "❌", RiskLevel.WARNING: "⚠️ ", RiskLevel.OK: "✅"}


def render_analysis(result: AnalysisResult) -> str:
    r = result.report
    lines = [_THICK, "  CALIBRA ANALYSIS", _THICK, "", "  Dataset"]
    lines.append(f"    Name       : {r.dataset_name}")
    lines.append(f"    Episodes   : {r.n_episodes:,}")
    lines.append(f"    Frames     : {r.n_samples:,}")
    lines.append(f"    Format     : {r.format}")
    if result.n_tasks:
        lines.append(f"    Tasks      : {result.n_tasks} distinct")
    if result.action_dim is not None:
        lines.append(f"    Action dim : {result.action_dim}")
    if r.policy_family:
        lines.append(f"    Policy     : {r.policy_family.upper()}")

    lines.append("")
    lines.append(_THIN)
    lines.append("  Integrity")
    for category, flags in result.integrity_by_category.items():
        level = _worst_level(flags)
        icon = _LEVEL_ICON[level] if level is not None else "·  "
        suffix = "" if level is not None else "  (not evaluated)"
        lines.append(f"    {icon} {category}{suffix}")
    lines.append(f"    Integrity score: {result.integrity_score}/100 — {result.integrity_status}")

    lines.append(_THIN)
    q = result.score_result
    cov = q["dimensions"]["coverage_diversity"]
    cov_pct = cov["score"] / cov["max"] * 100 if cov["max"] else 0.0
    lines.append(f"  Quality (Calibra Score)   {q['total_score']:5.1f} / 100   —  {q['category']}")
    lines.append(f"  Coverage / diversity      {cov_pct:5.1f} / 100")
    if result.redundancy is not None:
        lines.append(
            f"  Redundancy (estimated)    {result.redundancy:5.1%}  of state-space occupies duplicate regions"
        )
    elif r.n_episodes < 5:
        lines.append(
            "  Redundancy (estimated)    n/a — dataset has < 5 episodes (need >= 5 to diagnose)"
        )
    else:
        lines.append("  Redundancy (estimated)    n/a — requires proprioceptive/state observations")

    lines.append(_THIN)
    lines.append("  RECOMMENDATION")
    lines.append("")
    pr = result.prune_result
    if pr is None:
        lines.append("    Not enough episodes to compute a coreset recommendation (need >= 5).")
    else:
        if result.regime_diagnosis is not None:
            lines.append(
                f"    Regime             : {_REGIME_LABELS[result.regime_diagnosis.regime]}"
            )
        lines.append(f"    Training set       : {pr.n_kept:,} / {pr.n_original:,} episodes")
        lines.append(f"    Expected retention : {pr.keep_fraction_actual:.0%}")
        lines.append("")
        lines.append("    Reasons:")
        if pr.n_quality_failures:
            lines.append(
                f"      • removes {pr.n_quality_failures:,} corrupted/low-quality episodes"
            )
        if pr.n_diversity_pruned:
            lines.append(
                f"      • removes {pr.n_diversity_pruned:,} redundant episodes (diversity selection)"
            )
        lines.append("      • preserves behavioral coverage via greedy max-coverage selection")
        lines.append("")
        lines.append(
            "    This is a heuristic starting point (~1 - measured redundancy), not a\n"
            "    validated retention curve. Run the design-partner protocol\n"
            "    (`calibra experiment` + `calibra case-study`) before committing a\n"
            "    production training run to this number."
        )
        lines.append("")
        lines.append("    Export this coreset: calibra analyze <path> --export coreset_index.json")

    lines.append(_THICK)
    if r.calibra_version:
        lines.append(
            f"  Calibra v{r.calibra_version}  ·  config {r.config_hash}  ·  generated {r.generated_at}"
        )

    return "\n".join(lines)


def _to_json(result: AnalysisResult) -> dict:
    r = result.report

    def flags_json(flags: list[RiskFlag]) -> list[dict]:
        return [
            {
                "metric": f.metric,
                "level": f.level.value,
                "observed": f.observed.value,
                "interpretation": f.interpretation,
                "implication": f.implication,
            }
            for f in flags
        ]

    q = result.score_result
    cov = q["dimensions"]["coverage_diversity"]

    payload: dict = {
        "dataset_name": r.dataset_name,
        "source_path": r.source_path,
        "format": r.format,
        "n_episodes": r.n_episodes,
        "n_samples": r.n_samples,
        "n_tasks": result.n_tasks,
        "action_dim": result.action_dim,
        "policy_family": r.policy_family,
        "integrity": {
            "score": result.integrity_score,
            "status": result.integrity_status,
            "categories": {
                cat: {
                    "level": (lvl.value if (lvl := _worst_level(flags)) is not None else None),
                    "flags": flags_json(flags),
                }
                for cat, flags in result.integrity_by_category.items()
            },
        },
        "quality": {
            "total_score": q["total_score"],
            "category": q["category"],
            "dimensions": q["dimensions"],
        },
        "coverage_pct": round(cov["score"] / cov["max"] * 100, 1) if cov["max"] else None,
        "redundancy": result.redundancy,
        "recommendation": {
            "keep_fraction_requested": result.keep_fraction,
            "regime": result.regime_diagnosis.regime.value if result.regime_diagnosis else None,
            "prune_result": result.prune_result.to_dict() if result.prune_result else None,
            "note": (
                "Heuristic starting point (~1 - measured redundancy), not a validated "
                "retention curve. Validate with the design-partner protocol before a "
                "production training run."
            ),
        },
        "calibra_version": r.calibra_version,
        "config_hash": r.config_hash,
        "generated_at": r.generated_at,
    }
    return payload


# ── CLI entry point ───────────────────────────────────────────────────────────


def run_analyze(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra analyze",
        description=(
            "One-command dataset report: integrity (trust), Calibra Score (quality/"
            "coverage), estimated redundancy, and a training-set recommendation backed "
            "by the same coreset selector `calibra prune` uses."
        ),
    )
    p.add_argument("path", help="Path or Hub ID of the dataset to analyze")
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
        help="Target policy family for conditioned hints (e.g. 'diffusion', 'act')",
    )
    p.add_argument(
        "--keep",
        "-k",
        type=float,
        default=None,
        metavar="FRACTION",
        help="Override the automatic training-set retention recommendation (0-1)",
    )
    p.add_argument(
        "--export",
        metavar="PATH",
        default=None,
        help="Write the recommended coreset index to PATH (same format as `calibra prune --out`)",
    )
    p.add_argument("--json", "-j", action="store_true", help="Print the full result as JSON")
    p.add_argument(
        "--cache-dir",
        metavar="DIR",
        default=None,
        help="Cache directory for incremental analysis (see `calibra --cache-dir`)",
    )
    args = p.parse_args(argv)

    if args.keep is not None and not (0.0 < args.keep <= 1.0):
        print("error: --keep must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://") :]

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {dataset_path!r} ...")
    try:
        from calibra.ingestion.registry import load

        batch = load(dataset_path, reader=reader)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {batch.n_episodes} episodes  ·  {batch.n_samples} steps")
    log("Running full analysis (integrity + quality + coverage + recommendation) ...")

    cache = None
    if args.cache_dir:
        from calibra.cache import AuditCache

        cache = AuditCache(args.cache_dir)

    try:
        result = run_analysis(
            batch, policy_family=args.policy, keep_fraction=args.keep, cache=cache
        )
    except Exception as exc:
        print(f"error running analysis: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.export:
        if result.prune_result is None:
            print("error: cannot export — no coreset recommendation was computed", file=sys.stderr)
            sys.exit(1)
        with open(args.export, "w") as f:
            json.dump(result.prune_result.to_dict(), f, indent=2)
        log(f"Coreset index written to {args.export}")

    if args.json:
        print(json.dumps(_to_json(result), indent=2))
    else:
        print(render_analysis(result))
