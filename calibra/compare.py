"""
calibra compare — compare a local dataset against a named reference profile.

Usage (via CLI):
    calibra compare /path/to/dataset pusht
    calibra compare /path/to/dataset aloha
    calibra compare lerobot/my_dataset pusht --format lerobot

The reference name is matched against files in calibra/references/. Partial
matches work: "pusht" matches "pusht_velocity_command.json".
"""
from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from calibra.schema.report import DiagnosticReport
import calibra.claims as _claims

_REFS_DIR = Path(__file__).parent / "references"
_WIDTH = 56


# ── reference loading ─────────────────────────────────────────────────────────

def find_reference(name: str) -> Path:
    candidates = sorted(_REFS_DIR.glob("*.json"))
    for c in candidates:
        if c.stem == name:
            return c
    hits = [c for c in candidates if name.lower() in c.stem.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(
            f"Ambiguous reference {name!r}. Matches: {[c.stem for c in hits]}\n"
            "Use a more specific name."
        )
    available = [c.stem for c in candidates]
    raise ValueError(
        f"No reference profile found for {name!r}.\n"
        f"Available: {available}\n"
        "Add one with: python scripts/profile_pusht.py --dataset <id> "
        "--out calibra/references/<name>.json"
    )


def load_reference(name: str) -> dict:
    path = find_reference(name)
    with open(path) as f:
        return json.load(f)


# ── metric extraction ─────────────────────────────────────────────────────────

def _raw(report: DiagnosticReport, analyzer: str) -> dict:
    for r in report.analyzer_results:
        if r.analyzer_name == analyzer:
            return r.raw_metrics
    return {}


def metrics_from_report(report: DiagnosticReport) -> dict[str, Optional[float]]:
    t = _raw(report, "temporal_stability")
    s = _raw(report, "control_smoothness")
    c = _raw(report, "coverage_entropy")
    return {
        "jitter_cv":      t.get("jitter",             {}).get("mean_cv"),
        "dropout_rate":   t.get("dropout",            {}).get("mean_dropout_fraction"),
        "ldlj":           s.get("ldlj",               {}).get("mean_ldlj"),
        "spike_rate":     s.get("jerk_spikes",        {}).get("mean_spike_fraction"),
        "vel_disc_rate":  s.get("vel_discontinuities",{}).get("mean_disc_fraction"),
        "action_entropy": c.get("action_entropy",     {}).get("entropy_bits_per_dim"),
    }


def metrics_from_reference(ref: dict) -> dict[str, Optional[float]]:
    agg = ref.get("aggregate_metrics", {})
    t = agg.get("temporal_stability", {})
    s = agg.get("control_smoothness", {})
    c = agg.get("coverage_entropy", {})
    return {
        "jitter_cv":      t.get("jitter.mean_cv"),
        "dropout_rate":   t.get("dropout.mean_dropout_fraction"),
        "ldlj":           s.get("ldlj.mean_ldlj"),
        "spike_rate":     s.get("jerk_spikes.mean_spike_fraction"),
        "vel_disc_rate":  s.get("vel_discontinuities.mean_disc_fraction"),
        "action_entropy": c.get("action_entropy.entropy_bits_per_dim"),
    }


# ── interpretation rules ──────────────────────────────────────────────────────

def _interp_vel_disc(
    yours: float, ref: float, ref_mode: str, ref_label: str
) -> tuple[str, str]:
    """(interpretation_text, confidence_label)"""
    delta = yours - ref
    rel = abs(delta) / max(abs(ref), 1e-9)

    if ref_mode == "velocity":
        if rel < 0.30:
            return (
                f"Similar to {ref_label}. Within the expected range for "
                "velocity-command teleoperation.",
                "HIGH",
            )
        elif delta > 0:
            return (
                f"Rougher than {ref_label}. Investigate abrupt teleop "
                "corrections, command noise, or timestamp misalignment.",
                "HIGH",
            )
        else:
            return (
                f"Smoother than {ref_label}. Unusually clean for "
                "velocity-command data — verify the control mode.",
                "HIGH",
            )
    elif ref_mode == "position":
        if yours < 0.04:
            return (
                f"Similar to {ref_label}. Expected for position-command "
                "manipulation.",
                "HIGH",
            )
        elif yours > 0.10:
            return (
                f"Significantly rougher than {ref_label}. "
                "If using position commands: investigate control noise or "
                "abrupt operator corrections. "
                "If using velocity commands: compare to pusht instead "
                "(expected ~16%).",
                "HIGH",
            )
        else:
            return (
                f"Somewhat rougher than {ref_label}. "
                "Warrants investigation if your dataset uses position commands.",
                "HIGH",
            )
    else:
        if rel < 0.30:
            return (f"Similar to {ref_label} (Δ {delta:+.1%}).", "MODERATE")
        elif delta > 0:
            return (
                f"Rougher than {ref_label} (Δ {delta:+.1%}). "
                "Investigate teleop quality.",
                "MODERATE",
            )
        else:
            return (
                f"Smoother than {ref_label} (Δ {delta:+.1%}).",
                "MODERATE",
            )


def _interp_spike_rate(
    yours: float, ref: float, ref_mode: str, ref_label: str
) -> tuple[str, str]:
    delta = yours - ref
    rel = abs(delta) / max(abs(ref), 1e-9)
    if rel < 0.40:
        return (f"Similar to {ref_label}. No anomalous jerk spikes.", "MODERATE")
    elif delta > 0:
        return (
            f"Higher spike rate than {ref_label}. "
            "Check for dropped frames, bad episode boundaries, or "
            "bimodal speed profiles (fast approach + slow manipulation).",
            "MODERATE",
        )
    else:
        return (f"Cleaner than {ref_label}. Low jerk-spike rate.", "MODERATE")


def _interp_ldlj(
    yours: float, ref: float, ref_mode: str, ref_label: str,
    your_action_dim: Optional[int], ref_action_dim: Optional[int],
) -> tuple[str, str]:
    delta = yours - ref
    mode_mismatch = ref_mode not in ("unknown", "") and ref_mode != "unknown"
    dim_ratio = (
        (your_action_dim or 1) / max(ref_action_dim or 1, 1)
        if your_action_dim and ref_action_dim else 1.0
    )
    warn = "  ⚠  LDLJ is unreliable for cross-mode or cross-frequency comparison.\n  "
    if abs(dim_ratio - 1.0) > 0.5 or mode_mismatch:
        return (
            warn + "Both datasets are in the normal range for real teleoperation.\n"
            "  Interpret within-type only. See interpretations/ldlj.md",
            "LOW (cross-dataset)",
        )
    rel = abs(delta) / max(abs(ref), 1e-9)
    if rel < 0.15:
        return (f"Similar smoothness to {ref_label}.", "LOW (within-type)")
    elif delta > 0:  # less negative = smoother
        return (f"Smoother than {ref_label}.", "LOW (within-type)")
    else:
        return (
            f"Rougher than {ref_label}. "
            "Consider action smoothing (e.g. Savitzky-Golay) before training.",
            "LOW (within-type)",
        )


def _interp_temporal(
    yours: float, ref: float, key: str, ref_is_sim: bool
) -> tuple[str, str]:
    if ref_is_sim:
        return (
            "Reference is from a simulated dataset (machine-precision timestamps). "
            "Not informative until profiled against real hardware.",
            "NOT VALIDATED",
        )
    delta = yours - ref
    if key == "jitter_cv":
        if yours < 0.05:
            return ("Low jitter. Control loop is running at a consistent rate.", "MODERATE")
        elif yours < 0.15:
            return (
                "Moderate jitter. Acceptable for USB-camera + ROS recording stacks.",
                "MODERATE",
            )
        else:
            return (
                "High jitter. Consider resampling to a uniform frequency "
                "before training time-series policies.",
                "MODERATE",
            )
    else:  # dropout
        if yours < 0.01:
            return ("Very low dropout. Clean frame delivery.", "MODERATE")
        elif yours < 0.05:
            return (
                "Moderate dropout. Filter or interpolate affected episodes "
                "before training.",
                "MODERATE",
            )
        else:
            return (
                "High dropout. Significant frame loss — likely hardware or "
                "recording pipeline issue.",
                "MODERATE",
            )


def _interp_entropy(
    yours: float, ref: float, ref_label: str
) -> tuple[str, str]:
    delta = yours - ref
    if yours > 4.5:
        return (
            f"Healthy action-space coverage. Similar to {ref_label}.",
            "LOW",
        )
    elif yours > 3.0:
        return (
            "Moderate coverage. Some risk of limited trajectory diversity. "
            "Check whether demonstrations sample the full operating range.",
            "LOW",
        )
    else:
        return (
            "Low entropy — possible mode collapse or limited trajectory variation. "
            "Policy may generalise poorly to out-of-distribution states.",
            "LOW",
        )


# ── rendering ─────────────────────────────────────────────────────────────────

def _pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if v is not None else "n/a"

def _f2(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "n/a"

def _sci(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2e}" if abs(v) < 0.001 else f"{v:.4f}"

def _delta_arrow(delta: Optional[float]) -> str:
    if delta is None:
        return ""
    return "  ▲" if delta > 0 else "  ▼"

def _section(
    title: str,
    yours_str: str,
    ref_str: str,
    ref_name: str,
    delta_str: str,
    arrow: str,
    interpretation: str,
    evidence: str,
) -> str:
    lines = [
        title,
        f"  Yours:  {yours_str}",
        f"  {ref_name:<6}  {ref_str}",
        f"  Delta:  {delta_str}{arrow}",
        "",
    ]
    interp_wrapped = textwrap.fill(
        interpretation, width=_WIDTH, initial_indent="  ", subsequent_indent="  "
    )
    lines.append(interp_wrapped)
    if evidence:
        lines.append(f"\n  {evidence}")
    return "\n".join(lines) + "\n"


def render_comparison(
    your_path: str,
    your_metrics: dict[str, Optional[float]],
    your_n_episodes: int,
    your_action_dim: Optional[int],
    ref_data: dict,
    ref_metrics: dict[str, Optional[float]],
    ref_name: str,
) -> str:
    meta = ref_data.get("meta", {})
    ref_label = meta.get("dataset", ref_name)
    ref_mode = meta.get("control_mode", "unknown")
    ref_n_eps = meta.get("n_episodes", "?")
    ref_action_dim = meta.get("action_dim")
    ref_is_sim = _ref_is_sim(ref_metrics)

    mode_tag = f"{ref_mode}-command" if ref_mode in ("velocity", "position") else "unknown control mode"
    dim_tag = f"{ref_action_dim}D" if ref_action_dim else ""
    header_ref = f"{ref_label}  ({mode_tag} · {dim_tag} · {ref_n_eps} episodes)"

    divider = "─" * _WIDTH
    thick   = "━" * _WIDTH
    lines = [
        thick,
        f"calibra compare — {Path(your_path).name}  vs.  {ref_name}",
        thick,
        "",
        f"Reference: {header_ref}",
        f"Yours:     {Path(your_path).name}  ({your_n_episodes} episodes)",
        "",
        divider,
    ]

    # 1. Velocity discontinuity
    y, r = your_metrics["vel_disc_rate"], ref_metrics["vel_disc_rate"]
    delta = (y - r) if y is not None and r is not None else None
    interp, _ = _interp_vel_disc(y or 0, r or 0, ref_mode, ref_name) if (y is not None and r is not None) else ("Could not compute.", "")
    lines.append(_section(
        "VELOCITY DISCONTINUITY RATE",
        _pct(y), _pct(r), ref_name, _pct(delta), _delta_arrow(delta),
        interp, _claims.evidence_line("vel_disc_rate", ref_mode),
    ))
    lines.append(divider)

    # 2. Jerk spike rate
    y, r = your_metrics["spike_rate"], ref_metrics["spike_rate"]
    delta = (y - r) if y is not None and r is not None else None
    interp, _ = _interp_spike_rate(y or 0, r or 0, ref_mode, ref_name) if (y is not None and r is not None) else ("Could not compute.", "")
    lines.append(_section(
        "JERK SPIKE RATE",
        _pct(y), _pct(r), ref_name, _pct(delta), _delta_arrow(delta),
        interp, _claims.evidence_line("spike_rate", ref_mode),
    ))
    lines.append(divider)

    # 3. LDLJ
    y, r = your_metrics["ldlj"], ref_metrics["ldlj"]
    delta = (y - r) if y is not None and r is not None else None
    if y is not None and r is not None:
        interp, _ = _interp_ldlj(y, r, ref_mode, ref_name, your_action_dim, ref_action_dim)
    else:
        interp = "Could not compute."
    delta_str = f"{delta:+.2f}" if delta is not None else "n/a"
    arrow = ("  ▲ (smoother)" if delta > 0 else "  ▼ (rougher)") if delta is not None else ""
    lines.append(_section(
        "LDLJ",
        _f2(y), _f2(r), ref_name, delta_str, arrow,
        interp, _claims.evidence_line("ldlj", "any"),
    ))
    lines.append(divider)

    # 4. Timestamp jitter
    y, r = your_metrics["jitter_cv"], ref_metrics["jitter_cv"]
    delta = (y - r) if y is not None and r is not None else None
    interp, _ = _interp_temporal(y or 0, r or 0, "jitter_cv", ref_is_sim) if y is not None else ("Could not compute.", "")
    ref_str = (_sci(r) + "  (sim)") if ref_is_sim else _sci(r)
    hw_class = "any" if ref_is_sim else "hardware"
    lines.append(_section(
        "TIMESTAMP JITTER CV",
        _sci(y), ref_str, ref_name, _sci(delta), _delta_arrow(delta),
        interp, _claims.evidence_line("jitter_cv", hw_class),
    ))
    lines.append(divider)

    # 5. Timestamp dropout
    y, r = your_metrics["dropout_rate"], ref_metrics["dropout_rate"]
    delta = (y - r) if y is not None and r is not None else None
    interp, _ = _interp_temporal(y or 0, r or 0, "dropout_rate", ref_is_sim) if y is not None else ("Could not compute.", "")
    ref_str = (_pct(r) + "  (sim)") if ref_is_sim else _pct(r)
    lines.append(_section(
        "TIMESTAMP DROPOUT RATE",
        _pct(y), ref_str, ref_name, _pct(delta), _delta_arrow(delta),
        interp, _claims.evidence_line("dropout_rate", "any"),
    ))
    lines.append(divider)

    # 6. Action entropy
    y, r = your_metrics["action_entropy"], ref_metrics["action_entropy"]
    delta = (y - r) if y is not None and r is not None else None
    interp, _ = _interp_entropy(y or 0, r or 0, ref_name) if y is not None else ("Could not compute.", "")
    lines.append(_section(
        "ACTION ENTROPY",
        f"{_f2(y)} bits/dim", f"{_f2(r)} bits/dim", ref_name,
        f"{delta:+.2f} bits/dim" if delta is not None else "n/a",
        _delta_arrow(delta),
        interp, _claims.evidence_line("action_entropy", ref_mode),
    ))
    lines.append(thick)

    return "\n".join(lines)


def _ref_is_sim(ref_metrics: dict) -> bool:
    """Heuristic: reference is from sim if jitter CV is near machine precision."""
    jitter = ref_metrics.get("jitter_cv")
    return jitter is not None and jitter < 1e-3


# ── entry point ───────────────────────────────────────────────────────────────

def run_compare(argv: list[str]) -> None:
    import argparse
    from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
    from calibra.analyzers.coverage import CoverageEntropyAnalyzer
    from calibra.analyzers.task_structure import TaskStructureAnalyzer
    from calibra.analyzers.temporal import TemporalAnalyzer
    from calibra.pipeline import Pipeline

    p = argparse.ArgumentParser(
        prog="calibra compare",
        description="Compare a dataset against a named reference profile.",
    )
    p.add_argument("path",      help="Path or Hub ID of dataset to analyse")
    p.add_argument("reference", help="Reference name (e.g. 'pusht', 'aloha')")
    p.add_argument("--format", "-f", metavar="FMT",
                   choices=["hdf5", "lerobot", "rlds", "mcap"],
                   help="Force a format adapter (default: auto-detect)")
    p.add_argument("--gripper-dims", metavar="DIMS", default=None,
                   help="Comma-separated gripper dimension indices to exclude "
                        "from smoothness metrics (e.g. '6,13'). "
                        "Use '' to disable gripper exclusion.")
    args = p.parse_args(argv)

    # resolve reader
    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    # resolve gripper dims
    gripper_dims: list[int] = [-1]   # default: last dim
    if args.gripper_dims is not None:
        raw = args.gripper_dims.strip()
        gripper_dims = [int(x) for x in raw.split(",") if x.strip()] if raw else []

    log = lambda msg: print(msg, file=sys.stderr, flush=True)

    # load reference
    try:
        ref_data = load_reference(args.reference)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    # run pipeline on user's dataset
    log(f"Loading {args.path!r} ...")
    try:
        pipeline = Pipeline(analyzers=[
            TemporalAnalyzer(),
            ControlSmoothnessAnalyzer(gripper_dims=gripper_dims),
            CoverageEntropyAnalyzer(),
            TaskStructureAnalyzer(),
        ])
        report: DiagnosticReport = pipeline.analyze_path(args.path, reader=reader)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    log(f"  {report.n_episodes} episodes  ·  {report.n_samples} steps")

    # extract metrics and render
    your_metrics = metrics_from_report(report)
    ref_metrics  = metrics_from_reference(ref_data)

    # infer action dim from first episode via report metadata
    your_action_dim = None
    for result in report.analyzer_results:
        raw = result.raw_metrics
        dim = raw.get("action_entropy", {}).get("action_dim")
        if dim is not None:
            your_action_dim = int(dim)
            break

    output = render_comparison(
        your_path=args.path,
        your_metrics=your_metrics,
        your_n_episodes=report.n_episodes,
        your_action_dim=your_action_dim,
        ref_data=ref_data,
        ref_metrics=ref_metrics,
        ref_name=args.reference,
    )
    print(output)
