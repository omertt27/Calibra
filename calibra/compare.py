"""
calibra compare — compare a local dataset against a named reference profile.

Usage (via CLI):
    calibra compare /path/to/dataset pusht
    calibra compare /path/to/dataset aloha
    calibra compare lerobot/my_dataset pusht --format lerobot
    calibra compare hf://lerobot/aloha_mobile_cabinet aloha

The reference name is matched against files in calibra/references/. Partial
matches work: "pusht" matches "pusht_velocity_command.json".

hf:// URIs are supported and stripped before passing to the ingestion layer.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import urllib.request
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
        "jitter_cv": t.get("jitter", {}).get("mean_cv"),
        "dropout_rate": t.get("dropout", {}).get("mean_dropout_fraction"),
        "ldlj": s.get("ldlj", {}).get("mean_ldlj"),
        "spike_rate": s.get("jerk_spikes", {}).get("mean_spike_fraction"),
        "vel_disc_rate": s.get("vel_discontinuities", {}).get("mean_disc_fraction"),
        "action_entropy": c.get("action_entropy", {}).get("entropy_bits_per_dim"),
    }


def metrics_from_reference(ref: dict) -> dict[str, Optional[float]]:
    agg = ref.get("aggregate_metrics", {})
    t = agg.get("temporal_stability", {})
    s = agg.get("control_smoothness", {})
    c = agg.get("coverage_entropy", {})
    return {
        "jitter_cv": t.get("jitter.mean_cv"),
        "dropout_rate": t.get("dropout.mean_dropout_fraction"),
        "ldlj": s.get("ldlj.mean_ldlj"),
        "spike_rate": s.get("jerk_spikes.mean_spike_fraction"),
        "vel_disc_rate": s.get("vel_discontinuities.mean_disc_fraction"),
        "action_entropy": c.get("action_entropy.entropy_bits_per_dim"),
    }


# ── interpretation rules ──────────────────────────────────────────────────────


def _interp_vel_disc(yours: float, ref: float, ref_mode: str, ref_label: str) -> tuple[str, str]:
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
                f"Similar to {ref_label}. Expected for position-command manipulation.",
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
                f"Rougher than {ref_label} (Δ {delta:+.1%}). Investigate teleop quality.",
                "MODERATE",
            )
        else:
            return (
                f"Smoother than {ref_label} (Δ {delta:+.1%}).",
                "MODERATE",
            )


def _interp_spike_rate(
    yours: float,
    ref: float,
    ref_mode: str,
    ref_label: str,
    yours_is_scripted: bool = False,
    ref_is_scripted: bool = False,
) -> tuple[str, str]:
    delta = yours - ref
    rel = abs(delta) / max(abs(ref), 1e-9)

    if yours_is_scripted and not ref_is_scripted:
        return (
            f"Yours is a scripted/planner dataset (spike_rate {yours:.1%}) — "
            f"significantly higher than human teleop reference {ref_label} "
            f"({ref:.1%}). Planner waypoint transitions produce jerk spikes "
            "at every target switch. This is expected, not a recording defect. "
            "The prune CLI auto-raises --max-spike-rate to 0.30 for this data.",
            "HIGH (scripted vs human)",
        )
    if not yours_is_scripted and ref_is_scripted:
        return (
            f"Reference {ref_label} is a scripted/planner dataset "
            f"({ref:.1%} spike rate). Your dataset ({yours:.1%}) shows the "
            "lower spike rate typical of human teleoperation. "
            "The comparison is valid but these are structurally different datasets.",
            "HIGH (human vs scripted)",
        )

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
    yours: float,
    ref: float,
    ref_mode: str,
    ref_label: str,
    your_action_dim: Optional[int],
    ref_action_dim: Optional[int],
) -> tuple[str, str]:
    delta = yours - ref
    mode_mismatch = ref_mode not in ("unknown", "") and ref_mode != "unknown"
    dim_ratio = (
        (your_action_dim or 1) / max(ref_action_dim or 1, 1)
        if your_action_dim and ref_action_dim
        else 1.0
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


def _interp_temporal(yours: float, ref: float, key: str, ref_is_sim: bool) -> tuple[str, str]:
    if ref_is_sim:
        return (
            "Reference is from a simulated dataset (machine-precision timestamps). "
            "Not informative until profiled against real hardware.",
            "NOT VALIDATED",
        )
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
                "Moderate dropout. Filter or interpolate affected episodes before training.",
                "MODERATE",
            )
        else:
            return (
                "High dropout. Significant frame loss — likely hardware or "
                "recording pipeline issue.",
                "MODERATE",
            )


def _interp_entropy(yours: float, ref: float, ref_label: str) -> tuple[str, str]:
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
    confidence: str = "",
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
    conf_str = f"Confidence: {confidence}" if confidence else ""
    ev_parts = [p for p in (conf_str, evidence) if p]
    if ev_parts:
        lines.append(f"\n  {' · '.join(ev_parts)}")
    return "\n".join(lines) + "\n"


def render_comparison(
    your_path: str,
    your_metrics: dict[str, Optional[float]],
    your_n_episodes: int,
    your_action_dim: Optional[int],
    ref_data: dict,
    ref_metrics: dict[str, Optional[float]],
    ref_name: str,
    outlier_episodes: Optional[list] = None,
    yours_is_scripted: bool = False,
    ref_scripted: bool = False,
) -> str:
    meta = ref_data.get("meta", {})
    ref_label = meta.get("dataset", ref_name)
    ref_mode = meta.get("control_mode", "unknown")
    ref_n_eps = meta.get("n_episodes", "?")
    ref_action_dim = meta.get("action_dim")
    ref_is_sim = _ref_is_sim(ref_metrics)

    mode_tag = (
        f"{ref_mode}-command" if ref_mode in ("velocity", "position") else "unknown control mode"
    )
    dim_tag = f"{ref_action_dim}D" if ref_action_dim else ""
    header_ref = f"{ref_label}  ({mode_tag} · {dim_tag} · {ref_n_eps} episodes)"

    divider = "─" * _WIDTH
    thick = "━" * _WIDTH
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

    # Collection-method mismatch banner
    if yours_is_scripted != ref_scripted:
        warn_thick = "▓" * _WIDTH
        if yours_is_scripted and not ref_scripted:
            mismatch_lines = [
                warn_thick,
                "  ⚠  COLLECTION METHOD MISMATCH",
                "  Your dataset:  scripted/planner demonstrations",
                f"  Reference:     human teleoperation  ({ref_name})",
                "",
                "  Scripted datasets have spike_rate ~20–25% and vel_disc_rate",
                "  <1.5%. Human teleoperation has spike_rate <1% and higher",
                "  vel_disc_rate. The spike-rate comparison below is expected",
                "  to show a large gap — this is structural, not a defect.",
                warn_thick,
                "",
            ]
        else:
            mismatch_lines = [
                warn_thick,
                "  ⚠  COLLECTION METHOD MISMATCH",
                "  Your dataset:  human teleoperation",
                f"  Reference:     scripted/planner demonstrations  ({ref_name})",
                "",
                "  The reference was generated by a motion planner, not a human",
                "  operator. Its spike_rate is structurally elevated (~20–25%).",
                "  The spike-rate comparison below reflects this difference.",
                warn_thick,
                "",
            ]
        lines.extend(mismatch_lines)

    # 1. Velocity discontinuity
    y, r = your_metrics["vel_disc_rate"], ref_metrics["vel_disc_rate"]
    delta = (y - r) if y is not None and r is not None else None
    if y is not None and r is not None:
        interp, conf = _interp_vel_disc(y, r, ref_mode, ref_name)
    else:
        interp, conf = "Could not compute.", ""
    lines.append(
        _section(
            "VELOCITY DISCONTINUITY RATE",
            _pct(y),
            _pct(r),
            ref_name,
            _pct(delta),
            _delta_arrow(delta),
            interp,
            _claims.evidence_line("vel_disc_rate", ref_mode),
            conf,
        )
    )
    lines.append(divider)

    # 2. Jerk spike rate
    y, r = your_metrics["spike_rate"], ref_metrics["spike_rate"]
    delta = (y - r) if y is not None and r is not None else None
    if y is not None and r is not None:
        interp, conf = _interp_spike_rate(
            y,
            r,
            ref_mode,
            ref_name,
            yours_is_scripted=yours_is_scripted,
            ref_is_scripted=ref_scripted,
        )
    else:
        interp, conf = "Could not compute.", ""
    lines.append(
        _section(
            "JERK SPIKE RATE",
            _pct(y),
            _pct(r),
            ref_name,
            _pct(delta),
            _delta_arrow(delta),
            interp,
            _claims.evidence_line("spike_rate", ref_mode),
            conf,
        )
    )
    lines.append(divider)

    # 3. LDLJ
    y, r = your_metrics["ldlj"], ref_metrics["ldlj"]
    delta = (y - r) if y is not None and r is not None else None
    if y is not None and r is not None:
        interp, conf = _interp_ldlj(y, r, ref_mode, ref_name, your_action_dim, ref_action_dim)
    else:
        interp, conf = "Could not compute.", ""
    delta_str = f"{delta:+.2f}" if delta is not None else "n/a"
    arrow = ("  ▲ (smoother)" if delta > 0 else "  ▼ (rougher)") if delta is not None else ""
    lines.append(
        _section(
            "LDLJ",
            _f2(y),
            _f2(r),
            ref_name,
            delta_str,
            arrow,
            interp,
            _claims.evidence_line("ldlj", "any"),
            conf,
        )
    )
    lines.append(divider)

    # 4. Timestamp jitter
    y, r = your_metrics["jitter_cv"], ref_metrics["jitter_cv"]
    delta = (y - r) if y is not None and r is not None else None
    if y is not None:
        interp, conf = _interp_temporal(y, r or 0, "jitter_cv", ref_is_sim)
    else:
        interp, conf = "Could not compute.", ""
    ref_str = (_sci(r) + "  (sim)") if ref_is_sim else _sci(r)
    hw_class = "any" if ref_is_sim else "hardware"
    lines.append(
        _section(
            "TIMESTAMP JITTER CV",
            _sci(y),
            ref_str,
            ref_name,
            _sci(delta),
            _delta_arrow(delta),
            interp,
            _claims.evidence_line("jitter_cv", hw_class),
            conf,
        )
    )
    lines.append(divider)

    # 5. Timestamp dropout
    y, r = your_metrics["dropout_rate"], ref_metrics["dropout_rate"]
    delta = (y - r) if y is not None and r is not None else None
    if y is not None:
        interp, conf = _interp_temporal(y, r or 0, "dropout_rate", ref_is_sim)
    else:
        interp, conf = "Could not compute.", ""
    ref_str = (_pct(r) + "  (sim)") if ref_is_sim else _pct(r)
    lines.append(
        _section(
            "TIMESTAMP DROPOUT RATE",
            _pct(y),
            ref_str,
            ref_name,
            _pct(delta),
            _delta_arrow(delta),
            interp,
            _claims.evidence_line("dropout_rate", "any"),
            conf,
        )
    )
    lines.append(divider)

    # 6. Action entropy
    y, r = your_metrics["action_entropy"], ref_metrics["action_entropy"]
    delta = (y - r) if y is not None and r is not None else None
    if y is not None:
        interp, conf = _interp_entropy(y, r or 0, ref_name)
    else:
        interp, conf = "Could not compute.", ""
    lines.append(
        _section(
            "ACTION ENTROPY",
            f"{_f2(y)} bits/dim",
            f"{_f2(r)} bits/dim",
            ref_name,
            f"{delta:+.2f} bits/dim" if delta is not None else "n/a",
            _delta_arrow(delta),
            interp,
            _claims.evidence_line("action_entropy", ref_mode),
            conf,
        )
    )
    lines.append(thick)

    # ── Recommended Actions ───────────────────────────────────────────────────
    rec = _recommended_actions(your_path, your_metrics, ref_metrics, ref_is_sim, outlier_episodes)
    if rec:
        lines.append("")
        lines.append("RECOMMENDED ACTIONS")
        lines.append(divider)
        for action in rec:
            lines.append(f"  {action}")
        lines.append(divider)

    return "\n".join(lines)


def _recommended_actions(
    your_path: str,
    your_metrics: dict[str, Optional[float]],
    ref_metrics: dict[str, Optional[float]],
    ref_is_sim: bool,
    outlier_episodes: Optional[list],
) -> list[str]:
    """
    Build a prioritised list of concrete recommended actions based on observed
    metric values and episode-level outliers.
    """
    actions: list[str] = []

    # Jerk outliers → prune specific episodes
    if outlier_episodes:
        jerk_eps = [
            a
            for a in outlier_episodes
            if any(m in ("spike_rate", "vel_disc_rate") for m in a.metrics)
        ]
        if jerk_eps:
            ids = ", ".join(str(a.episode_id) for a in jerk_eps[:6])
            suffix = f" (and {len(jerk_eps) - 6} more)" if len(jerk_eps) > 6 else ""
            actions.append(
                f"Prune episode(s) {ids}{suffix} — jerk outliers detected by MAD analysis."
            )

    # High dropout → fix recording pipeline
    dropout = your_metrics.get("dropout_rate")
    if dropout is not None and dropout > 0.05:
        actions.append(
            f"Dropout rate is {dropout:.1%}. Fix the camera/sensor logging loop "
            "to eliminate dropped frames before starting training."
        )
    elif dropout is not None and dropout > 0.01:
        actions.append(
            f"Dropout rate is {dropout:.1%}. Filter or interpolate "
            "affected episodes before training."
        )

    # High jitter → resample
    jitter = your_metrics.get("jitter_cv")
    if jitter is not None and jitter > 0.30:
        actions.append(
            f"Timestamp jitter CV is {jitter:.2f} (high). "
            "Resample to a uniform control frequency before training "
            "time-series policies (ACT, Diffusion Policy)."
        )

    # High vel_disc under position control
    vd = your_metrics.get("vel_disc_rate")
    ref_vd = ref_metrics.get("vel_disc_rate")
    if vd is not None and ref_vd is not None and vd > 0.04 and ref_vd < 0.05:
        actions.append(
            f"Velocity discontinuity rate is {vd:.1%} (above 4% position-control "
            "threshold). Investigate command packet drops, hardware communication "
            "lag, or abrupt operator corrections."
        )

    # Low entropy
    entropy = your_metrics.get("action_entropy")
    if entropy is not None and entropy < 3.0:
        actions.append(
            f"Action entropy is low ({entropy:.2f} bits/dim). "
            "Collect more diverse demonstrations before training to improve "
            "out-of-distribution generalisation."
        )

    return actions


def _ref_is_sim(ref_metrics: dict) -> bool:
    """Heuristic: reference is from sim if jitter CV is near machine precision."""
    jitter = ref_metrics.get("jitter_cv")
    return jitter is not None and jitter < 1e-3


def _dataset_is_scripted(report: DiagnosticReport) -> bool:
    """True when the ControlSmoothnessAnalyzer emitted a scripted motion signature."""
    return any(f.metric == "motion_collection_signature" for f in report.flags)


def _ref_is_scripted(ref_metrics: dict[str, Optional[float]]) -> bool:
    """
    Infer whether a reference profile was collected by a scripted planner.
    Uses the same empirical discriminant as ControlSmoothnessAnalyzer:
      spike_rate > 10% AND vel_disc_rate < 1.5%.
    """
    spike = ref_metrics.get("spike_rate")
    vd = ref_metrics.get("vel_disc_rate")
    return spike is not None and vd is not None and spike > 0.10 and vd < 0.015


# ── community comparison ──────────────────────────────────────────────────────

# Direction: True = higher raw value is better (higher rank)
_COMMUNITY_HIGHER_BETTER = {"action_entropy", "contact_phase_fraction"}
# All other fingerprint metrics: lower value is better

_COMMUNITY_LABELS = {
    "ldlj": "LDLJ (smoothness)",
    "spike_rate": "Jerk spike rate",
    "vel_disc_rate": "Velocity discontinuity",
    "dropout_rate": "Timestamp dropout",
    "jitter_cv": "Timestamp jitter CV",
    "action_entropy": "Action entropy",
    "contact_phase_fraction": "Contact phase fraction",
}


def fetch_community_percentiles(policy_family: str) -> Optional[dict]:
    """GET /v1/percentiles from Calibra Cloud. Returns None on any failure."""
    base = os.environ.get("CALIBRA_CLOUD_URL", "https://app.calibra.io")
    url = f"{base}/v1/percentiles?policy_family={policy_family}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _pct_rank(val: float, p: dict, higher_better: bool) -> Optional[float]:
    """Estimate percentile rank (0–100, higher = better) for val given p25/50/75/90."""
    p25, p50, p75, p90 = p.get("p25"), p.get("p50"), p.get("p75"), p.get("p90")
    if None in (p25, p50, p75, p90):
        return None
    # Raw position in the sorted distribution (0 = lowest, 100 = highest)
    if val <= p25:
        raw = max(0.0, 25.0 * val / p25) if p25 != 0 else 25.0
    elif val <= p50:
        raw = 25.0 + 25.0 * (val - p25) / max(p50 - p25, 1e-9)
    elif val <= p75:
        raw = 50.0 + 25.0 * (val - p50) / max(p75 - p50, 1e-9)
    elif val <= p90:
        raw = 75.0 + 15.0 * (val - p75) / max(p90 - p75, 1e-9)
    else:
        raw = 90.0
    return raw if higher_better else 100.0 - raw


def render_community_section(
    your_metrics: dict[str, Optional[float]],
    community: dict,
    policy_family: str,
) -> str:
    n_community = community.get("n", 0)
    percentiles = community.get("percentiles", {})
    divider = "─" * _WIDTH
    thick = "━" * _WIDTH

    lines = [
        "",
        thick,
        f"COMMUNITY COMPARISON  ({policy_family} · {n_community} datasets in Calibra Cloud)",
        thick,
        f"  {'Metric':<28} {'Yours':>8}  {'p50':>8}  {'Rank':>6}  Status",
        divider,
    ]

    for key in ["vel_disc_rate", "spike_rate", "ldlj", "jitter_cv", "dropout_rate", "action_entropy"]:
        val = your_metrics.get(key)
        p = percentiles.get(key)
        label = _COMMUNITY_LABELS.get(key, key)

        if val is None or p is None:
            lines.append(f"  {label:<28} {'n/a':>8}  {'n/a':>8}  {'—':>6}")
            continue

        higher_better = key in _COMMUNITY_HIGHER_BETTER or key == "ldlj"
        rank = _pct_rank(val, p, higher_better)
        p50 = p.get("p50")

        if key in ("ldlj",):
            val_str = f"{val:.2f}"
            p50_str = f"{p50:.2f}" if p50 is not None else "n/a"
        elif key in ("jitter_cv",):
            val_str = f"{val:.4f}"
            p50_str = f"{p50:.4f}" if p50 is not None else "n/a"
        else:
            val_str = f"{val:.1%}"
            p50_str = f"{p50:.1%}" if p50 is not None else "n/a"

        if rank is None:
            rank_str, status = "—", ""
        elif rank >= 75:
            rank_str, status = f"top {100 - rank:.0f}%", "✅"
        elif rank >= 40:
            rank_str, status = f"{rank:.0f}th", "🟡"
        else:
            rank_str, status = f"bot {rank:.0f}%", "⚠️ "

        lines.append(f"  {label:<28} {val_str:>8}  {p50_str:>8}  {rank_str:>6}  {status}")

    lines += [
        divider,
        "  Rank = percentile better than X% of community (higher is better for all metrics).",
        "  Run `calibra calibrate --global` to download community-fitted prediction weights.",
        thick,
    ]
    return "\n".join(lines)


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
    p.add_argument("path", help="Path or Hub ID of dataset to analyse (hf:// URIs supported)")
    p.add_argument("reference", help="Reference name (e.g. 'pusht', 'aloha')")
    p.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--gripper-dims",
        metavar="DIMS",
        default=None,
        help="Comma-separated gripper dimension indices to exclude "
        "from smoothness metrics (e.g. '6,13'). "
        "Use '' to disable gripper exclusion.",
    )
    p.add_argument(
        "--no-recommendations", action="store_true", help="Skip the Recommended Actions section"
    )
    p.add_argument(
        "--community",
        action="store_true",
        help=(
            "Append a community percentile table showing how your dataset ranks "
            "against all datasets in Calibra Cloud. Requires an internet connection."
        ),
    )
    p.add_argument(
        "--policy",
        "-p",
        metavar="FAMILY",
        default="generic",
        help="Policy family for community percentile lookup (e.g. 'diffusion', 'act'). Default: generic",
    )
    args = p.parse_args(argv)

    # Strip hf:// prefix — the ingestion layer handles bare "org/repo" IDs.
    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://") :]

    # resolve reader
    reader = None
    if args.format:
        from calibra.__main__ import _get_reader

        reader = _get_reader(args.format)

    # resolve gripper dims
    gripper_dims: list[int] = [-1]  # default: last dim
    if args.gripper_dims is not None:
        raw = args.gripper_dims.strip()
        gripper_dims = [int(x) for x in raw.split(",") if x.strip()] if raw else []

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    # load reference
    try:
        ref_data = load_reference(args.reference)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    # run pipeline on user's dataset
    log(f"Loading {dataset_path!r} ...")
    try:
        pipeline = Pipeline(
            analyzers=[
                TemporalAnalyzer(),
                ControlSmoothnessAnalyzer(gripper_dims=gripper_dims),
                CoverageEntropyAnalyzer(),
                TaskStructureAnalyzer(),
            ]
        )
        report: DiagnosticReport = pipeline.analyze_path(dataset_path, reader=reader)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    log(f"  {report.n_episodes} episodes  ·  {report.n_samples} steps")

    # episode-level outlier detection for recommendations
    outlier_episodes = None
    if not args.no_recommendations:
        from calibra.anomalies import find_outliers

        outlier_episodes = find_outliers(report)

    # extract metrics and render
    your_metrics = metrics_from_report(report)
    ref_metrics = metrics_from_reference(ref_data)

    # detect collection-method mismatch
    yours_is_scripted = _dataset_is_scripted(report)
    ref_scripted = _ref_is_scripted(ref_metrics)

    # infer action dim from first episode via report metadata
    your_action_dim = None
    for result in report.analyzer_results:
        raw = result.raw_metrics
        dim = raw.get("action_entropy", {}).get("action_dim")
        if dim is not None:
            your_action_dim = int(dim)
            break

    output = render_comparison(
        your_path=dataset_path,
        your_metrics=your_metrics,
        your_n_episodes=report.n_episodes,
        your_action_dim=your_action_dim,
        ref_data=ref_data,
        ref_metrics=ref_metrics,
        ref_name=args.reference,
        outlier_episodes=outlier_episodes,
        yours_is_scripted=yours_is_scripted,
        ref_scripted=ref_scripted,
    )
    print(output)

    if args.community:
        log(f"Fetching community percentiles for policy_family={args.policy!r} ...")
        community = fetch_community_percentiles(args.policy)
        if community and community.get("n", 0) >= 5:
            print(render_community_section(your_metrics, community, args.policy))
        elif community is not None:
            print(
                f"\n[community] Only {community.get('n', 0)} records in Calibra Cloud for "
                f"'{args.policy}' — need ≥5 to show percentiles. Record outcomes with "
                "`calibra predict --record-outcome RATE` to contribute."
            )
        else:
            print(
                "\n[community] Could not reach Calibra Cloud. Check your connection or set "
                "CALIBRA_CLOUD_URL to a reachable server."
            )
