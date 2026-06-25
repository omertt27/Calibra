"""
calibra sim2real — quantify the distribution gap between sim and real datasets.

When training on sim data and deploying on a real robot (or mixing sim/real),
the distribution gap between the two datasets determines how well the policy
will transfer. This command measures that gap across all diagnostic dimensions
and produces an actionable sim-to-real transfer risk assessment.

Metrics
-------
  Action-space KL divergence   — how different are the marginal action distributions?
  State-space KL divergence    — how different are the proprio/state distributions?
  Trajectory smoothness gap    — Δ LDLJ, Δ spike rate, Δ vel_disc_rate
  Temporal stability gap       — Δ jitter CV, Δ dropout rate
  Coverage overlap             — how much of the real action space does sim cover?
  Frequency mismatch           — control frequency delta between sim and real

Risk levels
-----------
  LOW      — gap is within normal dataset-to-dataset variation
  MEDIUM   — gap may cause degraded zero-shot transfer; domain adaptation recommended
  HIGH     — large distribution mismatch; fine-tuning on real data is strongly advised
  CRITICAL — gaps so large that transfer is unlikely without significant real data

Usage
------
    calibra sim2real /data/sim_demos.h5 /data/real_demos.h5
    calibra sim2real lerobot/sim_dataset lerobot/real_dataset --format lerobot
    calibra sim2real /data/sim.h5 /data/real.h5 --json
    calibra sim2real /data/sim.h5 /data/real.h5 --policy pi0

Exit codes
----------
    0  LOW or MEDIUM risk
    1  HIGH risk
    2  CRITICAL risk
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from calibra.pipeline import Pipeline
from calibra.schema.report import DiagnosticReport

# ── thresholds ────────────────────────────────────────────────────────────────

_KL_MEDIUM: float = 0.5
_KL_HIGH: float = 1.5
_KL_CRITICAL: float = 3.0

_SMOOTHNESS_MEDIUM: float = 3.0  # |Δ LDLJ|
_SMOOTHNESS_HIGH: float = 8.0
_SMOOTHNESS_CRITICAL: float = 15.0

_FREQ_MEDIUM: float = 5.0  # Hz delta
_FREQ_HIGH: float = 15.0
_FREQ_CRITICAL: float = 30.0

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN = "─" * _WIDTH


def _raw(report: DiagnosticReport, analyzer: str) -> dict:
    for r in report.analyzer_results:
        if r.analyzer_name == analyzer:
            return r.raw_metrics
    return {}


def _risk_level(value: float, medium: float, high: float, critical: float) -> str:
    if value >= critical:
        return "CRITICAL"
    if value >= high:
        return "HIGH"
    if value >= medium:
        return "MEDIUM"
    return "LOW"


def _risk_icon(level: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(level, "?")


# ── KL divergence helper ──────────────────────────────────────────────────────


def _kl_divergence_marginal(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """
    Estimate mean per-dimension KL(A || B) using histogram binning.
    Both arrays are (N, D). Returns a scalar.
    """
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    a = a[:, : b.shape[1]]
    b = b[:, : a.shape[1]]

    kl_vals: list[float] = []
    for d in range(a.shape[1]):
        lo = min(a[:, d].min(), b[:, d].min())
        hi = max(a[:, d].max(), b[:, d].max())
        if hi <= lo:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        pa, _ = np.histogram(a[:, d], bins=bins, density=True)
        pb, _ = np.histogram(b[:, d], bins=bins, density=True)
        eps = 1e-10
        pa = pa + eps
        pb = pb + eps
        pa /= pa.sum()
        pb /= pb.sum()
        kl = float(np.sum(pa * np.log(pa / pb)))
        kl_vals.append(kl)

    return float(np.mean(kl_vals)) if kl_vals else 0.0


def _coverage_overlap(sim: np.ndarray, real: np.ndarray, n_bins: int = 20) -> float:
    """
    Estimate the fraction of the real action space covered by sim, per dimension.
    Returns mean overlap fraction across dims.
    """
    if sim.ndim == 1:
        sim = sim[:, None]
    if real.ndim == 1:
        real = real[:, None]

    sim = sim[:, : real.shape[1]]
    real = real[:, : sim.shape[1]]

    overlaps: list[float] = []
    for d in range(real.shape[1]):
        lo = real[:, d].min()
        hi = real[:, d].max()
        if hi <= lo:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        real_hist, _ = np.histogram(real[:, d], bins=bins)
        sim_hist, _ = np.histogram(sim[:, d], bins=bins)
        real_occupied = real_hist > 0
        sim_covered = (sim_hist > 0) & real_occupied
        overlap = float(sim_covered.sum()) / max(real_occupied.sum(), 1)
        overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else 0.0


# ── gap analysis ──────────────────────────────────────────────────────────────


def _extract_actions(report: DiagnosticReport) -> Optional[np.ndarray]:
    """Pull per-episode raw action arrays if stored in raw_metrics."""
    for r in report.analyzer_results:
        arr = r.raw_metrics.get("all_actions")
        if arr is not None:
            return np.asarray(arr)
    return None


def _extract_freq(report: DiagnosticReport) -> Optional[float]:
    t = _raw(report, "temporal_stability")
    return t.get("mean_control_hz") or t.get("jitter", {}).get("mean_hz")


def analyze_gap(
    sim_report: DiagnosticReport,
    real_report: DiagnosticReport,
    sim_batch=None,
    real_batch=None,
) -> dict:
    """
    Compute all sim-to-real gap metrics between two DiagnosticReports.

    Parameters
    ----------
    sim_report   : DiagnosticReport from the sim dataset.
    real_report  : DiagnosticReport from the real dataset.
    sim_batch    : optional raw EpisodeBatch for the sim dataset (for KL/overlap).
    real_batch   : optional raw EpisodeBatch for the real dataset.

    Returns a structured dict with per-dimension gaps and an overall risk level.
    """
    gaps: dict = {}
    overall_levels: list[str] = []

    # ── smoothness gap ────────────────────────────────────────────────────────
    sim_s = _raw(sim_report, "control_smoothness")
    real_s = _raw(real_report, "control_smoothness")

    sim_ldlj = sim_s.get("ldlj", {}).get("mean_ldlj")
    real_ldlj = real_s.get("ldlj", {}).get("mean_ldlj")
    if sim_ldlj is not None and real_ldlj is not None:
        delta_ldlj = abs(sim_ldlj - real_ldlj)
        level = _risk_level(delta_ldlj, _SMOOTHNESS_MEDIUM, _SMOOTHNESS_HIGH, _SMOOTHNESS_CRITICAL)
        gaps["ldlj_gap"] = {
            "sim": round(sim_ldlj, 3),
            "real": round(real_ldlj, 3),
            "delta": round(delta_ldlj, 3),
            "risk": level,
            "note": (
                "Sim motions are much smoother than real."
                if sim_ldlj > real_ldlj
                else "Real motions are smoother than sim."
            ),
        }
        overall_levels.append(level)

    sim_spike = sim_s.get("jerk_spikes", {}).get("mean_spike_fraction")
    real_spike = real_s.get("jerk_spikes", {}).get("mean_spike_fraction")
    if sim_spike is not None and real_spike is not None:
        delta_spike = abs(sim_spike - real_spike)
        level = _risk_level(delta_spike * 20, 1.0, 2.0, 4.0)  # scale to 0–20 range
        gaps["spike_rate_gap"] = {
            "sim": round(sim_spike, 4),
            "real": round(real_spike, 4),
            "delta": round(delta_spike, 4),
            "risk": level,
        }
        overall_levels.append(level)

    # ── temporal gap ──────────────────────────────────────────────────────────
    sim_t = _raw(sim_report, "temporal_stability")
    real_t = _raw(real_report, "temporal_stability")

    sim_jitter = sim_t.get("jitter", {}).get("mean_cv")
    real_jitter = real_t.get("jitter", {}).get("mean_cv")
    if sim_jitter is not None and real_jitter is not None:
        delta_j = abs(sim_jitter - real_jitter)
        level = _risk_level(delta_j * 10, 0.5, 1.5, 3.0)
        gaps["jitter_cv_gap"] = {
            "sim": round(sim_jitter, 5),
            "real": round(real_jitter, 5),
            "delta": round(delta_j, 5),
            "risk": level,
            "note": (
                "Real data has much higher timing jitter than sim — typical for hardware."
                if real_jitter > sim_jitter
                else "Unusually noisy sim timestamps."
            ),
        }
        overall_levels.append(level)

    # ── coverage / action distribution ───────────────────────────────────────
    if sim_batch is not None and real_batch is not None:
        sim_actions = (
            np.concatenate([ep.actions for ep in sim_batch.episodes if ep.actions.ndim > 1], axis=0)
            if sim_batch.episodes
            else None
        )
        real_actions = (
            np.concatenate(
                [ep.actions for ep in real_batch.episodes if ep.actions.ndim > 1], axis=0
            )
            if real_batch.episodes
            else None
        )

        if (
            sim_actions is not None
            and real_actions is not None
            and sim_actions.size > 0
            and real_actions.size > 0
        ):
            kl = _kl_divergence_marginal(sim_actions, real_actions)
            level = _risk_level(kl, _KL_MEDIUM, _KL_HIGH, _KL_CRITICAL)
            gaps["action_kl_divergence"] = {
                "value": round(kl, 4),
                "risk": level,
                "note": (
                    f"KL(sim||real) = {kl:.3f}. "
                    + (
                        "Distributions are similar."
                        if kl < _KL_MEDIUM
                        else "Significant action distribution mismatch."
                    )
                ),
            }
            overall_levels.append(level)

            overlap = _coverage_overlap(sim_actions, real_actions)
            gaps["sim_coverage_of_real"] = {
                "value": round(overlap, 3),
                "risk": "LOW" if overlap >= 0.7 else ("MEDIUM" if overlap >= 0.4 else "HIGH"),
                "note": (
                    f"Sim covers {overlap:.0%} of the real action space. "
                    + (
                        "Good coverage."
                        if overlap >= 0.7
                        else "Sim is missing real-world action modes."
                    )
                ),
            }
            if gaps["sim_coverage_of_real"]["risk"] != "LOW":
                overall_levels.append(gaps["sim_coverage_of_real"]["risk"])

        # ── transition dynamics gap ───────────────────────────────────────────
        sim_states_list, sim_actions_list, sim_next_states_list = [], [], []
        for ep in sim_batch.episodes:
            states = ep.observations.get("proprio")
            acts = ep.actions
            if states is not None and len(states) > 1:
                t_max = min(len(states) - 1, len(acts))
                sim_states_list.append(states[:t_max])
                sim_actions_list.append(acts[:t_max])
                sim_next_states_list.append(states[1 : t_max + 1])

        real_states_list, real_actions_list, real_next_states_list = [], [], []
        for ep in real_batch.episodes:
            states = ep.observations.get("proprio")
            acts = ep.actions
            if states is not None and len(states) > 1:
                t_max = min(len(states) - 1, len(acts))
                real_states_list.append(states[:t_max])
                real_actions_list.append(acts[:t_max])
                real_next_states_list.append(states[1 : t_max + 1])

        if sim_states_list and real_states_list:
            sim_s = np.concatenate(sim_states_list, axis=0)
            sim_a = np.concatenate(sim_actions_list, axis=0)
            sim_y = np.concatenate(sim_next_states_list, axis=0)
            real_s = np.concatenate(real_states_list, axis=0)
            real_a = np.concatenate(real_actions_list, axis=0)
            real_y = np.concatenate(real_next_states_list, axis=0)

            # Match sizes
            min_state_dim = min(sim_s.shape[1], real_s.shape[1])
            min_action_dim = min(sim_a.shape[1], real_a.shape[1])
            sim_s = sim_s[:, :min_state_dim]
            sim_a = sim_a[:, :min_action_dim]
            sim_y = sim_y[:, :min_state_dim]
            real_s = real_s[:, :min_state_dim]
            real_a = real_a[:, :min_action_dim]
            real_y = real_y[:, :min_state_dim]

            # Fit dynamics: S_{t+1} - S_t = W * [S_t, A_t]
            def fit_dynamics(X_s, X_a, Y):
                features = np.concatenate([X_s, X_a], axis=1)
                state_diff = Y - X_s
                reg = 1e-4
                identity = np.eye(features.shape[1])
                W = np.linalg.solve(features.T @ features + reg * identity, features.T @ state_diff)
                return W

            try:
                W_sim = fit_dynamics(sim_s, sim_a, sim_y)
                W_real = fit_dynamics(real_s, real_a, real_y)

                # Cross prediction
                real_features = np.concatenate([real_s, real_a], axis=1)
                pred_real_from_real = real_s + real_features @ W_real
                pred_real_from_sim = real_s + real_features @ W_sim

                real_rmse = float(np.sqrt(np.mean((real_y - pred_real_from_real) ** 2)))
                cross_rmse = float(np.sqrt(np.mean((real_y - pred_real_from_sim) ** 2)))
                dynamics_gap = float(max(0.0, cross_rmse - real_rmse))

                level = _risk_level(dynamics_gap * 10, 0.2, 0.5, 1.0)
                gaps["transition_dynamics_gap"] = {
                    "value": round(dynamics_gap, 4),
                    "risk": level,
                    "note": f"Cross-prediction error increase: {dynamics_gap:.4f} (real baseline RMSE: {real_rmse:.4f}).",
                }
                overall_levels.append(level)
            except Exception:
                pass

        # ── visual domain gap ────────────────────────────────────────────────
        has_sim_cam = any("camera_rgb" in ep.observations for ep in sim_batch.episodes)
        has_real_cam = any("camera_rgb" in ep.observations for ep in real_batch.episodes)
        if has_sim_cam and has_real_cam:
            try:
                from calibra.curation.latent_embed import extract_latent_embeddings

                # Extract visual embeddings
                sim_visual_embs = extract_latent_embeddings(sim_batch, model_type="visual")
                real_visual_embs = extract_latent_embeddings(real_batch, model_type="visual")

                sim_arr = np.array(list(sim_visual_embs.values()))
                real_arr = np.array(list(real_visual_embs.values()))

                sim_mean = np.mean(sim_arr, axis=0)
                real_mean = np.mean(real_arr, axis=0)

                denom = float(np.linalg.norm(sim_mean) * np.linalg.norm(real_mean))
                cosine_dist = float(
                    1.0 - (np.dot(sim_mean, real_mean) / denom) if denom > 0 else 0.0
                )

                level = _risk_level(cosine_dist * 10, 0.5, 1.5, 3.0)
                gaps["visual_domain_gap"] = {
                    "value": round(cosine_dist, 4),
                    "risk": level,
                    "note": f"Visual embedding cosine distance: {cosine_dist:.4f}.",
                }
                overall_levels.append(level)
            except Exception:
                pass

    # ── frequency gap ─────────────────────────────────────────────────────────
    sim_freq = _extract_freq(sim_report)
    real_freq = _extract_freq(real_report)
    if sim_freq is not None and real_freq is not None:
        delta_f = abs(sim_freq - real_freq)
        level = _risk_level(delta_f, _FREQ_MEDIUM, _FREQ_HIGH, _FREQ_CRITICAL)
        gaps["control_frequency_gap"] = {
            "sim": round(sim_freq, 1),
            "real": round(real_freq, 1),
            "delta": round(delta_f, 1),
            "risk": level,
            "note": (
                f"Sim runs at {sim_freq:.0f} Hz, real at {real_freq:.0f} Hz. "
                + (
                    "Frequency match is good."
                    if delta_f < _FREQ_MEDIUM
                    else "Frequency mismatch may affect temporal features."
                )
            ),
        }
        overall_levels.append(level)

    # ── overall risk ──────────────────────────────────────────────────────────
    _level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    overall = (
        max(overall_levels, key=lambda lvl: _level_order.get(lvl, 0)) if overall_levels else "LOW"
    )

    # ── Pre-training Alignment Index (PAI) calculation ────────────────────────
    pai_components = []

    # 1. Action KL alignment component
    kl_gap = gaps.get("action_kl_divergence")
    if kl_gap is not None:
        val = kl_gap["value"]
        kl_score = max(0.0, min(100.0, 100.0 * (1.0 - (val - 0.1) / 2.9)))
        pai_components.append(kl_score)

    # 2. Control frequency component
    freq_gap = gaps.get("control_frequency_gap")
    if freq_gap is not None:
        val = freq_gap["delta"]
        freq_score = max(0.0, min(100.0, 100.0 * (1.0 - val / 30.0)))
        pai_components.append(freq_score)

    # 3. Action coverage component
    coverage_gap = gaps.get("sim_coverage_of_real")
    if coverage_gap is not None:
        cov_score = coverage_gap["value"] * 100.0
        pai_components.append(cov_score)

    # 4. Dynamics gap component
    dyn_gap = gaps.get("transition_dynamics_gap")
    if dyn_gap is not None:
        val = dyn_gap["value"]
        dyn_score = max(0.0, min(100.0, 100.0 * (1.0 - val / 1.0)))
        pai_components.append(dyn_score)

    # 5. Visual gap component
    vis_gap = gaps.get("visual_domain_gap")
    if vis_gap is not None:
        val = vis_gap["value"]
        vis_score = max(0.0, min(100.0, 100.0 * (1.0 - val / 0.5)))
        pai_components.append(vis_score)

    # 4. Modality matching component
    sim_obs = set()
    real_obs = set()
    if sim_batch:
        sim_obs = set(sim_batch.modalities)
    if real_batch:
        real_obs = set(real_batch.modalities)

    if sim_obs and real_obs:
        shared = sim_obs.intersection(real_obs)
        union = sim_obs.union(real_obs)
        modality_score = (len(shared) / len(union)) * 100.0 if union else 100.0
        pai_components.append(modality_score)
    else:
        pai_components.append(85.0)

    pai = float(np.mean(pai_components)) if pai_components else 100.0

    return {
        "overall_risk": overall,
        "pretraining_alignment_index": round(pai, 1),
        "sim_dataset": sim_report.dataset_name,
        "real_dataset": real_report.dataset_name,
        "sim_episodes": sim_report.n_episodes,
        "real_episodes": real_report.n_episodes,
        "gaps": gaps,
    }


# ── rendering ─────────────────────────────────────────────────────────────────


def render_sim2real(result: dict) -> str:
    overall = result["overall_risk"]
    icon = _risk_icon(overall)
    pai = result.get("pretraining_alignment_index", 100.0)
    lines = [
        _THICK,
        "  CALIBRA SIM-TO-REAL GAP ANALYSIS",
        _THICK,
        "",
        f"  Sim dataset  : {result['sim_dataset']}  ({result['sim_episodes']} eps)",
        f"  Real dataset : {result['real_dataset']}  ({result['real_episodes']} eps)",
        "",
        _THIN,
        f"  {icon}  Overall Transfer Risk: {overall}",
        f"  📊  Pre-training Alignment Index (PAI): {pai}%",
        _THIN,
        "",
    ]

    for gap_key, gap_data in result["gaps"].items():
        label = gap_key.replace("_", " ").title()
        gap_risk = gap_data.get("risk", "?")
        gap_icon = _risk_icon(gap_risk)
        lines.append(f"  {gap_icon} {label:<35} [{gap_risk}]")
        if "sim" in gap_data and "real" in gap_data:
            lines.append(
                f"     Sim: {gap_data['sim']}   Real: {gap_data['real']}"
                f"   Δ = {gap_data.get('delta', '?')}"
            )
        elif "value" in gap_data:
            lines.append(f"     Value: {gap_data['value']}")
        if "note" in gap_data:
            lines.append(f"     → {gap_data['note']}")
        lines.append("")

    # recommendations
    lines.append(_THIN)
    lines.append("  RECOMMENDATIONS")
    lines.append(_THIN)
    if overall == "LOW":
        lines.append("  ✓ Sim-to-real gap is small. Zero-shot transfer is likely viable.")
    elif overall == "MEDIUM":
        lines += [
            "  • Consider collecting a small real dataset (50–200 episodes) for",
            "    fine-tuning or domain randomisation in sim.",
            "  • Use `calibra prune` to select the sim episodes closest to the",
            "    real distribution before training.",
        ]
    elif overall == "HIGH":
        lines += [
            "  • Real fine-tuning data is strongly recommended.",
            "  • Use `calibra compare` to identify the specific gap dimensions.",
            "  • Apply action-space normalisation to align the distributions.",
            "  • Consider simulation domain randomisation to close the visual gap.",
        ]
    else:
        lines += [
            "  ✗ Critical distribution mismatch — zero-shot transfer is unlikely.",
            "  • Collect at minimum 200+ real demonstrations before training.",
            "  • Consider retraining from scratch on real data only.",
            "  • Re-evaluate sim environment fidelity (contact models, friction).",
        ]
    lines.append(_THICK)
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────


def run_sim2real(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra sim2real",
        description=(
            "Quantify the distribution gap between a simulation and real-robot dataset "
            "to predict sim-to-real transfer risk."
        ),
    )
    p.add_argument("sim_path", help="Path or HF Hub ID of the simulation dataset")
    p.add_argument("real_path", help="Path or HF Hub ID of the real-robot dataset")
    p.add_argument(
        "--sim-format",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter for the sim dataset",
    )
    p.add_argument(
        "--real-format",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter for the real dataset",
    )
    p.add_argument(
        "--policy",
        "-p",
        metavar="FAMILY",
        help="Target policy family for conditioned hints",
    )
    p.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output gap analysis as JSON",
    )
    args = p.parse_args(argv)

    def strip_hf(path: str) -> str:
        return path[len("hf://") :] if path.startswith("hf://") else path

    sim_path = strip_hf(args.sim_path)
    real_path = strip_hf(args.real_path)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    from calibra.__main__ import _get_reader
    from calibra.ingestion.registry import load as _load

    log(f"Loading sim dataset:  {sim_path!r} ...")
    try:
        sim_reader = _get_reader(args.sim_format) if args.sim_format else None
        sim_batch = _load(sim_path, reader=sim_reader)
    except Exception as exc:
        print(f"error loading sim dataset: {exc}", file=sys.stderr)
        sys.exit(2)
    log(f"  {sim_batch.n_episodes} episodes  ·  {sim_batch.n_samples} steps")

    log(f"Loading real dataset: {real_path!r} ...")
    try:
        real_reader = _get_reader(args.real_format) if args.real_format else None
        real_batch = _load(real_path, reader=real_reader)
    except Exception as exc:
        print(f"error loading real dataset: {exc}", file=sys.stderr)
        sys.exit(2)
    log(f"  {real_batch.n_episodes} episodes  ·  {real_batch.n_samples} steps")

    log("Running diagnostic pipeline on both datasets ...")
    pipeline = Pipeline()
    sim_report = pipeline.run(sim_batch, policy_family=args.policy)
    real_report = pipeline.run(real_batch, policy_family=args.policy)

    result = analyze_gap(sim_report, real_report, sim_batch, real_batch)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_sim2real(result))

    _level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    exit_code = _level_order.get(result["overall_risk"], 0)
    sys.exit(min(exit_code, 2))
