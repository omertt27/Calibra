"""
Assembles a CalibraReport (public JSON contract) from a DiagnosticReport
(internal pipeline output) plus caller-supplied dataset metadata.

This is the only place that reads calibra internals to produce the public
format. Downstream consumers (leaderboard, badge generator, dataset pages,
historical diffs) must be able to work from the JSON alone — they should
never need to import this module or anything else from calibra.

Immediate milestone: produce one schema-valid report for lerobot/pusht,
then verify that the JSON can drive an HTML dataset page, a badge, and a
historical diff without reading calibra Python objects.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Optional

from calibra import __version__
from calibra.certify import _grade
from calibra.schema.public_report import (
    AnomalySummary,
    AuditConfig,
    AuditResults,
    CalibraReport,
    DatasetInfo,
    DetectorCalibrationSummary,
    DimensionResult,
    EnvironmentInfo,
    EpisodeHash,
    EpisodeVerdicts,
    Finding,
    MetricValue,
    OverallResult,
    PolicyRecommendation,
    Recommendations,
    ReportMeta,
    SamplingConfig,
)
from calibra.schema.report import CompatibilityHint, DiagnosticReport, RiskFlag, RiskLevel
from calibra.schema.scoring import (
    CURRENT_RUBRIC,
    DIMENSION_WEIGHTS,
    dimension_score,
    flag_level_to_score,
    get_methodology,
    overall_score,
    route_metric_to_dimension,
    score_to_grade,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _pruning_to_verdicts(pruning_result) -> EpisodeVerdicts:
    """Convert an internal PruningResult to the public EpisodeVerdicts contract."""
    reject_ids = list(pruning_result.quality_fail_ids) + list(pruning_result.diversity_pruned_ids)
    return EpisodeVerdicts(
        keep_episode_ids=list(pruning_result.keep_episode_ids),
        reject_episode_ids=reject_ids,
        reason_codes=dict(pruning_result.fail_reasons),
        quality_scores=dict(pruning_result.quality_scores),
        n_original=pruning_result.n_original,
        n_kept=pruning_result.n_kept,
        keep_fraction_actual=round(pruning_result.keep_fraction_actual, 4),
        method=pruning_result.method,
    )


def _flag_to_finding(flag: RiskFlag) -> Finding:
    code = flag.metric.upper().replace(".", "_")
    return Finding(
        severity=flag.level.value.lower(),
        code=code,
        metric=flag.metric,
        message=flag.interpretation,
        implication=flag.implication,
        affected_fraction=flag.affected_fraction,
        observed_value=flag.observed.value,
        observed_unit=flag.observed.unit,
        threshold=flag.threshold,
    )


def _hints_to_recommendations(hints: list[CompatibilityHint]) -> Recommendations:
    def _map(compatible: Optional[bool], explanation: str) -> PolicyRecommendation:
        if compatible is True:
            return PolicyRecommendation(status="recommended", reason=explanation or None)
        if compatible is False:
            return PolicyRecommendation(status="not_recommended", reason=explanation or None)
        return PolicyRecommendation(status="review", reason=explanation or None)

    recs: dict[str, PolicyRecommendation] = {}
    for hint in hints:
        family = hint.policy_family.lower().replace("-", "_").replace(" ", "_")
        recs[family] = _map(hint.compatible, hint.explanation)

    return Recommendations(
        behavior_cloning=recs.get(
            "behavior_cloning", recs.get("bc", PolicyRecommendation(status="review"))
        ),
        act=recs.get("act", PolicyRecommendation(status="review")),
        diffusion_policy=recs.get(
            "diffusion_policy", recs.get("diffusion", PolicyRecommendation(status="review"))
        ),
        gr00t=recs.get("gr00t", PolicyRecommendation(status="review")),
    )


def _build_dimensions(flags: list[RiskFlag]) -> dict[str, DimensionResult]:
    dim_flags: dict[str, list[RiskFlag]] = {d: [] for d in DIMENSION_WEIGHTS}
    for flag in flags:
        dim = route_metric_to_dimension(flag.metric)
        dim_flags[dim].append(flag)

    dimensions: dict[str, DimensionResult] = {}
    for dim_name, weight in DIMENSION_WEIGHTS.items():
        metric_values: dict[str, MetricValue] = {}
        scores: list[float] = []

        for flag in dim_flags[dim_name]:
            s = flag_level_to_score(flag.level)
            scores.append(s)
            metric_values[flag.metric] = MetricValue(
                value=flag.observed.value,
                unit=flag.observed.unit,
                score=s,
                ci_lower=flag.observed.ci_lower,
                ci_upper=flag.observed.ci_upper,
                ci_level=flag.observed.ci_level,
                ci_method=flag.observed.ci_method or "bootstrap",
                methodology=get_methodology(flag.metric),
            )

        dimensions[dim_name] = DimensionResult(
            score=round(dimension_score(scores), 1),
            weight=weight,
            metrics=metric_values,
        )

    return dimensions


def _compute_confidence(flags: list[RiskFlag]) -> float:
    """
    Approximates audit confidence from CI widths.
    Returns 0-1; higher = tighter confidence intervals.
    Falls back to 0.9 when no flags carry CI bounds.
    """
    ratios: list[float] = []
    for flag in flags:
        obs = flag.observed
        if obs.value is not None and obs.ci_lower is not None and obs.ci_upper is not None:
            denom = abs(obs.value) + 1e-9
            relative_width = (obs.ci_upper - obs.ci_lower) / denom
            ratios.append(max(0.0, 1.0 - relative_width))
    return round(sum(ratios) / len(ratios), 3) if ratios else 0.9


def _build_anomaly_summary(
    diag: DiagnosticReport,
    dataset: Optional[str] = None,
    task_family: Optional[str] = None,
) -> Optional[AnomalySummary]:
    """
    Run episode-level anomaly detection on `diag` and return a calibrated summary.

    Returns None if the diagnostic report lacks per-episode data (fewer than 5
    episodes or missing per-episode metrics), keeping report generation cheap
    when the data is not present.

    Architecture note: a detected anomaly ≠ confirmed corruption ≠ DROP.
    The summary records what the detectors observed; the decision layer
    (EpisodeCharacterization / CurationReport) determines what to do.
    """
    from collections import Counter

    from calibra.anomalies import find_outliers, firing_rate_summary
    from calibra.calibration import DEFAULT_REGISTRY

    if diag.n_episodes < 5:
        return None

    anomalies = find_outliers(diag, dataset=dataset, task_family=task_family)
    if not anomalies and diag.n_episodes > 0:
        return AnomalySummary(
            total_flags=0,
            affected_episodes=0,
            affected_episode_rate=0.0,
            n_total_episodes=diag.n_episodes,
        )

    n_episodes = diag.n_episodes
    all_flags = [f for a in anomalies for f in a.flags]
    n_flags = len(all_flags)
    n_affected = len(anomalies)

    # Concentration: fraction of flags in top-5 most-flagged episodes
    per_ep = Counter(f.episode_id for a in anomalies for f in a.flags)
    top5 = sum(v for _, v in per_ep.most_common(5))
    top5_frac = top5 / n_flags if n_flags > 0 else 0.0

    # Position clustering
    sorted_idxs = sorted(a.episode_idx for a in anomalies)
    start_cluster = sum(1 for i in sorted_idxs if i <= int(n_episodes * 0.10))
    end_cluster = sum(1 for i in sorted_idxs if i >= int(n_episodes * 0.90))
    position_note = ""
    if start_cluster >= 2 and n_affected > 0 and start_cluster / n_affected >= 0.30:
        position_note = "concentrated near dataset start"
    elif end_cluster >= 2 and n_affected > 0 and end_cluster / n_affected >= 0.30:
        position_note = "concentrated near dataset end"

    # Per-detector calibration summaries
    rate_entries = firing_rate_summary(anomalies, n_episodes)
    detector_summaries: list[DetectorCalibrationSummary] = []
    for entry in rate_entries:
        det = entry["detector"]
        frac = entry["fraction"]
        baseline_rate = entry["benign_baseline_rate"]

        profile = DEFAULT_REGISTRY.lookup(det, dataset=dataset, task_family=task_family)
        n_baseline = profile.n_episodes if profile else None
        baseline_ds = profile.dataset if profile else None

        # Concentration per detector
        det_idxs = sorted(
            a.episode_idx for a in anomalies if any(f.metric == det for f in a.flags)
        )
        if len(det_idxs) == 0:
            conc = "unknown"
        elif len(det_idxs) == 1:
            conc = "spread"
        else:
            span = det_idxs[-1] - det_idxs[0]
            if span <= max(3, int(n_episodes * 0.10)):
                conc = "clustered"
            elif det_idxs[-1] >= int(n_episodes * 0.85):
                conc = "endpoint"
            else:
                conc = "spread"

        detector_summaries.append(
            DetectorCalibrationSummary(
                detector=det,
                n_flagged=entry["n_flagged"],
                fraction_flagged=round(frac, 4),
                benign_firing_rate=baseline_rate,
                corrupted_episode_detection_rate=None,  # populated by benchmark script
                baseline_dataset=baseline_ds,
                sample_count=n_baseline,
                concentration=conc,
            )
        )

    return AnomalySummary(
        total_flags=n_flags,
        affected_episodes=n_affected,
        affected_episode_rate=round(n_affected / n_episodes, 4) if n_episodes > 0 else 0.0,
        n_total_episodes=n_episodes,
        detectors=detector_summaries,
        top5_episode_flag_fraction=round(top5_frac, 3),
        position_note=position_note,
    )


def _config_hash(
    profile: Optional[str],
    rubric: str,
    sampling: SamplingConfig,
) -> str:
    config = {"profile": profile, "rubric": rubric, "sampling": sampling.model_dump()}
    canonical = json.dumps(config, sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ── public API ────────────────────────────────────────────────────────────────


def assemble_public_report(
    diag: DiagnosticReport,
    dataset_info: DatasetInfo,
    sampling: Optional[SamplingConfig] = None,
    profile: Optional[str] = None,
    pruning_result=None,
    episode_hashes: Optional[dict] = None,
    task_family: Optional[str] = None,
) -> CalibraReport:
    """
    Convert a DiagnosticReport into the public CalibraReport contract.

    After this call, no further access to calibra internals is needed.
    Write the result to JSON and all downstream consumers work from that.
    """
    if sampling is None:
        sampling = SamplingConfig(mode="full")

    grade_label, _ = _grade(diag)
    certification_map = {
        "CERTIFIED": "pass",
        "PROVISIONALLY CERTIFIED": "provisional",
        "NOT CERTIFIED": "fail",
    }
    certification = certification_map.get(grade_label, "provisional")
    critical_failures = [f.metric for f in diag.flags_at_level(RiskLevel.CRITICAL)]

    dimensions = _build_dimensions(diag.flags)
    dim_scores = {name: d.score for name, d in dimensions.items()}
    numeric_score = round(overall_score(dim_scores), 1)
    confidence = _compute_confidence(diag.flags)

    results = AuditResults(
        overall=OverallResult(
            score=numeric_score,
            grade=score_to_grade(numeric_score),
            confidence=confidence,
            certification=certification,
            critical_failures=critical_failures,
        ),
        dimensions=dimensions,
        findings=[_flag_to_finding(f) for f in diag.flags],
        recommendations=_hints_to_recommendations(diag.hints),
    )

    rubric = CURRENT_RUBRIC
    sampling_cfg = sampling
    audit_cfg = AuditConfig(
        profile=profile,
        configuration_hash=_config_hash(profile, rubric, sampling_cfg),
        scoring_rubric=rubric,
        sampling=sampling_cfg,
        environment=EnvironmentInfo(
            python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.system().lower() + "-" + platform.machine().lower(),
        ),
    )

    # Anomaly summary — runs find_outliers internally so callers don't pay the
    # cost unless they produce a full report. Returns None for small datasets.
    anomaly_summary = _build_anomaly_summary(
        diag,
        dataset=dataset_info.repository_id,
        task_family=task_family,
    )

    # Compute report ID from the body (excluding the id field itself)
    now = datetime.now(timezone.utc)
    verdicts = _pruning_to_verdicts(pruning_result) if pruning_result is not None else None
    ep_hashes = [
        EpisodeHash(episode_id=eid, hash=h) for eid, h in sorted((episode_hashes or {}).items())
    ]
    id_body = {
        "schema_version": "1.2.0",
        "generated_at": now.isoformat(),
        "calibra_version": __version__,
        "dataset": dataset_info.model_dump(),
        "audit": audit_cfg.model_dump(),
        "results": results.model_dump(),
        "episode_verdicts": verdicts.model_dump() if verdicts is not None else None,
        "anomaly_summary": anomaly_summary.model_dump() if anomaly_summary is not None else None,
        "episode_hashes": [h.model_dump() for h in ep_hashes],
    }
    report_id = CalibraReport.compute_id(id_body)

    return CalibraReport(
        schema_version="1.2.0",
        report=ReportMeta(
            id=report_id,
            generated_at=now,
            calibra_version=__version__,
            status="complete",
        ),
        dataset=dataset_info,
        audit=audit_cfg,
        results=results,
        episode_verdicts=verdicts,
        anomaly_summary=anomaly_summary,
        episode_hashes=ep_hashes,
    )


def dataset_info_from_report(diag: DiagnosticReport) -> DatasetInfo:
    """
    Infer DatasetInfo from a DiagnosticReport when no explicit metadata
    is available. Fills in only what the pipeline exposes directly.
    Callers should override revision and robot fields when known.
    """
    source = diag.source_path
    if "/" in source and not source.startswith("/") and not source[1:3] == ":\\":
        provider = "huggingface"
        repository_id = source
    else:
        provider = "local"
        repository_id = source

    return DatasetInfo(
        provider=provider,
        repository_id=repository_id,
        revision=None,
        dataset_format=diag.format,
        episodes_total=diag.n_episodes,
        episodes_audited=diag.n_episodes,
        frames_total=diag.n_samples,
    )
