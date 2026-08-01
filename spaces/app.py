"""
Calibra — Dataset Integrity

Answers the first question robotics practitioners ask about a new dataset:
"can I trust it?" — before quality, coverage, or optimization matter.

Public demo: sample check on up to SAMPLE_EPISODE_CAP episodes.
Full check: pip install calibra-robotics && calibra integrity hf://<dataset>
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

import gradio as gr

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_EPISODE_CAP = 20
AUDIT_TIMEOUT_S = 90
BENCHMARK_DATASET_ID = "omert27/calibra-robot-dataset-quality-benchmark"
SPACE_URL = "https://huggingface.co/spaces/omert27/robot-dataset-health-check"

# ── community stats ───────────────────────────────────────────────────────────
_COMMUNITY_STATS: Optional[dict] = None
_CACHE: dict[str, dict] = {}


def _boot() -> None:
    global _COMMUNITY_STATS, _CACHE
    try:
        from huggingface_hub import hf_hub_download

        stats_path = hf_hub_download(
            repo_id=BENCHMARK_DATASET_ID, filename="community_stats.json", repo_type="dataset"
        )
        with open(stats_path, encoding="utf-8") as f:
            _COMMUNITY_STATS = json.load(f)

        manifest_path = hf_hub_download(
            repo_id=BENCHMARK_DATASET_ID, filename="manifest.json", repo_type="dataset"
        )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for ds in manifest.get("datasets", []):
            if ds.get("status") == "ok" and ds.get("score") is not None:
                _CACHE[ds["repository_id"]] = ds
    except Exception:
        pass


# ── audit ─────────────────────────────────────────────────────────────────────


def _hf_revision(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        return getattr(HfApi().dataset_info(repo_id=repo_id), "sha", None)
    except Exception:
        return None


def _run_integrity_check(batch) -> dict:
    """Mirrors `calibra integrity`'s own analyzer set and grouping — kept as a
    separate Pipeline run (not folded into the main audit below) so Integrity
    findings never leak into the Quality/Coverage dimension scoring, which
    routes unmatched metrics into a catch-all bucket (see
    calibra/schema/scoring.py `route_metric_to_dimension`)."""
    from calibra.analyzers.blur import BlurAnalyzer
    from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
    from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer
    from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
    from calibra.analyzers.task_structure import TaskStructureAnalyzer
    from calibra.analyzers.temporal import TemporalAnalyzer
    from calibra.integrity import _integrity_flags, _integrity_score
    from calibra.pipeline import Pipeline
    from calibra.schema.report import RiskLevel

    analyzers = [
        TemporalAnalyzer(),
        TaskStructureAnalyzer(),
        DuplicateFrameAnalyzer(),
        CameraFreezeAnalyzer(),
        BlurAnalyzer(),
        ControlSmoothnessAnalyzer(),
    ]
    report = Pipeline(analyzers=analyzers).run(batch)
    flags = _integrity_flags(report)
    score, status = _integrity_score(flags)
    return {
        "critical": [f for f in flags if f.level == RiskLevel.CRITICAL],
        "warnings": [f for f in flags if f.level == RiskLevel.WARNING],
        "passed": [f for f in flags if f.level == RiskLevel.OK],
        "score": score,
        "status": status,
    }


def _run_sample_audit(dataset_id: str) -> dict:
    from calibra.ingestion.registry import load
    from calibra.pipeline import Pipeline
    from calibra.report_json import assemble_public_report
    from calibra.schema.public_report import DatasetInfo, SamplingConfig

    revision = _hf_revision(dataset_id)
    batch = load(dataset_id)
    n_total = batch.n_episodes
    is_sample = n_total > SAMPLE_EPISODE_CAP

    if is_sample:
        batch.episodes = batch.episodes[:SAMPLE_EPISODE_CAP]
        batch._n_samples_hint = None

    integrity = _run_integrity_check(batch)
    diag = Pipeline().run(batch)

    dataset_info = DatasetInfo(
        provider="huggingface",
        repository_id=dataset_id,
        revision=revision,
        dataset_format=diag.format,
        episodes_total=n_total,
        episodes_audited=diag.n_episodes,
        frames_total=diag.n_samples,
    )
    public = assemble_public_report(
        diag,
        dataset_info=dataset_info,
        sampling=SamplingConfig(
            mode="random" if is_sample else "full",
            fraction=diag.n_episodes / n_total if is_sample else 1.0,
        ),
    )

    overall = public.results.overall
    return {
        "score": overall.score,
        "grade": overall.grade,
        "cert": overall.certification,
        "n_episodes": diag.n_episodes,
        "n_episodes_total": n_total,
        "n_samples": diag.n_samples,
        "fmt": diag.format,
        "findings": public.results.findings,
        "dimensions": public.results.dimensions,
        "is_sample": is_sample,
        "report_path": _write_temp_report(public, dataset_id),
        "integrity": integrity,
    }


def run_audit(dataset_id: str, progress=gr.Progress()):
    dataset_id = dataset_id.strip()

    if not dataset_id:
        raise gr.Error("Enter a dataset ID — e.g. lerobot/pusht")

    parts = dataset_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise gr.Error(
            f"'{dataset_id}' doesn't look like a Hugging Face dataset ID. "
            "Expected format: org/name  (e.g. lerobot/pusht)"
        )

    # ── cache hit ─────────────────────────────────────────────────────────────
    if dataset_id in _CACHE:
        progress(0.3, desc="Found in community benchmark ...")
        cached = _CACHE[dataset_id]
        progress(1.0)
        return (
            _render_cached_card(dataset_id, cached),
            None,
            _make_badge_markdown(cached["score"], cached.get("grade", "?"), dataset_id),
        )

    # ── live audit ────────────────────────────────────────────────────────────
    progress(0.10, desc=f"Loading {dataset_id} ...")
    result: dict = {}
    error: list = []

    def _worker():
        try:
            result.update(_run_sample_audit(dataset_id))
        except Exception as exc:
            error.append(str(exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    elapsed = 0
    while thread.is_alive() and elapsed < AUDIT_TIMEOUT_S:
        thread.join(timeout=3)
        elapsed += 3
        progress(
            min(0.15 + elapsed / AUDIT_TIMEOUT_S * 0.75, 0.90),
            desc=f"Running quality checks ({elapsed}s) ...",
        )

    if thread.is_alive():
        raise gr.Error(
            f"Audit timed out after {AUDIT_TIMEOUT_S}s — dataset may be too large for the demo. "
            f"Run locally:  calibra audit hf://{dataset_id}"
        )
    if error:
        msg = error[0]
        if "not found" in msg.lower() or "404" in msg:
            raise gr.Error(f"Dataset '{dataset_id}' not found on Hugging Face Hub.")
        if any(k in msg.lower() for k in ("lerobot", "parquet", "episode_index")):
            raise gr.Error(
                f"'{dataset_id}' doesn't appear to be a LeRobot dataset. "
                "This tool audits LeRobot-format datasets only."
            )
        raise gr.Error(f"Audit failed: {msg[:300]}")

    progress(0.95, desc="Rendering results ...")
    return (
        _render_health_card(dataset_id, result),
        result.get("report_path"),
        _make_badge_markdown(result["score"], result["grade"], dataset_id),
    )


# ── design tokens ─────────────────────────────────────────────────────────────

_SCORE_BANDS = [(90, "#22c55e"), (75, "#84cc16"), (60, "#f59e0b"), (40, "#f97316"), (0, "#ef4444")]
_BADGE_COLORS = {"A": "brightgreen", "B": "green", "C": "yellow", "D": "orange", "F": "red"}

_CERT_TEXT = {
    "pass": ("✓ Certified", "#22c55e"),
    "provisional": ("~ Provisionally Certified", "#f59e0b"),
    "fail": ("✗ Not Certified", "#ef4444"),
}

_SCORE_MEANING = [
    (90, "Excellent quality — ready for training."),
    (80, "Good quality — minor issues worth a quick review."),
    (70, "Generally usable — some quality issues worth reviewing before training."),
    (60, "Moderate issues — see Recommended Next Steps below."),
    (40, "Several episodes need review before training — see Recommended Next Steps."),
    (0, "Many episodes need review before training — see Recommended Next Steps."),
]

_DIM_LABELS = {
    "temporal_integrity": "Temporal Consistency",
    "motion_quality": "Motion Quality",
    "behavioral_coverage": "Behavioral Coverage",
    # "Integrity" is reserved for the front-door dataset-trust check above —
    # this dimension (episode length/phase balance) is renamed to avoid
    # implying it's part of that layer.
    "task_integrity": "Task Structure",
}

_INTEGRITY_STATUS_COLOR = {"Healthy": "#22c55e", "Warning": "#f59e0b", "Critical": "#ef4444"}

_SEV_ORDER = {"critical": 0, "warning": 1, "ok": 2, "info": 3}


def _band_color(score: float) -> str:
    for threshold, color in _SCORE_BANDS:
        if score >= threshold:
            return color
    return "#ef4444"


def _score_meaning(score: float) -> str:
    for threshold, text in _SCORE_MEANING:
        if score >= threshold:
            return text
    return ""


def _pct_rank(score: float, distribution: list) -> int:
    if not distribution:
        return 50
    return round(100 * sum(1 for s in distribution if score > s) / len(distribution))


# ── badge ─────────────────────────────────────────────────────────────────────


def _make_badge_markdown(score: float, grade: str, dataset_id: str) -> str:
    color = _BADGE_COLORS.get(grade, "lightgrey")
    message = urllib.parse.quote(f"{grade} · {score:.0f}/100", safe="")
    label = urllib.parse.quote("Calibra Health", safe="")
    img_url = f"https://img.shields.io/badge/{label}-{message}-{color}"
    return f"[![Calibra Health]({img_url})]({SPACE_URL})"


# ── similar datasets ──────────────────────────────────────────────────────────


def _similar_html(dataset_id: str, score: float) -> str:
    if not _CACHE:
        return ""
    ranked = sorted(
        ((rid, d) for rid, d in _CACHE.items() if rid != dataset_id),
        key=lambda x: abs(x[1]["score"] - score),
    )[:4]
    if not ranked:
        return ""

    items = ""
    for rid, d in ranked:
        s = d["score"]
        color = _band_color(s)
        short = rid.split("/")[-1].replace("_", " ")
        delta = s - score
        sign = "+" if delta >= 0 else "−"
        delta_color = "#22c55e" if delta >= 0 else "#f97316"
        items += (
            f'<a href="https://huggingface.co/datasets/{rid}" target="_blank"'
            f'   style="display:flex;justify-content:space-between;align-items:center;'
            f'   padding:6px 0;border-bottom:1px solid #313244;text-decoration:none">'
            f'  <span style="font-size:13px;color:#cdd6f4">{short}</span>'
            f'  <span style="font-size:13px;color:{color};font-weight:600">'
            f'    {s:.0f} <span style="color:{delta_color};font-size:11px;font-weight:400">'
            f"      ({sign}{abs(delta):.0f})</span>"
            f"  </span>"
            f"</a>"
        )

    return f"""
<div style="border-top:1px solid #313244;margin:14px 0"></div>
<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Similar datasets</div>
{items}
<div style="font-size:11px;color:#45475a;margin-top:6px">
  Nearest by health score in the community benchmark
</div>
"""


# ── percentile section ────────────────────────────────────────────────────────


def _percentile_section(score: float, dimensions: dict) -> str:
    if not _COMMUNITY_STATS:
        return ""

    overall_dist = _COMMUNITY_STATS.get("scores", [])
    dim_dists = _COMMUNITY_STATS.get("dimension_distributions", {})
    n = _COMMUNITY_STATS.get("n_datasets", 0)
    overall_pct = _pct_rank(score, overall_dist)
    top_pct = 100 - overall_pct

    if top_pct <= 10:
        headline, hcolor = f"Top {top_pct}% of audited datasets", "#22c55e"
    elif top_pct <= 30:
        headline, hcolor = f"Top {top_pct}% of audited datasets", "#84cc16"
    elif top_pct <= 60:
        headline, hcolor = f"Better than {overall_pct}% of audited datasets", "#f59e0b"
    else:
        headline, hcolor = f"Bottom {100 - overall_pct}% of audited datasets", "#f97316"

    dim_rows = ""
    for dim_key, dim_result in sorted(dimensions.items()):
        dist = dim_dists.get(dim_key, [])
        if not dist:
            continue
        pct = _pct_rank(dim_result.score, dist)
        top = 100 - pct
        color = _band_color(dim_result.score)
        label = _DIM_LABELS.get(dim_key, dim_key.replace("_", " ").title())
        if top <= 33:
            rank_str, rcolor = f"▲ top {top}%", "#22c55e"
        elif pct <= 33:
            rank_str, rcolor = f"▼ bottom {pct}%", "#ef4444"
        else:
            rank_str, rcolor = "≈ middle", "#f59e0b"

        dim_rows += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:4px 0;border-bottom:1px solid #313244">'
            f'<span style="font-size:13px;color:#a6adc8">{label}</span>'
            f'<div style="display:flex;align-items:center;gap:12px">'
            f'<span style="font-size:13px;color:{color};font-weight:600">{dim_result.score:.0f}</span>'
            f'<span style="font-size:12px;color:{rcolor};min-width:100px;text-align:right">'
            f"{rank_str}</span>"
            f"</div></div>"
        )

    return f"""
<div style="border-top:1px solid #313244;margin:14px 0"></div>
<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Community Rank</div>
<div style="font-size:30px;font-weight:800;color:{hcolor};margin-bottom:8px">{headline}</div>
<div style="background:#313244;border-radius:4px;height:8px;width:100%;
            margin-bottom:14px;overflow:hidden">
  <div style="background:{hcolor};height:100%;width:{overall_pct}%"></div>
</div>
{dim_rows}
<div style="font-size:11px;color:#45475a;margin-top:8px">
  Percentiles based on {n} audited public LeRobot datasets —
  sample grows as the <a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"
  style="color:#6c7086">community benchmark</a> expands.
</div>
"""


# ── integrity (front door) ──────────────────────────────────────────────────

_INTEGRITY_ICON = {
    "critical": ("✗", "#ef4444"),
    "warning": ("⚠", "#f59e0b"),
    "passed": ("✓", "#22c55e"),
}


def _integrity_row(f, kind: str) -> str:
    icon, color = _INTEGRITY_ICON[kind]
    text = f.interpretation
    row = (
        f'<div style="display:flex;gap:10px;align-items:flex-start;margin:5px 0">'
        f'<span style="color:{color};font-size:14px;min-width:16px;margin-top:1px">{icon}</span>'
        f'<span style="color:#cdd6f4;font-size:14px">{text}'
    )
    if kind != "passed":
        row += f'<div style="color:#6c7086;font-size:12px;margin-top:1px">{f.implication}</div>'
    row += "</span></div>"
    return row


def _integrity_html(integrity: dict) -> str:
    status = integrity["status"]
    color = _INTEGRITY_STATUS_COLOR.get(status, "#6c7086")
    critical, warnings, passed = integrity["critical"], integrity["warnings"], integrity["passed"]

    rows = "".join(_integrity_row(f, "critical") for f in critical)
    rows += "".join(_integrity_row(f, "warning") for f in warnings)
    rows += "".join(_integrity_row(f, "passed") for f in passed)
    if not rows:
        rows = (
            '<div style="color:#6c7086;font-size:13px">'
            "No integrity checks applicable to this dataset's modalities.</div>"
        )

    return f"""
<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
  <div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em">
    Dataset Integrity — can I trust this?
  </div>
  <div style="font-size:13px;font-weight:700;color:{color}">{status} · {integrity["score"]}/100</div>
</div>
{rows}
<div style="border-top:1px solid #313244;margin:16px 0"></div>
"""


# ── checklist ─────────────────────────────────────────────────────────────────


def _non_integrity_findings(findings: list) -> list:
    """Excludes metrics already surfaced in the Integrity section above, so a
    timestamp/sync issue isn't shown twice under two different headings."""
    from calibra.integrity import _INTEGRITY_METRICS

    return [f for f in findings if f.metric not in _INTEGRITY_METRICS]


def _findings_header_html(findings: list) -> str:
    n = sum(1 for f in findings if f.severity in ("critical", "warning"))
    if not n:
        return ""
    return (
        '<div style="font-size:11px;color:#6c7086;text-transform:uppercase;'
        'letter-spacing:.06em;margin-bottom:8px">Quality &amp; Coverage Findings</div>'
    )


def _checklist_html(findings: list, n_ep: int) -> str:
    problems = sorted(
        [f for f in findings if f.severity in ("critical", "warning")],
        key=lambda f: _SEV_ORDER.get(f.severity, 9),
    )[:8]

    if not problems:
        return (
            '<div style="display:flex;gap:10px;align-items:center">'
            '<span style="color:#22c55e;font-size:16px">✓</span>'
            '<span style="color:#cdd6f4;font-size:14px">No quality issues detected</span>'
            "</div>"
        )

    rows = []
    for f in problems:
        color = "#ef4444" if f.severity == "critical" else "#f59e0b"
        if f.affected_fraction is not None and n_ep > 0:
            n = max(1, round(f.affected_fraction * n_ep))
            text = f"{n} episode{'s' if n != 1 else ''} — {f.message}"
        else:
            text = f.message
        rows.append(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin:5px 0">'
            f'<span style="color:{color};font-size:15px;min-width:16px;margin-top:1px">⚠</span>'
            f'<span style="color:#cdd6f4;font-size:14px">{text}</span>'
            f"</div>"
        )
    return "\n".join(rows)


# ── recommended next steps ──────────────────────────────────────────────────────
#
# Findings route into a conservative action taxonomy. "Consider recollecting"
# only ever fires for a broad-based (majority-of-dataset), CRITICAL sync/
# integrity failure — never for jerk, duration, or rare-behavior findings,
# which stay in "Inspect" regardless of severity. An unusual trajectory can
# be a recording bug or the most valuable demonstration in the dataset;
# Calibra can't tell which, so it surfaces and ranks, it doesn't decide.

_VERIFY_METRICS = {
    "timestamp_jitter_cv",
    "timestamp_dropout_rate",
    "action_dropout_rate",
    "contact_dropout",
    "camera_physics_drift",
    "action_obs_misalignment",
}
_REDUNDANCY_METRICS = {"transition_redundancy"}

# key -> (emoji, color, label)
_NEXT_STEP_STYLE = {
    "recollect": ("🔴", "#ef4444", "Consider recollecting"),
    "verify": ("🟠", "#f97316", "Verify"),
    "inspect": ("🟡", "#f59e0b", "Inspect"),
    "redundancy": ("🔵", "#89b4fa", "Review"),
}

_NEXT_STEP_DETAIL = {
    "recollect": "episode{s} — recording pipeline issue, not an isolated anomaly",
    "verify": "episode{s} with possible recording anomalies",
    "inspect": "unusual episode{s}",
    "redundancy": "highly similar demonstration{s}",
}


def _categorize_finding(f) -> str:
    if f.metric in _REDUNDANCY_METRICS:
        return "redundancy"
    if f.metric in _VERIFY_METRICS:
        if f.severity == "critical" and (f.affected_fraction or 0) >= 0.5:
            return "recollect"
        return "verify"
    return "inspect"


def _next_steps_html(findings: list, n_ep: int) -> str:
    problems = [f for f in findings if f.severity in ("critical", "warning")]
    if not problems:
        return (
            '<div style="display:flex;gap:10px;align-items:center">'
            '<span style="color:#22c55e;font-size:16px">✓</span>'
            '<span style="color:#cdd6f4;font-size:14px">'
            "Dataset is otherwise healthy — no next steps flagged</span>"
            "</div>"
        )

    buckets: dict[str, list] = {"recollect": [], "verify": [], "inspect": [], "redundancy": []}
    for f in problems:
        buckets[_categorize_finding(f)].append(f)

    rows = []
    for key in ("recollect", "verify", "inspect", "redundancy"):
        items = buckets[key]
        if not items:
            continue
        items.sort(key=lambda f: (_SEV_ORDER.get(f.severity, 9), -(f.affected_fraction or 0)))
        primary = items[0]
        n = None
        if primary.affected_fraction is not None and n_ep > 0:
            n = max(1, round(primary.affected_fraction * n_ep))

        emoji, color, label = _NEXT_STEP_STYLE[key]
        if n:
            detail = _NEXT_STEP_DETAIL[key].format(s="s" if n != 1 else "")
            text = f"{n} {detail}"
        else:
            text = primary.message

        rows.append(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin:5px 0">'
            f'<span style="min-width:16px;margin-top:1px">{emoji}</span>'
            f'<span style="color:#cdd6f4;font-size:14px">'
            f'<span style="color:{color};font-weight:600">{label}</span> {text}</span>'
            f"</div>"
        )
    return "\n".join(rows)


# ── optional coreset ──────────────────────────────────────────────────────────

_STRATEGY_RATIONALE = {
    "light quality filter": "Removes only the clearest outliers — coverage and diversity are preserved.",
    "quality filter": "Trims noisy or low-quality episodes while keeping most of the dataset's diversity.",
    "hybrid (quality + diversity)": "Balances quality filtering with diversity preservation to avoid overfitting.",
    "heavy quality filter": "Keeps only episodes that clearly pass quality and diversity checks.",
}


def _coreset_html(
    score: float, findings: list, n_ep: int, n_total: int, is_sample: bool, dataset_id: str
) -> str:
    n_issues = sum(1 for f in findings if f.severity in ("critical", "warning"))
    if score >= 88 and n_issues == 0:
        keep_pct, strategy = 90, "light quality filter"
    elif score >= 75:
        keep_pct, strategy = 70, "quality filter"
    elif score >= 60:
        keep_pct, strategy = 50, "hybrid (quality + diversity)"
    elif score >= 40:
        keep_pct, strategy = 35, "hybrid (quality + diversity)"
    else:
        keep_pct, strategy = 20, "heavy quality filter"

    keep_n = max(1, round(n_total * keep_pct / 100))
    sample_note = (
        (
            f' <span style="color:#6c7086;font-size:11px">— estimate from {n_ep}/{n_total} ep sample</span>'
        )
        if is_sample
        else ""
    )
    rationale = _STRATEGY_RATIONALE.get(strategy, "")

    return f"""
<div style="background:#181825;border:1px solid #313244;border-radius:10px;
            padding:16px 18px;margin-top:14px">
  <div style="font-size:13px;font-weight:600;color:#cdd6f4;margin-bottom:4px">
    Build a smaller training set
  </div>
  <div style="font-size:12px;color:#6c7086;margin-bottom:10px">
    Optional — after reviewing the episodes flagged above, Calibra can build a
    quality- and coverage-aware coreset.
  </div>
  <div style="font-size:14px;color:#cdd6f4;font-weight:500">
    Keep ~{keep_pct}% &nbsp;·&nbsp; {keep_n:,} episodes &nbsp;·&nbsp; {strategy}{sample_note}
  </div>
  <div style="margin-top:4px;font-size:12px;color:#a6adc8">{rationale}</div>
  <div style="margin-top:10px;font-size:12px;color:#6c7086">
    <code style="background:#313244;padding:3px 8px;border-radius:4px">
      calibra prune hf://{dataset_id} --keep {keep_pct / 100:.2f}</code>
  </div>
</div>
"""


# ── card wrapper ──────────────────────────────────────────────────────────────


def _card_wrapper(inner: str) -> str:
    return (
        "<div style=\"font-family:'Inter','Segoe UI',sans-serif;background:#1e1e2e;"
        'border-radius:14px;padding:24px 28px;color:#cdd6f4;max-width:780px">' + inner + "</div>"
    )


def _score_header(
    dataset_id: str,
    score: float,
    grade: str,
    cert: str,
    n_episodes: int,
    n_frames: int,
    fmt: str,
    sample_tag: str = "",
) -> str:
    color = _band_color(score)
    cert_label, cert_color = _CERT_TEXT.get(cert, ("Unknown", "#6b7280"))
    meaning = _score_meaning(score)
    return f"""
<div style="font-size:11px;color:#6c7086;letter-spacing:.06em;margin-bottom:10px">
  QUALITY &amp; COVERAGE SCORE
</div>
<div style="display:flex;align-items:flex-start;gap:24px;margin-bottom:18px">
  <div>
    <div style="font-size:11px;color:#6c7086;text-transform:uppercase;
                letter-spacing:.06em;margin-bottom:4px">Score</div>
    <div>
      <span style="font-size:68px;font-weight:800;color:{color};line-height:1">{score:.0f}</span>
      <span style="font-size:18px;color:#6c7086"> / 100</span>
    </div>
    <div style="font-size:13px;color:#a6adc8;margin-top:6px;max-width:260px">{meaning}</div>
    <div style="margin-top:10px;display:flex;align-items:center;gap:8px">
      <span style="font-size:11px;color:#6c7086">grade {grade}</span>
      <span style="padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;
                   background:{cert_color}22;border:1px solid {cert_color};color:{cert_color}">
        {cert_label}
      </span>
    </div>
  </div>
  <div style="margin-left:auto;text-align:right;font-size:13px;color:#6c7086;line-height:2">
    <div><span style="color:#a6adc8">{dataset_id}</span>{sample_tag}</div>
    <div>{n_episodes:,} episodes &nbsp;·&nbsp; {n_frames:,} frames</div>
    <div>{fmt}</div>
  </div>
</div>
<div style="border-top:1px solid #313244;margin-bottom:14px"></div>
"""


def _full_audit_block(dataset_id: str) -> str:
    return f"""
<div style="border-top:1px solid #313244;margin:16px 0 12px"></div>
<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Full Audit</div>
<div style="font-size:13px;color:#a6adc8;margin-bottom:8px">
  Demo analyzes up to {SAMPLE_EPISODE_CAP} episodes.
  For all episodes, per-episode verdicts, and a certifiable report:
</div>
<div style="background:#181825;border-radius:8px;padding:10px 14px;font-size:13px;
            font-family:monospace;color:#cdd6f4">
  pip install calibra-robotics<br>
  calibra audit hf://{dataset_id}
</div>
"""


def _footer(dataset_id: str) -> str:
    return f"""
<div style="margin-top:16px;border-top:1px solid #313244;padding-top:10px;
            display:flex;justify-content:space-between;font-size:11px;color:#45475a">
  <span>Powered by <a href="https://github.com/omertt27/Calibra"
    style="color:#6c7086;text-decoration:none">Calibra</a> — robotics dataset observability</span>
  <span><a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"
    style="color:#6c7086;text-decoration:none">Community benchmark →</a></span>
</div>
"""


# ── full render ───────────────────────────────────────────────────────────────


def _render_health_card(dataset_id: str, r: dict) -> str:
    n_ep = r["n_episodes"]
    n_total = r["n_episodes_total"]
    is_sample = r["is_sample"]
    sample_tag = (
        (
            f' <span style="font-size:11px;background:#313244;padding:2px 8px;'
            f'border-radius:10px;color:#a6adc8">sample {n_ep}/{n_total} ep</span>'
        )
        if is_sample
        else ""
    )

    quality_findings = _non_integrity_findings(r["findings"])

    inner = (
        _integrity_html(r["integrity"])
        + _score_header(
            dataset_id,
            r["score"],
            r["grade"],
            r["cert"],
            n_total,
            r["n_samples"],
            r["fmt"],
            sample_tag,
        )
        + _findings_header_html(quality_findings)
        + _checklist_html(quality_findings, n_ep)
        + _percentile_section(r["score"], r["dimensions"])
        + _similar_html(dataset_id, r["score"])
        + '<div style="border-top:1px solid #313244;margin:14px 0"></div>'
        + '<div style="font-size:11px;color:#6c7086;text-transform:uppercase;'
        'letter-spacing:.06em;margin-bottom:8px">Recommended Next Steps</div>'
        + _next_steps_html(r["findings"], n_ep)
        + _coreset_html(r["score"], r["findings"], n_ep, n_total, is_sample, dataset_id)
        + _full_audit_block(dataset_id)
        + _footer(dataset_id)
    )
    return _card_wrapper(inner)


def _render_cached_card(dataset_id: str, cached: dict) -> str:
    score = cached["score"]
    grade = cached.get("grade", "?")
    cert = cached.get("certification", "")
    n_ep = cached.get("n_episodes") or 0
    n_fr = cached.get("n_frames") or 0
    n_iss = cached.get("n_critical") or 0

    class _Dim:
        def __init__(self, s):
            self.score = s

    dim_objs = {
        k: _Dim(v["score"])
        for k, v in (cached.get("dimensions") or {}).items()
        if isinstance(v, dict) and v.get("score") is not None
    }

    cached_tag = (
        ' <span style="font-size:11px;background:#313244;padding:2px 8px;'
        'border-radius:10px;color:#a6adc8">full audit cached</span>'
    )
    issue_line = (
        (
            f'<div style="color:#f59e0b;font-size:13px;margin-top:6px">'
            f"{n_iss} quality issue{'s' if n_iss != 1 else ''} detected</div>"
        )
        if n_iss
        else (
            '<div style="color:#22c55e;font-size:13px;margin-top:6px">No quality issues detected</div>'
        )
    )
    detail_link = (
        f'<div style="font-size:13px;color:#a6adc8;margin-bottom:14px">'
        f"  Full per-episode results in the "
        f'  <a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"'
        f'     style="color:#89b4fa">community benchmark dataset →</a>'
        f"</div>"
    )

    integrity_note = (
        '<div style="background:#181825;border:1px solid #313244;border-radius:8px;'
        'padding:10px 14px;margin-bottom:14px;font-size:12px;color:#a6adc8">'
        "This cached result predates the Integrity check — timestamp/sync, "
        "episode completeness, duplicate frames, camera freeze, blur, and "
        "jittery/jerky motion. "
        "<code style='background:#313244;padding:1px 5px;border-radius:3px'>"
        f"calibra integrity hf://{dataset_id}</code> runs it directly.</div>"
    )

    inner = (
        integrity_note
        + _score_header(dataset_id, score, grade, cert, n_ep, n_fr, "lerobot", cached_tag)
        + issue_line
        + "<br>"
        + detail_link
        + _percentile_section(score, dim_objs)
        + _similar_html(dataset_id, score)
        + _full_audit_block(dataset_id)
        + _footer(dataset_id)
    )
    return _card_wrapper(inner)


def _write_temp_report(public, dataset_id: str) -> str:
    slug = dataset_id.replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(tempfile.gettempdir(), f"calibra_{slug}_{ts}.json")
    public.write(path)
    return path


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["lerobot/pusht"],
    ["lerobot/aloha_sim_insertion_human"],
    ["lerobot/xarm_lift_medium"],
    ["lerobot/aloha_static_coffee"],
    ["lerobot/unitreeh1_fold_clothes"],
]

_boot()

with gr.Blocks(
    title="Calibra — Dataset Integrity",
    theme=gr.themes.Default(primary_hue="violet"),
    css="""
    .gr-button-primary { background: #7c3aed !important; border-color: #7c3aed !important; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown("""
# Calibra — Dataset Integrity

**Before diversity or coreset selection, can you trust this dataset?**

Enter a LeRobot dataset ID. Calibra checks Integrity first — timestamp
consistency, episode completeness, duplicate frames, camera freeze, blur,
jittery/jerky motion — then Quality and Coverage. Pre-checked datasets
return instantly from the benchmark cache.
""")

    with gr.Row():
        inp = gr.Textbox(
            label="LeRobot Dataset ID",
            placeholder="lerobot/pusht",
            scale=5,
        )
        btn = gr.Button("Check Integrity", variant="primary", scale=1, min_width=140)

    out_html = gr.HTML()
    out_download = gr.File(label="Download Full Report (JSON)")
    out_badge = gr.Textbox(
        label="README badge — paste into your dataset card",
        interactive=False,
        visible=False,
    )

    gr.Examples(examples=EXAMPLES, inputs=inp, label="Try these")

    gr.Markdown("""
---
**What gets checked**

| Layer | Checks | Answers |
|-------|--------|---------|
| **Integrity** (first) | Timestamp consistency, sensor sync, episode completeness, duplicate frames, camera freeze, blur, jerky/jittery motion (LDLJ, jerk spikes, velocity discontinuities) | Can I trust this dataset? |
| Quality | Action-state tracking error, scripted-vs-teleop motion signature | Is this data clean? |
| Coverage | Trajectory diversity, redundancy fraction, entropy | Does my robot see enough variety? |
| Task Structure | Episode length distribution, phase balance, inactivity periods | Are episodes complete and well-formed? |

After Integrity comes Quality, Coverage, and — for building smaller training
sets — Optimization.

**Full check locally** (all episodes, per-episode verdicts, certifiable report):
```bash
pip install calibra-robotics
calibra integrity hf://lerobot/pusht
calibra audit hf://lerobot/pusht      # quality + coverage scoring
```

*Powered by [Calibra](https://github.com/omertt27/Calibra) — open-source robotics dataset
observability*
""")

    def _with_badge_visible(dataset_id):
        html, download, badge = run_audit(dataset_id)
        return html, download, gr.update(value=badge, visible=badge is not None)

    btn.click(
        fn=_with_badge_visible,
        inputs=inp,
        outputs=[out_html, out_download, out_badge],
    )
    inp.submit(
        fn=_with_badge_visible,
        inputs=inp,
        outputs=[out_html, out_download, out_badge],
    )

if __name__ == "__main__":
    demo.launch()
