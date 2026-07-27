"""
Robot Dataset Health Check — powered by Calibra

Public demo: sample audit on up to SAMPLE_EPISODE_CAP episodes.
Full audit: pip install calibra-robotics && calibra audit hf://<dataset>
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
SAMPLE_EPISODE_CAP   = 20
AUDIT_TIMEOUT_S      = 90
BENCHMARK_DATASET_ID = "omert27/calibra-robot-dataset-quality-benchmark"
SPACE_URL            = "https://huggingface.co/spaces/omert27/robot-dataset-health-check"

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


def _run_sample_audit(dataset_id: str) -> dict:
    from calibra.ingestion.registry import load
    from calibra.pipeline import Pipeline
    from calibra.report_json import assemble_public_report
    from calibra.schema.public_report import DatasetInfo, SamplingConfig

    revision  = _hf_revision(dataset_id)
    batch     = load(dataset_id)
    n_total   = batch.n_episodes
    is_sample = n_total > SAMPLE_EPISODE_CAP

    if is_sample:
        batch.episodes = batch.episodes[:SAMPLE_EPISODE_CAP]
        batch._n_samples_hint = None

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
        "score":            overall.score,
        "grade":            overall.grade,
        "cert":             overall.certification,
        "n_episodes":       diag.n_episodes,
        "n_episodes_total": n_total,
        "n_samples":        diag.n_samples,
        "fmt":              diag.format,
        "findings":         public.results.findings,
        "dimensions":       public.results.dimensions,
        "is_sample":        is_sample,
        "report_path":      _write_temp_report(public, dataset_id),
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
    error:  list  = []

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
        progress(min(0.15 + elapsed / AUDIT_TIMEOUT_S * 0.75, 0.90),
                 desc=f"Running quality checks ({elapsed}s) ...")

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
    "pass":        ("✓ Certified",               "#22c55e"),
    "provisional": ("~ Provisionally Certified",  "#f59e0b"),
    "fail":        ("✗ Not Certified",             "#ef4444"),
}

_SCORE_MEANING = [
    (90, "Excellent quality — ready for training."),
    (80, "Good quality — minor issues worth a quick review."),
    (70, "Generally usable — some quality issues worth reviewing before training."),
    (60, "Moderate issues — consider selective episode retention."),
    (40, "Significant quality issues — pruning recommended before training."),
    (0,  "Substantial quality issues — major cleanup recommended."),
]

_DIM_LABELS = {
    "temporal_integrity":  "Temporal Consistency",
    "motion_quality":      "Motion Quality",
    "behavioral_coverage": "Behavioral Coverage",
    "task_integrity":      "Task Integrity",
}

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
    color   = _BADGE_COLORS.get(grade, "lightgrey")
    message = urllib.parse.quote(f"{grade} · {score:.0f}/100", safe="")
    label   = urllib.parse.quote("Calibra Health", safe="")
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
        s     = d["score"]
        g     = d.get("grade", "?")
        color = _band_color(s)
        short = rid.split("/")[-1].replace("_", " ")
        delta = s - score
        sign  = "+" if delta >= 0 else "−"
        delta_color = "#22c55e" if delta >= 0 else "#f97316"
        items += (
            f'<a href="https://huggingface.co/datasets/{rid}" target="_blank"'
            f'   style="display:flex;justify-content:space-between;align-items:center;'
            f'   padding:6px 0;border-bottom:1px solid #313244;text-decoration:none">'
            f'  <span style="font-size:13px;color:#cdd6f4">{short}</span>'
            f'  <span style="font-size:13px;color:{color};font-weight:600">'
            f'    {s:.0f} <span style="color:{delta_color};font-size:11px;font-weight:400">'
            f'      ({sign}{abs(delta):.0f})</span>'
            f'  </span>'
            f'</a>'
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
    dim_dists    = _COMMUNITY_STATS.get("dimension_distributions", {})
    n            = _COMMUNITY_STATS.get("n_datasets", 0)
    overall_pct  = _pct_rank(score, overall_dist)
    top_pct      = 100 - overall_pct

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
        dist  = dim_dists.get(dim_key, [])
        if not dist:
            continue
        pct   = _pct_rank(dim_result.score, dist)
        top   = 100 - pct
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
            f'{rank_str}</span>'
            f'</div></div>'
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


# ── checklist ─────────────────────────────────────────────────────────────────

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
            n    = max(1, round(f.affected_fraction * n_ep))
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


# ── recommendation ────────────────────────────────────────────────────────────

_STRATEGY_RATIONALE = {
    "light quality filter":       "Removes only the clearest outliers — coverage and diversity are preserved.",
    "quality filter":             "Trims noisy or low-quality episodes while keeping most of the dataset's diversity.",
    "hybrid (quality + diversity)": "Balances quality filtering with diversity preservation to avoid overfitting.",
    "aggressive prune":           "Removes a large share of problematic episodes, prioritizing reliability over dataset size.",
}


def _recommend_html(score: float, findings: list, n_ep: int, n_total: int,
                    is_sample: bool, dataset_id: str) -> str:
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
        keep_pct, strategy = 20, "aggressive prune"

    keep_n = max(1, round(n_total * keep_pct / 100))
    sample_note = (
        f' <span style="color:#6c7086;font-size:11px">— estimate from {n_ep}/{n_total} ep sample</span>'
    ) if is_sample else ""
    rationale = _STRATEGY_RATIONALE.get(strategy, "")

    return (
        f'<div style="font-size:15px;color:#cdd6f4;font-weight:500">'
        f"Keep ~{keep_pct}% &nbsp;·&nbsp; {keep_n:,} episodes &nbsp;·&nbsp; {strategy}"
        f"{sample_note}</div>"
        f'<div style="margin-top:4px;font-size:12px;color:#a6adc8">{rationale}</div>'
        f'<div style="margin-top:10px;font-size:12px;color:#6c7086">'
        f'<code style="background:#313244;padding:3px 8px;border-radius:4px">'
        f"calibra prune hf://{dataset_id} --keep {keep_pct/100:.2f}</code></div>"
    )


# ── card wrapper ──────────────────────────────────────────────────────────────

def _card_wrapper(inner: str) -> str:
    return (
        '<div style="font-family:\'Inter\',\'Segoe UI\',sans-serif;background:#1e1e2e;'
        'border-radius:14px;padding:24px 28px;color:#cdd6f4;max-width:780px">'
        + inner + "</div>"
    )


def _score_header(dataset_id: str, score: float, grade: str, cert: str,
                  n_episodes: int, n_frames: int, fmt: str,
                  sample_tag: str = "") -> str:
    color      = _band_color(score)
    cert_label, cert_color = _CERT_TEXT.get(cert, ("Unknown", "#6b7280"))
    meaning    = _score_meaning(score)
    return f"""
<div style="font-size:11px;color:#6c7086;letter-spacing:.06em;margin-bottom:10px">
  ROBOT DATASET HEALTH CHECK
</div>
<div style="display:flex;align-items:flex-start;gap:24px;margin-bottom:18px">
  <div>
    <div style="font-size:11px;color:#6c7086;text-transform:uppercase;
                letter-spacing:.06em;margin-bottom:4px">Health Score</div>
    <div>
      <span style="font-size:68px;font-weight:800;color:{color};line-height:1">{score:.0f}</span>
      <span style="font-size:18px;color:#6c7086"> / 100</span>
    </div>
    <div style="font-size:13px;color:#a6adc8;margin-top:6px;max-width:260px">{meaning}</div>
    <div style="margin-top:10px;display:flex;align-items:center;gap:8px">
      <span style="font-size:13px;color:#6c7086">Grade:</span>
      <span style="font-size:14px;font-weight:600;color:{color}">{grade}</span>
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
    n_ep      = r["n_episodes"]
    n_total   = r["n_episodes_total"]
    is_sample = r["is_sample"]
    sample_tag = (
        f' <span style="font-size:11px;background:#313244;padding:2px 8px;'
        f'border-radius:10px;color:#a6adc8">sample {n_ep}/{n_total} ep</span>'
    ) if is_sample else ""

    inner = (
        _score_header(dataset_id, r["score"], r["grade"], r["cert"],
                      n_total, r["n_samples"], r["fmt"], sample_tag)
        + _checklist_html(r["findings"], n_ep)
        + _percentile_section(r["score"], r["dimensions"])
        + _similar_html(dataset_id, r["score"])
        + f'<div style="border-top:1px solid #313244;margin:14px 0"></div>'
        + f'<div style="font-size:11px;color:#6c7086;text-transform:uppercase;'
          f'letter-spacing:.06em;margin-bottom:8px">Retention Recommendation</div>'
        + _recommend_html(r["score"], r["findings"], n_ep, n_total, is_sample, dataset_id)
        + _full_audit_block(dataset_id)
        + _footer(dataset_id)
    )
    return _card_wrapper(inner)


def _render_cached_card(dataset_id: str, cached: dict) -> str:
    score  = cached["score"]
    grade  = cached.get("grade", "?")
    cert   = cached.get("certification", "")
    n_ep   = cached.get("n_episodes") or 0
    n_fr   = cached.get("n_frames") or 0
    n_iss  = cached.get("n_critical") or 0

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
        f'<div style="color:#f59e0b;font-size:13px;margin-top:6px">'
        f'{n_iss} quality issue{"s" if n_iss != 1 else ""} detected</div>'
    ) if n_iss else (
        '<div style="color:#22c55e;font-size:13px;margin-top:6px">No quality issues detected</div>'
    )
    detail_link = (
        f'<div style="font-size:13px;color:#a6adc8;margin-bottom:14px">'
        f'  Full per-episode results in the '
        f'  <a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"'
        f'     style="color:#89b4fa">community benchmark dataset →</a>'
        f'</div>'
    )

    inner = (
        _score_header(dataset_id, score, grade, cert, n_ep, n_fr, "lerobot", cached_tag)
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
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
    title="Robot Dataset Health Check",
    theme=gr.themes.Default(primary_hue="violet"),
    css="""
    .gr-button-primary { background: #7c3aed !important; border-color: #7c3aed !important; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown("""
# Robot Dataset Health Check

**Audit any LeRobot dataset before training.**

Enter a dataset ID and get a health score, quality issue findings,
community percentile comparison, similar datasets, and a keep-fraction recommendation.
Pre-audited datasets return instantly from the benchmark cache.
""")

    with gr.Row():
        inp = gr.Textbox(
            label="LeRobot Dataset ID",
            placeholder="lerobot/pusht",
            scale=5,
        )
        btn = gr.Button("Check Health", variant="primary", scale=1, min_width=140)

    out_html     = gr.HTML()
    out_download = gr.File(label="Download Full Report (JSON)")
    out_badge    = gr.Textbox(
        label="README badge — paste into your dataset card",
        interactive=False,
        visible=False,
    )

    gr.Examples(examples=EXAMPLES, inputs=inp, label="Try these")

    gr.Markdown("""
---
**What gets checked**

| Dimension | Checks | Why it matters |
|-----------|--------|-----------------|
| Temporal Consistency | Frame dropout rate, timestamp jitter, synchronization lag | Prevents observation/action misalignment during policy learning |
| Motion Quality | Action jerk spikes, velocity discontinuities, smoothness (LDLJ) | Reduces noisy demonstrations and unstable learned actions |
| Behavioral Coverage | Trajectory diversity, redundancy fraction, entropy | Improves state-space diversity and reduces overfitting |
| Task Integrity | Episode length distribution, phase balance, inactivity periods | Flags incomplete or abnormal demonstrations |

**Full audit locally** (all episodes, per-episode verdicts, certifiable report):
```bash
pip install calibra-robotics
calibra audit hf://lerobot/pusht
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
