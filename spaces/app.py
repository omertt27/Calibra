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
from datetime import datetime, timezone
from typing import Optional

import gradio as gr

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_EPISODE_CAP = 20
AUDIT_TIMEOUT_S    = 90
BENCHMARK_DATASET_ID = "omert27/calibra-robot-dataset-quality-benchmark"

# ── community stats ───────────────────────────────────────────────────────────
_COMMUNITY_STATS: Optional[dict] = None
_CACHE: dict[str, dict] = {}


def _boot() -> None:
    global _COMMUNITY_STATS, _CACHE
    try:
        from huggingface_hub import hf_hub_download

        stats_path = hf_hub_download(
            repo_id=BENCHMARK_DATASET_ID,
            filename="community_stats.json",
            repo_type="dataset",
        )
        with open(stats_path, encoding="utf-8") as f:
            _COMMUNITY_STATS = json.load(f)

        manifest_path = hf_hub_download(
            repo_id=BENCHMARK_DATASET_ID,
            filename="manifest.json",
            repo_type="dataset",
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

    overall     = public.results.overall
    report_path = _write_temp_report(public, dataset_id)

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
        "report_path":      report_path,
    }


def run_audit(dataset_id: str, progress=gr.Progress(track_tqdm=True)):
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
        html = _render_cached_card(dataset_id, _CACHE[dataset_id])
        progress(1.0)
        return html, None

    # ── live audit ────────────────────────────────────────────────────────────
    progress(0.10, desc=f"Loading {dataset_id} ...")

    result: dict = {}
    error: list[str] = []

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
    html = _render_health_card(dataset_id, result)
    progress(1.0)
    return html, result.get("report_path")


# ── design tokens ─────────────────────────────────────────────────────────────

_SCORE_BANDS = [(90, "#22c55e"), (75, "#84cc16"), (60, "#f59e0b"), (40, "#f97316"), (0, "#ef4444")]

_CERT_TEXT = {
    "pass":        ("✓ Certified",               "#22c55e"),
    "provisional": ("~ Provisionally Certified",  "#f59e0b"),
    "fail":        ("✗ Not Certified",             "#ef4444"),
}

_SCORE_MEANING = [
    (90, "Excellent quality — ready for training."),
    (80, "Good quality — minor issues worth a quick review."),
    (70, "Generally usable — quality issues worth reviewing before training."),
    (60, "Moderate issues — consider selective episode retention."),
    (40, "Significant problems — pruning recommended before training."),
    (0,  "Severe quality issues — major cleanup required."),
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


def _percentile(score: float, distribution: list[float]) -> int:
    """Return percentile rank 0-100 (how many scores this beats)."""
    if not distribution:
        return 50
    return round(100 * sum(1 for s in distribution if score > s) / len(distribution))


# ── rendering helpers ─────────────────────────────────────────────────────────

def _checklist_html(findings: list, n_ep: int) -> str:
    problems = sorted(
        [f for f in findings if f.severity in ("critical", "warning")],
        key=lambda f: _SEV_ORDER.get(f.severity, 9),
    )[:8]

    if not problems:
        return (
            '<div style="display:flex;gap:10px;align-items:center">'
            '<span style="color:#22c55e;font-size:16px">✓</span>'
            '<span style="color:#cdd6f4;font-size:14px">All quality checks passed</span>'
            "</div>"
        )

    rows = []
    sev_icon  = {"critical": "⚠", "warning": "⚠"}
    sev_color = {"critical": "#ef4444", "warning": "#f59e0b"}
    for f in problems:
        color = sev_color.get(f.severity, "#6b7280")
        icon  = sev_icon.get(f.severity, "•")
        if f.affected_fraction is not None and n_ep > 0:
            n    = max(1, round(f.affected_fraction * n_ep))
            text = f"{n} episode{'s' if n != 1 else ''} — {f.message}"
        else:
            text = f.message
        rows.append(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin:5px 0">'
            f'<span style="color:{color};font-size:15px;min-width:16px;margin-top:1px">{icon}</span>'
            f'<span style="color:#cdd6f4;font-size:14px">{text}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def _percentile_section(score: float, dimensions: dict) -> str:
    if not _COMMUNITY_STATS:
        return ""

    overall_dist = _COMMUNITY_STATS.get("scores", [])
    dim_dists    = _COMMUNITY_STATS.get("dimension_distributions", {})
    n            = _COMMUNITY_STATS.get("n_datasets", 0)

    overall_pct  = _percentile(score, overall_dist)
    top_pct      = 100 - overall_pct

    # headline
    if top_pct <= 10:
        headline_text, headline_color = f"Top {top_pct}% of audited datasets", "#22c55e"
    elif top_pct <= 30:
        headline_text, headline_color = f"Top {top_pct}% of audited datasets", "#84cc16"
    elif top_pct <= 60:
        headline_text, headline_color = f"Better than {overall_pct}% of audited datasets", "#f59e0b"
    else:
        headline_text, headline_color = f"Bottom {100 - overall_pct}% of audited datasets", "#f97316"

    # per-dimension rows
    dim_rows = ""
    for dim_key, dim_result in sorted(dimensions.items()):
        dist  = dim_dists.get(dim_key, [])
        if not dist:
            continue
        pct   = _percentile(dim_result.score, dist)
        top   = 100 - pct
        color = _band_color(dim_result.score)
        label = _DIM_LABELS.get(dim_key, dim_key.replace("_", " ").title())
        if top <= 33:
            rank_str   = f"▲ top {top}%"
            rank_color = "#22c55e"
        elif top <= 66:
            rank_str   = f"≈ middle {100 - top - pct}%"
            rank_color = "#f59e0b"
        else:
            rank_str   = f"▼ bottom {pct}%"
            rank_color = "#ef4444"

        dim_rows += (
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;padding:4px 0;border-bottom:1px solid #313244">'
            f'<span style="font-size:13px;color:#a6adc8">{label}</span>'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<span style="font-size:13px;color:{color};font-weight:600">{dim_result.score:.0f}</span>'
            f'<span style="font-size:12px;color:{rank_color};min-width:90px;text-align:right">'
            f'{rank_str}</span>'
            f"</div></div>"
        )

    return f"""
<div style="border-top:1px solid #313244;margin:14px 0"></div>
<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Compared with community</div>
<div style="font-size:20px;font-weight:700;color:{headline_color};margin-bottom:12px">
  {headline_text}
</div>
{dim_rows}
<div style="font-size:11px;color:#45475a;margin-top:8px">
  Percentiles based on {n} audited public LeRobot datasets.
  Sample grows as the
  <a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"
     style="color:#6c7086">community benchmark</a> expands.
</div>
"""


def _dim_bars_html(dimensions: dict) -> str:
    rows = []
    for name, dim in sorted(dimensions.items()):
        color = _band_color(dim.score)
        label = _DIM_LABELS.get(name, name.replace("_", " ").title())
        rows.append(
            f'<div style="margin:5px 0">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:12px;color:#a6adc8;margin-bottom:3px">'
            f"<span>{label}</span>"
            f'<span style="color:{color};font-weight:600">{dim.score:.0f}</span></div>'
            f'<div style="background:#313244;border-radius:3px;height:4px">'
            f'<div style="background:{color};width:{int(dim.score)}%;height:4px;border-radius:3px">'
            f"</div></div></div>"
        )
    return "\n".join(rows)


def _recommend_html(score: float, findings: list, n_ep: int, n_total: int, is_sample: bool,
                    dataset_id: str) -> str:
    n_crit = sum(1 for f in findings if f.severity == "critical")
    if score >= 88 and n_crit == 0:
        keep_pct, strategy = 90, "light quality filter"
    elif score >= 75:
        keep_pct, strategy = 70, "quality filter"
    elif score >= 60:
        keep_pct, strategy = 50, "hybrid (quality + diversity)"
    elif score >= 40:
        keep_pct, strategy = 35, "hybrid (quality + diversity)"
    else:
        keep_pct, strategy = 20, "aggressive prune"

    keep_n      = max(1, round(n_total * keep_pct / 100))
    sample_note = (
        f'<span style="color:#6c7086;font-size:11px"> — estimate from {n_ep}/{n_total} ep sample</span>'
    ) if is_sample else ""

    return (
        f'<div style="font-size:15px;color:#cdd6f4;font-weight:500">'
        f"Keep ~{keep_pct}% &nbsp;·&nbsp; {keep_n:,} episodes &nbsp;·&nbsp; {strategy}"
        f"{sample_note}</div>"
        f'<div style="margin-top:10px;font-size:12px;color:#6c7086">'
        f'<code style="background:#313244;padding:3px 8px;border-radius:4px">'
        f"calibra prune hf://{dataset_id} --keep {keep_pct/100:.2f}</code></div>"
    )


def _card_wrapper(inner: str) -> str:
    return (
        '<div style="font-family:\'Inter\',\'Segoe UI\',sans-serif;background:#1e1e2e;'
        'border-radius:14px;padding:24px 28px;color:#cdd6f4;max-width:760px">'
        + inner + "</div>"
    )


def _render_health_card(dataset_id: str, r: dict) -> str:
    score      = r["score"]
    color      = _band_color(score)
    cert_label, cert_color = _CERT_TEXT.get(r["cert"], ("Unknown", "#6b7280"))
    n_ep       = r["n_episodes"]
    n_total    = r["n_episodes_total"]
    is_sample  = r["is_sample"]
    findings   = r["findings"]
    dimensions = r["dimensions"]
    meaning    = _score_meaning(score)

    sample_tag = (
        f' <span style="font-size:11px;background:#313244;padding:2px 8px;'
        f'border-radius:10px;color:#a6adc8">sample {n_ep}/{n_total} ep</span>'
    ) if is_sample else ""

    inner = f"""
<div style="font-size:11px;color:#6c7086;letter-spacing:.06em;margin-bottom:10px">
  ROBOT DATASET HEALTH CHECK
</div>

<div style="display:flex;align-items:flex-start;gap:24px;margin-bottom:18px">
  <div>
    <div>
      <span style="font-size:68px;font-weight:800;color:{color};line-height:1">{score:.0f}</span>
      <span style="font-size:18px;color:#6c7086">/100</span>
    </div>
    <div style="font-size:13px;color:#a6adc8;margin-top:4px;max-width:220px">{meaning}</div>
  </div>
  <div>
    <div style="font-size:38px;font-weight:700;color:{color}">{r["grade"]}</div>
    <div style="margin-top:4px">
      <span style="padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;
                   background:{cert_color}22;border:1px solid {cert_color};color:{cert_color}">
        {cert_label}
      </span>
    </div>
  </div>
  <div style="margin-left:auto;text-align:right;font-size:13px;color:#6c7086;line-height:2">
    <div><span style="color:#a6adc8">{dataset_id}</span>{sample_tag}</div>
    <div>{n_total:,} episodes &nbsp;·&nbsp; {r["n_samples"]:,} frames</div>
    <div>{r["fmt"]}</div>
  </div>
</div>

<div style="border-top:1px solid #313244;margin-bottom:14px"></div>

{_checklist_html(findings, n_ep)}

{_percentile_section(score, dimensions)}

<div style="border-top:1px solid #313244;margin:14px 0"></div>

<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Retention Recommendation</div>
{_recommend_html(score, findings, n_ep, n_total, is_sample, dataset_id)}

<div style="border-top:1px solid #313244;margin:16px 0 12px"></div>

<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Full Audit</div>
<div style="font-size:13px;color:#a6adc8;margin-bottom:6px">
  This demo analyzes up to {SAMPLE_EPISODE_CAP} episodes. For all episodes, per-episode verdicts,
  and a certifiable report:
</div>
<div style="background:#181825;border-radius:8px;padding:10px 14px;font-size:13px;
            font-family:monospace;color:#cdd6f4">
  pip install calibra-robotics<br>
  calibra audit hf://{dataset_id}
</div>

<div style="margin-top:16px;border-top:1px solid #313244;padding-top:10px;
            display:flex;justify-content:space-between;font-size:11px;color:#45475a">
  <span>Powered by <a href="https://github.com/omertt27/Calibra"
    style="color:#6c7086;text-decoration:none">Calibra</a></span>
  <span><a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"
    style="color:#6c7086;text-decoration:none">Community benchmark →</a></span>
</div>
"""
    return _card_wrapper(inner)


def _render_cached_card(dataset_id: str, cached: dict) -> str:
    score      = cached["score"]
    grade      = cached.get("grade", "?")
    cert       = cached.get("certification", "")
    color      = _band_color(score)
    cert_label, cert_color = _CERT_TEXT.get(cert, ("Unknown", "#6b7280"))
    n_ep       = cached.get("n_episodes") or 0
    n_frames   = cached.get("n_frames") or 0
    n_crit     = cached.get("n_critical") or 0
    meaning    = _score_meaning(score)

    # build fake dimension objects for percentile section
    class _Dim:
        def __init__(self, s): self.score = s

    dim_objs = {
        k: _Dim(v["score"])
        for k, v in (cached.get("dimensions") or {}).items()
        if isinstance(v, dict) and v.get("score") is not None
    }

    inner = f"""
<div style="font-size:11px;color:#6c7086;letter-spacing:.06em;margin-bottom:10px">
  ROBOT DATASET HEALTH CHECK &nbsp;·&nbsp;
  <span style="background:#313244;padding:2px 8px;border-radius:10px">full audit cached</span>
</div>

<div style="display:flex;align-items:flex-start;gap:24px;margin-bottom:18px">
  <div>
    <div>
      <span style="font-size:68px;font-weight:800;color:{color};line-height:1">{score:.0f}</span>
      <span style="font-size:18px;color:#6c7086">/100</span>
    </div>
    <div style="font-size:13px;color:#a6adc8;margin-top:4px;max-width:220px">{meaning}</div>
  </div>
  <div>
    <div style="font-size:38px;font-weight:700;color:{color}">{grade}</div>
    <div style="margin-top:4px">
      <span style="padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;
                   background:{cert_color}22;border:1px solid {cert_color};color:{cert_color}">
        {cert_label}
      </span>
    </div>
  </div>
  <div style="margin-left:auto;text-align:right;font-size:13px;color:#6c7086;line-height:2">
    <div><span style="color:#a6adc8">{dataset_id}</span></div>
    <div>{n_ep:,} episodes &nbsp;·&nbsp; {n_frames:,} frames</div>
    <div style="color:#ef4444">{n_crit} critical finding{'s' if n_crit != 1 else ''}</div>
  </div>
</div>

<div style="border-top:1px solid #313244;margin-bottom:14px"></div>

<div style="font-size:14px;color:#a6adc8">
  Full per-episode audit results in the
  <a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"
     style="color:#89b4fa">community benchmark dataset →</a>
</div>

{_percentile_section(score, dim_objs)}

<div style="border-top:1px solid #313244;margin:14px 0"></div>

<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Full Audit</div>
<div style="background:#181825;border-radius:8px;padding:10px 14px;font-size:13px;
            font-family:monospace;color:#cdd6f4">
  pip install calibra-robotics<br>
  calibra audit hf://{dataset_id}
</div>

<div style="margin-top:16px;border-top:1px solid #313244;padding-top:10px;
            display:flex;justify-content:space-between;font-size:11px;color:#45475a">
  <span>Powered by <a href="https://github.com/omertt27/Calibra"
    style="color:#6c7086;text-decoration:none">Calibra</a></span>
  <span><a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"
    style="color:#6c7086;text-decoration:none">Community benchmark →</a></span>
</div>
"""
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

Enter a dataset ID and get a health score, concrete quality findings,
community percentile comparison, and a keep-fraction recommendation.
Pre-audited datasets return instantly from the community benchmark cache.
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

    gr.Examples(examples=EXAMPLES, inputs=inp, label="Try these")

    gr.Markdown("""
---
**What gets checked**

| Dimension | Checks |
|-----------|--------|
| Temporal Consistency | Frame dropout rate, timestamp jitter, synchronization lag |
| Motion Quality | Action jerk spikes, velocity discontinuities, smoothness (LDLJ) |
| Behavioral Coverage | Trajectory diversity, redundancy fraction, entropy |
| Task Integrity | Episode length distribution, phase balance, inactivity periods |

**Run a full audit locally** (all episodes, per-episode verdicts, certifiable report):
```bash
pip install calibra-robotics
calibra audit hf://lerobot/pusht
```

*Powered by [Calibra](https://github.com/omertt27/Calibra) — open-source dataset quality \
tooling for robotics imitation learning*
""")

    btn.click(fn=run_audit, inputs=inp, outputs=[out_html, out_download])
    inp.submit(fn=run_audit, inputs=inp, outputs=[out_html, out_download])

if __name__ == "__main__":
    demo.launch()
