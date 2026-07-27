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
SAMPLE_EPISODE_CAP = 20       # max episodes analyzed in the demo
AUDIT_TIMEOUT_S    = 90       # seconds before we give up and ask user to run CLI
BENCHMARK_DATASET_ID = "omert27/calibra-robot-dataset-quality-benchmark"

# ── community stats (loaded at Space startup) ─────────────────────────────────
_COMMUNITY_STATS: Optional[dict] = None
_CACHE: dict[str, dict] = {}   # dataset_id → cached result dict


def _boot() -> None:
    """Attempt to load benchmark data at startup (fails silently if not yet published)."""
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


# ── audit runner ──────────────────────────────────────────────────────────────

def _hf_revision(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi
        return getattr(HfApi().dataset_info(repo_id=repo_id), "sha", None)
    except Exception:
        return None


def _run_sample_audit(dataset_id: str) -> dict:
    """
    Load the dataset, cap at SAMPLE_EPISODE_CAP episodes, run the pipeline.
    Returns a dict with keys: score, grade, cert, n_episodes, n_episodes_total,
    n_samples, fmt, findings, dimensions, report_json_path.
    """
    from calibra.ingestion.registry import load
    from calibra.pipeline import Pipeline
    from calibra.report_json import assemble_public_report
    from calibra.schema.public_report import DatasetInfo, SamplingConfig

    revision = _hf_revision(dataset_id)

    # Load full batch, then cap episodes before running the pipeline
    batch = load(dataset_id)
    n_total_episodes = batch.n_episodes

    is_sample = n_total_episodes > SAMPLE_EPISODE_CAP
    if is_sample:
        batch.episodes = batch.episodes[:SAMPLE_EPISODE_CAP]
        batch._n_samples_hint = None  # recompute from truncated list

    diag = Pipeline().run(batch)

    dataset_info = DatasetInfo(
        provider="huggingface",
        repository_id=dataset_id,
        revision=revision,
        dataset_format=diag.format,
        episodes_total=n_total_episodes,
        episodes_audited=diag.n_episodes,
        frames_total=diag.n_samples,
    )
    public = assemble_public_report(
        diag,
        dataset_info=dataset_info,
        sampling=SamplingConfig(
            mode="random" if is_sample else "full",
            fraction=diag.n_episodes / n_total_episodes if is_sample else 1.0,
        ),
    )

    overall = public.results.overall
    report_path = _write_temp_report(public, dataset_id)

    return {
        "score": overall.score,
        "grade": overall.grade,
        "cert": overall.certification,
        "n_episodes": diag.n_episodes,
        "n_episodes_total": n_total_episodes,
        "n_samples": diag.n_samples,
        "fmt": diag.format,
        "findings": public.results.findings,
        "dimensions": public.results.dimensions,
        "is_sample": is_sample,
        "report_path": report_path,
    }


def run_audit(dataset_id: str, progress=gr.Progress(track_tqdm=True)):
    dataset_id = dataset_id.strip()

    if not dataset_id:
        raise gr.Error("Enter a dataset ID, e.g. lerobot/pusht")

    parts = dataset_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise gr.Error(
            f"'{dataset_id}' is not a valid Hugging Face dataset ID. "
            "Expected format: org/name  (e.g. lerobot/pusht)"
        )

    # ── check cache ───────────────────────────────────────────────────────────
    if dataset_id in _CACHE:
        progress(0.3, desc="Found in community benchmark cache ...")
        cached = _CACHE[dataset_id]
        html = _render_cached_card(dataset_id, cached)
        progress(1.0)
        return html, None

    # ── live audit with timeout ───────────────────────────────────────────────
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

    # Poll with progress ticks
    elapsed = 0
    tick = 3
    while thread.is_alive() and elapsed < AUDIT_TIMEOUT_S:
        thread.join(timeout=tick)
        elapsed += tick
        pct = min(0.15 + (elapsed / AUDIT_TIMEOUT_S) * 0.75, 0.90)
        progress(pct, desc=f"Auditing sample ({elapsed}s) ...")

    if thread.is_alive():
        raise gr.Error(
            f"Audit timed out after {AUDIT_TIMEOUT_S}s. "
            "The dataset may be too large for the demo. "
            f"Run locally: calibra audit hf://{dataset_id}"
        )

    if error:
        msg = error[0]
        if "not found" in msg.lower() or "404" in msg:
            raise gr.Error(
                f"Dataset '{dataset_id}' not found. Check the ID and try again."
            )
        if "lerobot" in msg.lower() or "parquet" in msg.lower() or "episode" in msg.lower():
            raise gr.Error(
                f"'{dataset_id}' does not appear to be a LeRobot dataset. "
                "This tool audits LeRobot-format datasets only."
            )
        raise gr.Error(f"Audit failed: {msg[:300]}")

    progress(0.95, desc="Rendering results ...")
    html = _render_health_card(dataset_id, result)
    progress(1.0)
    return html, result.get("report_path")


# ── rendering ─────────────────────────────────────────────────────────────────

_SCORE_BANDS = [(80, "#22c55e"), (60, "#f59e0b"), (40, "#f97316"), (0, "#ef4444")]
_CERT_TEXT   = {
    "pass":        ("✓ Certified",              "#22c55e"),
    "provisional": ("~ Provisionally Certified", "#f59e0b"),
    "fail":        ("✗ Not Certified",            "#ef4444"),
}
_SEV_ORDER = {"critical": 0, "warning": 1, "ok": 2, "info": 3}
_SEV_ICON  = {"critical": "⚠", "warning": "⚠", "ok": "✓", "info": "ℹ"}
_SEV_COLOR = {
    "critical": "#ef4444", "warning": "#f59e0b",
    "ok": "#22c55e",       "info": "#6b7280",
}


def _band_color(score: float) -> str:
    for threshold, color in _SCORE_BANDS:
        if score >= threshold:
            return color
    return "#ef4444"


def _checklist_html(findings: list, n_episodes: int) -> str:
    problems = sorted(
        [f for f in findings if f.severity in ("critical", "warning")],
        key=lambda f: _SEV_ORDER.get(f.severity, 9),
    )[:8]

    if not problems:
        return (
            '<div style="display:flex;gap:10px;align-items:center;margin:4px 0">'
            '<span style="color:#22c55e;font-size:16px">✓</span>'
            '<span style="color:#cdd6f4;font-size:14px">All quality checks passed</span>'
            "</div>"
        )

    rows = []
    for f in problems:
        icon  = _SEV_ICON[f.severity]
        color = _SEV_COLOR[f.severity]
        if f.affected_fraction is not None and n_episodes > 0:
            n = max(1, round(f.affected_fraction * n_episodes))
            text = f"{n} episode{'s' if n != 1 else ''} — {f.message}"
        else:
            text = f.message
        rows.append(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin:5px 0">'
            f'<span style="color:{color};font-size:16px;min-width:16px">{icon}</span>'
            f'<span style="color:#cdd6f4;font-size:14px">{text}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def _dim_bars_html(dimensions: dict) -> str:
    rows = []
    for name, dim in sorted(dimensions.items())[:6]:
        color     = _band_color(dim.score)
        bar_width = int(dim.score)
        label     = name.replace("_", " ").title()
        rows.append(
            f'<div style="margin:5px 0">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:12px;color:#a6adc8;margin-bottom:3px">'
            f"<span>{label}</span>"
            f'<span style="color:{color};font-weight:600">{dim.score:.0f}</span></div>'
            f'<div style="background:#313244;border-radius:3px;height:4px">'
            f'<div style="background:{color};width:{bar_width}%;height:4px;border-radius:3px"></div>'
            f"</div></div>"
        )
    return "\n".join(rows)


def _recommend_html(score: float, findings: list, n_ep: int, n_total: int, is_sample: bool) -> str:
    n_crit = sum(1 for f in findings if f.severity == "critical")
    if score >= 88 and n_crit == 0:
        keep_pct, strategy = 90, "Light quality filter"
    elif score >= 75:
        keep_pct, strategy = 70, "Quality filter"
    elif score >= 60:
        keep_pct, strategy = 50, "Hybrid (quality + diversity)"
    elif score >= 40:
        keep_pct, strategy = 35, "Hybrid (quality + diversity)"
    else:
        keep_pct, strategy = 20, "Aggressive prune"

    keep_n = max(1, round(n_total * keep_pct / 100))
    sample_note = (
        f'<div style="font-size:11px;color:#6c7086;margin-top:3px">'
        f"Sample audit ({n_ep} of {n_total} episodes) — "
        f"estimate based on sampled data</div>"
    ) if is_sample else ""

    return (
        f'<div style="font-size:15px;color:#cdd6f4;font-weight:500">'
        f"Keep ~{keep_pct}% &nbsp;·&nbsp; {keep_n:,} episodes &nbsp;·&nbsp; {strategy}"
        f"</div>"
        f"{sample_note}"
        f'<div style="margin-top:8px;font-size:12px;color:#6c7086">'
        f'<code style="background:#313244;padding:2px 8px;border-radius:4px">'
        f"calibra prune hf://{'{dataset_id}'} --keep {keep_pct/100:.2f}</code></div>"
    )


def _percentile_html(score: float) -> str:
    if not _COMMUNITY_STATS:
        return ""
    scores = _COMMUNITY_STATS.get("scores", [])
    if not scores:
        return ""
    pct = round(100 * sum(1 for s in scores if score > s) / len(scores))
    top = 100 - pct
    label = f"Top {top}%" if top <= 50 else f"Bottom {100 - top}%"
    n = len(scores)
    return (
        f'<div style="margin-top:10px;font-size:13px;color:#a6adc8">'
        f'Community: <strong style="color:#cdd6f4">{label}</strong>'
        f' of {n} audited datasets &nbsp;·&nbsp; '
        f'mean score {_COMMUNITY_STATS.get("mean", "—")}</div>'
    )


def _card_wrapper(inner: str) -> str:
    return (
        '<div style="font-family:\'Inter\',sans-serif;background:#1e1e2e;'
        'border-radius:14px;padding:24px;color:#cdd6f4;max-width:720px">'
        + inner
        + "</div>"
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

    sample_tag = (
        f'<span style="font-size:11px;background:#313244;padding:2px 8px;'
        f'border-radius:10px;color:#a6adc8;margin-left:8px">'
        f"sample ({n_ep}/{n_total} ep)</span>"
    ) if is_sample else ""

    inner = f"""
<div style="font-size:11px;color:#6c7086;letter-spacing:.06em;margin-bottom:10px">
  ROBOT DATASET HEALTH CHECK
</div>

<div style="display:flex;align-items:center;gap:20px;margin-bottom:18px">
  <div>
    <span style="font-size:68px;font-weight:800;color:{color};line-height:1">{score:.0f}</span>
    <span style="font-size:18px;color:#6c7086">/100</span>
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
  <div style="margin-left:auto;text-align:right;font-size:13px;color:#6c7086;line-height:1.9">
    <div><span style="color:#a6adc8">{dataset_id}</span>{sample_tag}</div>
    <div>{n_total:,} episodes &nbsp;·&nbsp; {r["n_samples"]:,} frames &nbsp;·&nbsp; {r["fmt"]}</div>
    {_percentile_html(score)}
  </div>
</div>

<div style="border-top:1px solid #313244;margin-bottom:14px"></div>

{_checklist_html(findings, n_ep)}

<div style="border-top:1px solid #313244;margin:14px 0"></div>

<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:6px">Recommendation</div>
{_recommend_html(score, findings, n_ep, n_total, is_sample)}

<div style="border-top:1px solid #313244;margin:14px 0"></div>

<div style="font-size:11px;color:#6c7086;text-transform:uppercase;letter-spacing:.06em;
            margin-bottom:8px">Dimensions</div>
{_dim_bars_html(dimensions)}

<div style="border-top:1px solid #313244;margin-top:16px;padding-top:10px;
            display:flex;justify-content:space-between;font-size:11px;color:#45475a">
  <span>Powered by <a href="https://github.com/omertt27/Calibra"
    style="color:#6c7086;text-decoration:none">Calibra</a>
    &nbsp;·&nbsp; Demo uses sample of up to {SAMPLE_EPISODE_CAP} episodes
  </span>
  <span><code style="background:#313244;padding:1px 6px;border-radius:3px">
    calibra audit hf://{dataset_id}</code> for full audit</span>
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
    pct        = _percentile_html(score)

    inner = f"""
<div style="font-size:11px;color:#6c7086;letter-spacing:.06em;margin-bottom:10px">
  ROBOT DATASET HEALTH CHECK &nbsp;·&nbsp;
  <span style="background:#313244;padding:2px 8px;border-radius:10px">
    from community benchmark
  </span>
</div>

<div style="display:flex;align-items:center;gap:20px;margin-bottom:18px">
  <div>
    <span style="font-size:68px;font-weight:800;color:{color};line-height:1">{score:.0f}</span>
    <span style="font-size:18px;color:#6c7086">/100</span>
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
  <div style="margin-left:auto;text-align:right;font-size:13px;color:#6c7086;line-height:1.9">
    <div><span style="color:#a6adc8">{dataset_id}</span></div>
    <div>{n_ep:,} episodes &nbsp;·&nbsp; {n_frames:,} frames</div>
    <div style="color:#ef4444">{n_crit} critical finding{'s' if n_crit != 1 else ''}</div>
    {pct}
  </div>
</div>

<div style="border-top:1px solid #313244;margin:14px 0"></div>

<div style="font-size:13px;color:#a6adc8">
  Full audit results available in the
  <a href="https://huggingface.co/datasets/{BENCHMARK_DATASET_ID}"
     style="color:#89b4fa">community benchmark dataset</a>.
</div>

<div style="border-top:1px solid #313244;margin-top:16px;padding-top:10px;
            font-size:11px;color:#45475a;display:flex;justify-content:space-between">
  <span>Powered by <a href="https://github.com/omertt27/Calibra"
    style="color:#6c7086;text-decoration:none">Calibra</a></span>
  <span><code style="background:#313244;padding:1px 6px;border-radius:3px">
    calibra audit hf://{dataset_id}</code> for full local audit</span>
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
    gr.Markdown(
        """
# Robot Dataset Health Check

**Audit any LeRobot dataset before training.**

Enter a dataset ID and get a health score, concrete findings, and a
keep-fraction recommendation in ~30 seconds.
Public demo uses a sample of up to 20 episodes. For a full audit, run locally.
"""
    )

    with gr.Row():
        inp = gr.Textbox(
            label="LeRobot Dataset ID",
            placeholder="lerobot/pusht",
            scale=5,
        )
        btn = gr.Button("Check Health", variant="primary", scale=1, min_width=140)

    out_html     = gr.HTML()
    out_download = gr.File(label="Download Full Report (JSON)", visible=True)

    gr.Examples(examples=EXAMPLES, inputs=inp, label="Try these")

    gr.Markdown(
        """
---
**What gets checked:** frame dropout · timestamp jitter · action jerk spikes ·
velocity discontinuities · trajectory redundancy · behavioral coverage ·
episode length distribution · phase balance

**Full audit locally:**
```bash
pip install 'calibra-robotics[lerobot]'
calibra audit hf://lerobot/pusht
```

*Powered by [Calibra](https://github.com/omertt27/Calibra) — open-source dataset
quality tooling for robotics imitation learning*
"""
    )

    btn.click(fn=run_audit, inputs=inp, outputs=[out_html, out_download])
    inp.submit(fn=run_audit, inputs=inp, outputs=[out_html, out_download])

if __name__ == "__main__":
    demo.launch()
