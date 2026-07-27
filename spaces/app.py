"""
Robot Dataset Health Check — powered by Calibra

Enter any LeRobot dataset ID. Get a health score, concrete findings,
a keep-fraction recommendation, and a downloadable report in ~30 seconds.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

import gradio as gr

# ── community benchmark stats (pre-loaded at startup) ─────────────────────────
# Populated from omertt27/calibra-robot-dataset-quality-benchmark once published.
# Until then falls back to None (percentile display is hidden).
_COMMUNITY_STATS: Optional[dict] = None

BENCHMARK_DATASET_ID = "omertt27/calibra-robot-dataset-quality-benchmark"


def _load_community_stats() -> Optional[dict]:
    """Try to fetch pre-computed community percentile stats from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=BENCHMARK_DATASET_ID,
            filename="community_stats.json",
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── audit runner ──────────────────────────────────────────────────────────────


def _hf_revision(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id=repo_id)
        return getattr(info, "sha", None)
    except Exception:
        return None


def run_audit(dataset_id: str, progress=gr.Progress(track_tqdm=True)):
    dataset_id = dataset_id.strip()

    if not dataset_id:
        raise gr.Error("Please enter a dataset ID, e.g. lerobot/pusht")

    parts = dataset_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise gr.Error(
            f"'{dataset_id}' is not a valid Hugging Face dataset ID. "
            "Expected format: org/name (e.g. lerobot/pusht)"
        )

    progress(0.05, desc="Connecting to Hugging Face Hub ...")
    revision = _hf_revision(dataset_id)

    progress(0.15, desc=f"Loading {dataset_id} ...")

    try:
        from calibra.pipeline import Pipeline
        from calibra.report_json import assemble_public_report
        from calibra.schema.public_report import DatasetInfo, SamplingConfig
    except ImportError as exc:
        raise gr.Error(f"Calibra import error: {exc}") from exc

    progress(0.20, desc="Running quality checks ...")

    try:
        diag = Pipeline().analyze_path(dataset_id)
    except Exception as exc:
        msg = str(exc)
        if "not found" in msg.lower() or "404" in msg:
            raise gr.Error(
                f"Dataset '{dataset_id}' not found on Hugging Face Hub. "
                "Verify the ID and try again."
            ) from exc
        raise gr.Error(f"Audit failed: {msg[:300]}") from exc

    progress(0.85, desc="Assembling report ...")

    dataset_info = DatasetInfo(
        provider="huggingface",
        repository_id=dataset_id,
        revision=revision,
        dataset_format=diag.format,
        episodes_total=diag.n_episodes,
        episodes_audited=diag.n_episodes,
        frames_total=diag.n_samples,
    )
    public = assemble_public_report(
        diag,
        dataset_info=dataset_info,
        sampling=SamplingConfig(mode="full"),
    )

    progress(0.95, desc="Formatting results ...")

    overall = public.results.overall
    findings = public.results.findings
    dimensions = public.results.dimensions

    health_html = _render_health_card(
        dataset_id=dataset_id,
        score=overall.score,
        grade=overall.grade,
        cert=overall.certification,
        n_episodes=diag.n_episodes,
        n_samples=diag.n_samples,
        findings=findings,
        dimensions=dimensions,
    )

    report_path = _write_temp_report(public, dataset_id)

    progress(1.0, desc="Done.")
    return health_html, report_path


# ── rendering ─────────────────────────────────────────────────────────────────

_SCORE_THRESHOLDS = [(80, "#22c55e"), (60, "#f59e0b"), (40, "#f97316"), (0, "#ef4444")]

_CERT_TEXT = {
    "pass": ("✓ Certified", "#22c55e"),
    "provisional": ("~ Provisional", "#f59e0b"),
    "fail": ("✗ Not Certified", "#ef4444"),
}

_SEV_ORDER = {"critical": 0, "warning": 1, "ok": 2, "info": 3}
_SEV_ICON = {"critical": "⚠", "warning": "⚠", "ok": "✓", "info": "✓"}
_SEV_COLOR = {"critical": "#ef4444", "warning": "#f59e0b", "ok": "#22c55e", "info": "#6b7280"}


def _score_color(score: float) -> str:
    for threshold, color in _SCORE_THRESHOLDS:
        if score >= threshold:
            return color
    return "#ef4444"


def _checklist_items(findings: list, n_episodes: int) -> list[tuple[str, str, str]]:
    """Return (icon, text, color) tuples for the checklist display."""
    items: list[tuple[str, str, str]] = []

    sorted_findings = sorted(
        [f for f in findings if f.severity in ("critical", "warning")],
        key=lambda f: _SEV_ORDER.get(f.severity, 9),
    )

    for f in sorted_findings[:8]:
        icon = _SEV_ICON[f.severity]
        color = _SEV_COLOR[f.severity]

        if f.affected_fraction is not None and n_episodes > 0:
            n = max(1, round(f.affected_fraction * n_episodes))
            text = f"{n} episode{'s' if n != 1 else ''} — {f.message}"
        else:
            text = f.message

        items.append((icon, text, color))

    if not items:
        items.append(("✓", "All quality checks passed", "#22c55e"))

    return items


def _keep_recommendation(score: float, findings: list, n_episodes: int) -> str:
    """Derive a simple keep-fraction recommendation from score and findings."""
    n_critical = sum(1 for f in findings if f.severity == "critical")
    n_warning = sum(1 for f in findings if f.severity == "warning")

    if score >= 88 and n_critical == 0:
        keep_pct = 90
        strategy = "Light quality filter"
    elif score >= 75:
        keep_pct = 70
        strategy = "Quality filter"
    elif score >= 60:
        keep_pct = 50
        strategy = "Hybrid (quality + diversity)"
    elif score >= 40:
        keep_pct = 35
        strategy = "Hybrid (quality + diversity)"
    else:
        keep_pct = 20
        strategy = "Aggressive prune (quality + diversity)"

    keep_n = max(1, round(n_episodes * keep_pct / 100))
    return f"Keep ~{keep_pct}% ({keep_n} episodes) &nbsp;·&nbsp; Strategy: {strategy}"


def _percentile_html(score: float) -> str:
    """Show community percentile if benchmark stats are loaded."""
    if _COMMUNITY_STATS is None:
        return ""
    scores = _COMMUNITY_STATS.get("scores", [])
    if not scores:
        return ""
    pct = round(100 * sum(1 for s in scores if score > s) / len(scores))
    label = f"Top {100 - pct}%" if pct >= 50 else f"Bottom {pct}%"
    return (
        f'<div style="margin-top:10px;font-size:13px;color:#a6adc8">'
        f'Community: <strong style="color:#cdd6f4">{label}</strong> '
        f'of {len(scores)} audited datasets</div>'
    )


def _render_health_card(
    dataset_id: str,
    score: float,
    grade: str,
    cert: str,
    n_episodes: int,
    n_samples: int,
    findings: list,
    dimensions: dict,
) -> str:
    color = _score_color(score)
    cert_label, cert_color = _CERT_TEXT.get(cert, ("Unknown", "#6b7280"))
    items = _checklist_items(findings, n_episodes)
    recommendation = _keep_recommendation(score, findings, n_episodes)
    percentile = _percentile_html(score)

    checklist_html = "\n".join(
        f'<div style="display:flex;align-items:flex-start;gap:10px;margin:5px 0">'
        f'<span style="color:{c};font-size:16px;min-width:16px">{icon}</span>'
        f'<span style="color:#cdd6f4;font-size:14px">{text}</span>'
        f"</div>"
        for icon, text, c in items
    )

    dim_bars = ""
    for name, dim in sorted(dimensions.items())[:5]:
        dim_color = _score_color(dim.score)
        bar_width = int(dim.score)
        label = name.replace("_", " ").title()
        dim_bars += (
            f'<div style="margin:6px 0">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:12px;color:#a6adc8;margin-bottom:3px">'
            f"<span>{label}</span><span style='color:{dim_color}'>{dim.score:.0f}</span></div>"
            f'<div style="background:#313244;border-radius:3px;height:5px">'
            f'<div style="background:{dim_color};width:{bar_width}%;height:5px;border-radius:3px"></div>'
            f"</div></div>"
        )

    return f"""
<div style="font-family:'Inter',sans-serif;background:#1e1e2e;border-radius:14px;
            padding:24px;color:#cdd6f4;max-width:700px">

  <!-- header -->
  <div style="font-size:12px;color:#6c7086;margin-bottom:12px;letter-spacing:0.05em">
    ROBOT DATASET HEALTH CHECK
  </div>

  <!-- score row -->
  <div style="display:flex;align-items:center;gap:20px;margin-bottom:20px">
    <div>
      <span style="font-size:64px;font-weight:800;color:{color};line-height:1">{score:.0f}</span>
      <span style="font-size:20px;color:#6c7086;margin-left:2px">/100</span>
    </div>
    <div>
      <div style="font-size:36px;font-weight:700;color:{color}">{grade}</div>
      <div style="margin-top:4px">
        <span style="padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;
                     background:{cert_color}22;border:1px solid {cert_color};color:{cert_color}">
          {cert_label}
        </span>
      </div>
    </div>
    <div style="margin-left:auto;text-align:right;font-size:13px;color:#6c7086;line-height:2">
      <div><span style="color:#a6adc8">{dataset_id}</span></div>
      <div>{n_episodes:,} episodes &nbsp;·&nbsp; {n_samples:,} frames</div>
    </div>
  </div>

  <!-- divider -->
  <div style="border-top:1px solid #313244;margin-bottom:16px"></div>

  <!-- checklist -->
  <div style="margin-bottom:16px">
    {checklist_html}
  </div>

  <!-- divider -->
  <div style="border-top:1px solid #313244;margin-bottom:16px"></div>

  <!-- recommendation -->
  <div style="font-size:13px;color:#6c7086;margin-bottom:4px;text-transform:uppercase;
              letter-spacing:0.05em">Recommendation</div>
  <div style="font-size:15px;color:#cdd6f4;font-weight:500">{recommendation}</div>
  <div style="font-size:12px;color:#6c7086;margin-top:4px">
    Run <code style="background:#313244;padding:2px 6px;border-radius:4px">
    calibra prune {dataset_id} --keep 0.35</code> to apply
  </div>

  <!-- percentile (only when benchmark data is loaded) -->
  {percentile}

  <!-- divider -->
  <div style="border-top:1px solid #313244;margin:16px 0"></div>

  <!-- dimension bars -->
  <div style="font-size:13px;color:#6c7086;margin-bottom:8px;text-transform:uppercase;
              letter-spacing:0.05em">Dimensions</div>
  {dim_bars}

  <!-- footer -->
  <div style="margin-top:20px;border-top:1px solid #313244;padding-top:12px;
              font-size:11px;color:#45475a;display:flex;justify-content:space-between">
    <span>Powered by <a href="https://github.com/omertt27/Calibra"
      style="color:#6c7086;text-decoration:none">Calibra</a></span>
    <span>calibra audit {dataset_id}</span>
  </div>
</div>
"""


def _write_temp_report(public, dataset_id: str) -> str:
    slug = dataset_id.replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"calibra_{slug}_{ts}.json"
    path = os.path.join(tempfile.gettempdir(), fname)
    public.write(path)
    return path


# ── load community stats at startup ───────────────────────────────────────────
_COMMUNITY_STATS = _load_community_stats()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["lerobot/pusht"],
    ["lerobot/aloha_sim_insertion_human"],
    ["lerobot/xarm_lift_medium"],
    ["lerobot/aloha_static_coffee"],
    ["lerobot/unitreeh1_fold_clothes"],
]

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

**Audit any LeRobot dataset in ~30 seconds.**

Checks timing integrity, control smoothness, trajectory diversity, and more.
Gives you a 0–100 health score, concrete findings, and a keep-fraction recommendation.
"""
    )

    with gr.Row():
        dataset_input = gr.Textbox(
            label="LeRobot Dataset ID",
            placeholder="lerobot/pusht",
            scale=5,
        )
        audit_btn = gr.Button("Check Health", variant="primary", scale=1, min_width=140)

    health_display = gr.HTML()
    report_download = gr.File(label="Download Full Report (JSON)", visible=True)

    gr.Examples(
        examples=EXAMPLES,
        inputs=dataset_input,
        label="Try these",
    )

    gr.Markdown(
        """
---
**What gets checked:** frame dropout · timestamp jitter · action jerk spikes ·
velocity discontinuities · trajectory redundancy · behavioral coverage ·
episode length distribution · phase balance

Run locally: &nbsp; `pip install 'calibra-robotics[lerobot]'` &nbsp; → &nbsp; `calibra audit lerobot/pusht`

*Powered by [Calibra](https://github.com/omertt27/Calibra) · open-source dataset quality tooling for robotics*
"""
    )

    audit_btn.click(
        fn=run_audit,
        inputs=dataset_input,
        outputs=[health_display, report_download],
    )
    dataset_input.submit(
        fn=run_audit,
        inputs=dataset_input,
        outputs=[health_display, report_download],
    )

if __name__ == "__main__":
    demo.launch()
