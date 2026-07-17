"""
calibra site — static site generator.

Reads the results/ directory tree (CalibraReport JSON files produced by
audit-all) and generates a self-contained static website with no calibra
Python imports required by any downstream consumer.

Output:
  site/index.html                  — sortable, filterable dataset leaderboard
  site/<org>/<slug>/index.html     — per-dataset detail page
  site/<org>/<slug>/badge.svg      — embeddable shields.io-style quality badge
  site/<org>/<slug>/history.json   — score history across dataset revisions

Usage:
    calibra site --results ./results --out ./site
    calibra site --results ./results --out ./site --title "Calibra Leaderboard"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── score / grade / cert helpers ─────────────────────────────────────────────

def _score_hex(score: float) -> str:
    if score >= 90: return "#22c55e"
    if score >= 80: return "#10b981"
    if score >= 70: return "#f59e0b"
    if score >= 60: return "#f97316"
    return "#ef4444"


def _cert_hex(cert: str) -> str:
    return {"pass": "#22c55e", "provisional": "#f59e0b", "fail": "#ef4444"}.get(cert, "#64748b")


def _cert_label(cert: str) -> str:
    return {"pass": "Certified", "provisional": "Provisional", "fail": "Not Certified"}.get(cert, cert)


def _severity_hex(sev: str) -> str:
    return {"critical": "#ef4444", "warning": "#f59e0b", "info": "#6366f1", "ok": "#22c55e"}.get(sev, "#64748b")


def _policy_hex(status: str) -> str:
    return {"recommended": "#22c55e", "review": "#f59e0b", "not_recommended": "#ef4444"}.get(status, "#64748b")


def _policy_label(status: str) -> str:
    return {"recommended": "Recommended", "review": "Review", "not_recommended": "Not Recommended"}.get(status, status)


def _dim_label(key: str) -> str:
    return {
        "temporal_integrity": "Temporal",
        "motion_quality": "Motion",
        "behavioral_coverage": "Coverage",
        "task_integrity": "Task",
    }.get(key, key.replace("_", " ").title())


# ── data loading ─────────────────────────────────────────────────────────────

def _scan_results(results_dir: Path) -> list[dict]:
    """Load all latest.json files, return sorted by score descending."""
    reports = []
    for latest in sorted(results_dir.rglob("latest.json")):
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            if data.get("status") == "failed":
                continue
            reports.append(data)
        except Exception:
            continue
    reports.sort(key=lambda r: r.get("results", {}).get("overall", {}).get("score", 0), reverse=True)
    return reports


def _collect_history(results_dir: Path, repo_id: str) -> list[dict]:
    """Collect score history across all revision directories."""
    dataset_dir = results_dir / repo_id
    history = []
    if not dataset_dir.exists():
        return history
    for rev_dir in sorted(dataset_dir.iterdir()):
        if not rev_dir.is_dir() or rev_dir.name == "latest.json":
            continue
        for json_file in sorted(rev_dir.glob("????????T??????Z.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                overall = data.get("results", {}).get("overall", {})
                history.append({
                    "revision": rev_dir.name,
                    "timestamp": data.get("report", {}).get("generated_at", ""),
                    "score": overall.get("score"),
                    "grade": overall.get("grade"),
                    "calibra_version": data.get("report", {}).get("calibra_version", ""),
                })
            except Exception:
                continue
    return sorted(history, key=lambda h: h["timestamp"])


# ── badge SVG ─────────────────────────────────────────────────────────────────

def _badge_svg(score: float, grade: str, cert: str) -> str:
    score_text = f"{score:.0f} {grade}"
    score_color = _score_hex(score)
    label_w = 56
    value_w = max(40, len(score_text) * 7 + 10)
    total_w = label_w + value_w
    lx = label_w // 2
    rx = label_w + value_w // 2

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" role="img" '
        f'aria-label="Calibra: {score_text}">'
        f'<title>Calibra: {score_text}</title>'
        f'<linearGradient id="s" x2="0" y2="100%">'
        f'<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        f'<stop offset="1" stop-opacity=".1"/>'
        f'</linearGradient>'
        f'<clipPath id="r"><rect width="{total_w}" height="20" rx="3"/></clipPath>'
        f'<g clip-path="url(#r)">'
        f'<rect width="{label_w}" height="20" fill="#555"/>'
        f'<rect x="{label_w}" width="{value_w}" height="20" fill="{score_color}"/>'
        f'<rect width="{total_w}" height="20" fill="url(#s)"/>'
        f'</g>'
        f'<g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">'
        f'<text x="{lx}" y="15" fill="#010101" fill-opacity=".3">Calibra</text>'
        f'<text x="{lx}" y="14">Calibra</text>'
        f'<text x="{rx}" y="15" fill="#010101" fill-opacity=".3">{score_text}</text>'
        f'<text x="{rx}" y="14">{score_text}</text>'
        f'</g>'
        f'</svg>'
    )


# ── shared HTML head ──────────────────────────────────────────────────────────

_HEAD = """\
<!DOCTYPE html>
<html lang="en" class="bg-slate-950 text-slate-100">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config={darkMode:'class',theme:{extend:{colors:{brand:{500:'#6366f1',600:'#4f46e5'}}}}}
</script>
<style>
body{font-family:'Inter',system-ui,sans-serif;background:#020617}
.card{background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px}
.score-ring{width:72px;height:72px;border-radius:50%;border:4px solid;
  display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0}
.badge-pill{display:inline-block;padding:2px 10px;border-radius:9999px;
  font-size:11px;font-weight:700;letter-spacing:.03em}
.dim-bar-bg{background:#1e293b;border-radius:4px;height:6px;overflow:hidden;margin-top:6px}
.dim-bar-fill{height:100%;border-radius:4px;transition:width .4s}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:#6366f1}
th.sort-asc::after{content:' ↑'}
th.sort-desc::after{content:' ↓'}
.finding-row:not(:last-child){border-bottom:1px solid #1e293b}
</style>
</head>
<body class="min-h-screen">
"""

_FOOT = """\
<footer class="mt-16 border-t border-slate-800 py-8 text-center text-slate-500 text-xs">
  Generated by <span class="text-indigo-400 font-semibold">Calibra __CALIBRA_VER__</span>
  &nbsp;·&nbsp; __GENERATED_AT__
</footer>
</body></html>
"""


# ── leaderboard page ──────────────────────────────────────────────────────────

def _leaderboard_row_data(reports: list[dict]) -> list[dict]:
    rows = []
    for r in reports:
        ds = r.get("dataset", {})
        overall = r.get("results", {}).get("overall", {})
        dims = r.get("results", {}).get("dimensions", {})
        meta = r.get("report", {})
        repo_id = ds.get("repository_id", "")
        slug = repo_id.replace("/", "__")
        score = overall.get("score")
        rows.append({
            "repo_id": repo_id,
            "slug": slug,
            "score": score,
            "grade": overall.get("grade", "?"),
            "certification": overall.get("certification", ""),
            "temporal": dims.get("temporal_integrity", {}).get("score"),
            "motion": dims.get("motion_quality", {}).get("score"),
            "coverage": dims.get("behavioral_coverage", {}).get("score"),
            "task": dims.get("task_integrity", {}).get("score"),
            "episodes": ds.get("episodes_total"),
            "frames": ds.get("frames_total"),
            "format": ds.get("dataset_format", ""),
            "updated": meta.get("generated_at", "")[:10],
        })
    return rows


def _render_leaderboard(reports: list[dict], out_dir: Path, title: str) -> None:
    rows = _leaderboard_row_data(reports)
    scores = [r["score"] for r in rows if r["score"] is not None]
    mean_score = round(sum(scores) / len(scores), 1) if scores else 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    calibra_ver = reports[0].get("report", {}).get("calibra_version", "") if reports else ""

    # Stats bar
    certs = [r["certification"] for r in rows]
    n_pass = certs.count("pass")
    n_prov = certs.count("provisional")
    n_fail = certs.count("fail")

    data_json = json.dumps(rows, ensure_ascii=False)

    page = _HEAD.replace("__TITLE__", title)
    page += f"""
<div class="max-w-7xl mx-auto px-4 py-8">

  <!-- Header -->
  <div class="mb-8">
    <div class="flex items-center gap-3 mb-2">
      <div class="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
      </div>
      <h1 class="text-2xl font-bold text-white">{title}</h1>
    </div>
    <p class="text-slate-400 text-sm">Dataset quality rankings powered by Calibra. Scores are audited, reproducible, and revision-stamped.</p>
  </div>

  <!-- Stats -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
    <div class="card text-center">
      <div class="text-2xl font-bold text-white">{len(rows)}</div>
      <div class="text-slate-400 text-xs mt-1">Datasets</div>
    </div>
    <div class="card text-center">
      <div class="text-2xl font-bold" style="color:{_score_hex(mean_score)}">{mean_score}</div>
      <div class="text-slate-400 text-xs mt-1">Mean Score</div>
    </div>
    <div class="card text-center">
      <div class="text-2xl font-bold text-green-400">{n_pass}</div>
      <div class="text-slate-400 text-xs mt-1">Certified</div>
    </div>
    <div class="card text-center">
      <div class="text-2xl font-bold text-slate-300">{now}</div>
      <div class="text-slate-400 text-xs mt-1">Last Updated</div>
    </div>
  </div>

  <!-- Controls -->
  <div class="flex flex-wrap gap-3 mb-4">
    <input id="search" type="text" placeholder="Search datasets..."
      class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-64"
      oninput="applyFilters()">
    <select id="fgrade" class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" onchange="applyFilters()">
      <option value="">All Grades</option>
      <option>A</option><option>B</option><option>C</option><option>D</option><option>F</option>
    </select>
    <select id="fcert" class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500" onchange="applyFilters()">
      <option value="">All Certifications</option>
      <option value="pass">Certified</option>
      <option value="provisional">Provisional</option>
      <option value="fail">Not Certified</option>
    </select>
  </div>

  <!-- Table -->
  <div class="card overflow-x-auto">
    <table class="w-full text-sm" id="leaderboard-table">
      <thead>
        <tr class="text-slate-400 border-b border-slate-800">
          <th class="text-left py-3 px-3 w-8">#</th>
          <th class="text-left py-3 px-3 sortable" onclick="sortBy('repo_id')">Dataset</th>
          <th class="text-right py-3 px-3 sortable" onclick="sortBy('score')">Score</th>
          <th class="text-center py-3 px-3 sortable" onclick="sortBy('grade')">Grade</th>
          <th class="text-center py-3 px-3 sortable" onclick="sortBy('certification')">Cert</th>
          <th class="text-right py-3 px-3 sortable hidden md:table-cell" onclick="sortBy('temporal')">Temporal</th>
          <th class="text-right py-3 px-3 sortable hidden md:table-cell" onclick="sortBy('motion')">Motion</th>
          <th class="text-right py-3 px-3 sortable hidden lg:table-cell" onclick="sortBy('coverage')">Coverage</th>
          <th class="text-right py-3 px-3 sortable hidden lg:table-cell" onclick="sortBy('task')">Task</th>
          <th class="text-right py-3 px-3 sortable hidden xl:table-cell" onclick="sortBy('episodes')">Episodes</th>
          <th class="text-right py-3 px-3 hidden xl:table-cell">Updated</th>
        </tr>
      </thead>
      <tbody id="tbody" class="divide-y divide-slate-800/50"></tbody>
    </table>
    <div id="empty" class="hidden text-center py-12 text-slate-500">No datasets match your filters.</div>
  </div>
</div>

<script>
const DATA = {data_json};
let sortCol = 'score', sortDir = -1;

function fmt(v) {{ return v == null ? '—' : typeof v === 'number' ? v.toFixed(1) : v; }}
function scoreColor(s) {{
  if (s == null) return '#64748b';
  if (s >= 90) return '#22c55e';
  if (s >= 80) return '#10b981';
  if (s >= 70) return '#f59e0b';
  if (s >= 60) return '#f97316';
  return '#ef4444';
}}
function certLabel(c) {{
  return {{pass:'Certified',provisional:'Provisional',fail:'Not Certified'}}[c] || c;
}}
function certColor(c) {{
  return {{pass:'#22c55e',provisional:'#f59e0b',fail:'#ef4444'}}[c] || '#64748b';
}}

function rowHtml(d, rank) {{
  const sc = scoreColor(d.score);
  const cc = certColor(d.certification);
  const parts = d.repo_id.split('/');
  const slug = parts.join('__');
  const href = parts.join('/') + '/index.html';
  return `<tr class="hover:bg-slate-800/40 transition-colors">
    <td class="py-3 px-3 text-slate-500 text-xs">${{rank}}</td>
    <td class="py-3 px-3">
      <a href="${{href}}" class="font-medium text-slate-100 hover:text-indigo-400 transition-colors">${{d.repo_id}}</a>
      <span class="ml-2 text-slate-500 text-xs">${{d.format || ''}}</span>
    </td>
    <td class="py-3 px-3 text-right">
      <span class="font-bold" style="color:${{sc}}">${{fmt(d.score)}}</span>
    </td>
    <td class="py-3 px-3 text-center">
      <span class="badge-pill text-white" style="background:${{sc}}">${{d.grade}}</span>
    </td>
    <td class="py-3 px-3 text-center">
      <span class="badge-pill" style="color:${{cc}};border:1px solid ${{cc}}">${{certLabel(d.certification)}}</span>
    </td>
    <td class="py-3 px-3 text-right hidden md:table-cell text-slate-300">${{fmt(d.temporal)}}</td>
    <td class="py-3 px-3 text-right hidden md:table-cell text-slate-300">${{fmt(d.motion)}}</td>
    <td class="py-3 px-3 text-right hidden lg:table-cell text-slate-300">${{fmt(d.coverage)}}</td>
    <td class="py-3 px-3 text-right hidden lg:table-cell text-slate-300">${{fmt(d.task)}}</td>
    <td class="py-3 px-3 text-right hidden xl:table-cell text-slate-400">${{d.episodes != null ? d.episodes.toLocaleString() : '—'}}</td>
    <td class="py-3 px-3 text-right hidden xl:table-cell text-slate-500 text-xs">${{d.updated}}</td>
  </tr>`;
}}

function applyFilters() {{
  const q = document.getElementById('search').value.toLowerCase();
  const fg = document.getElementById('fgrade').value;
  const fc = document.getElementById('fcert').value;
  let data = DATA.filter(d =>
    (!q || d.repo_id.toLowerCase().includes(q)) &&
    (!fg || d.grade === fg) &&
    (!fc || d.certification === fc)
  );
  data.sort((a, b) => {{
    let av = a[sortCol], bv = b[sortCol];
    av = av ?? (typeof av === 'number' ? -Infinity : '');
    bv = bv ?? (typeof bv === 'number' ? -Infinity : '');
    if (typeof av === 'string') av = av.toLowerCase(), bv = String(bv).toLowerCase();
    return sortDir * (av > bv ? 1 : av < bv ? -1 : 0);
  }});
  const tbody = document.getElementById('tbody');
  const empty = document.getElementById('empty');
  if (!data.length) {{ tbody.innerHTML=''; empty.classList.remove('hidden'); return; }}
  empty.classList.add('hidden');
  tbody.innerHTML = data.map((d, i) => rowHtml(d, i + 1)).join('');
  document.querySelectorAll('th.sortable').forEach(th => {{
    th.classList.remove('sort-asc','sort-desc');
  }});
}}

function sortBy(col) {{
  if (sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = col === 'repo_id' ? 1 : -1; }}
  const ths = document.querySelectorAll('th.sortable');
  ths.forEach(th => th.classList.remove('sort-asc','sort-desc'));
  const idx = ['repo_id','score','grade','certification','temporal','motion','coverage','task','episodes'].indexOf(col);
  if (idx >= 0) ths[idx].classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
  applyFilters();
}}

applyFilters();
</script>
"""
    calibra_ver_safe = calibra_ver or "–"
    page += _FOOT.replace("__CALIBRA_VER__", calibra_ver_safe).replace("__GENERATED_AT__", now)

    out_path = out_dir / "index.html"
    out_path.write_text(page, encoding="utf-8")
    print(f"  leaderboard  → {out_path}", file=sys.stderr)


# ── dataset detail page ───────────────────────────────────────────────────────

def _render_dataset_page(report: dict, history: list[dict], out_dir: Path) -> None:
    ds = report.get("dataset", {})
    overall = report.get("results", {}).get("overall", {})
    dims = report.get("results", {}).get("dimensions", {})
    findings = report.get("results", {}).get("findings", [])
    recs = report.get("results", {}).get("recommendations", {})
    meta = report.get("report", {})
    audit = report.get("audit", {})

    repo_id = ds.get("repository_id", "unknown")
    score = overall.get("score", 0)
    grade = overall.get("grade", "?")
    cert = overall.get("certification", "")
    confidence = overall.get("confidence", 0)
    critical_failures = overall.get("critical_failures", [])

    score_color = _score_hex(score)
    cert_color = _cert_hex(cert)
    cert_label = _cert_label(cert)
    calibra_ver = meta.get("calibra_version", "")
    generated_at = meta.get("generated_at", "")[:10]

    hf_url = f"https://huggingface.co/datasets/{repo_id}" if ds.get("provider") == "huggingface" else ""

    # ── dimension cards ──────────────────────────────────────────────────────
    dim_cards_html = ""
    for dim_key, dim_label in [
        ("temporal_integrity", "Temporal Integrity"),
        ("motion_quality", "Motion Quality"),
        ("behavioral_coverage", "Behavioral Coverage"),
        ("task_integrity", "Task Integrity"),
    ]:
        dim = dims.get(dim_key, {})
        ds_score = dim.get("score", 100.0)
        weight = int(dim.get("weight", 0) * 100)
        dc = _score_hex(ds_score)
        metrics = dim.get("metrics", {})
        metric_rows = ""
        for mname, mval in metrics.items():
            raw = mval.get("value")
            unit = mval.get("unit", "")
            mscore = mval.get("score")
            mc = _score_hex(mscore) if mscore is not None else "#64748b"
            raw_str = f"{raw:.4g} {unit}".strip() if raw is not None else "—"
            score_str = f"{mscore:.0f}" if mscore is not None else "—"
            metric_rows += (
                f'<tr class="text-xs border-b border-slate-800/50">'
                f'<td class="py-1.5 text-slate-300">{mname}</td>'
                f'<td class="py-1.5 text-slate-500 text-right">{raw_str}</td>'
                f'<td class="py-1.5 text-right font-semibold" style="color:{mc}">{score_str}</td>'
                f'</tr>'
            )
        metric_table = (
            f'<table class="w-full mt-3"><tbody>{metric_rows}</tbody></table>'
            if metric_rows else ""
        )
        dim_cards_html += f"""
    <div class="card">
      <div class="flex items-center justify-between mb-1">
        <span class="text-sm font-semibold text-slate-200">{dim_label}</span>
        <span class="text-xs text-slate-500">weight {weight}%</span>
      </div>
      <div class="flex items-end gap-2">
        <span class="text-2xl font-bold" style="color:{dc}">{ds_score:.1f}</span>
        <span class="text-slate-500 text-xs mb-1">/100</span>
      </div>
      <div class="dim-bar-bg"><div class="dim-bar-fill" style="width:{ds_score}%;background:{dc}"></div></div>
      {metric_table}
    </div>"""

    # ── findings ─────────────────────────────────────────────────────────────
    findings_html = ""
    if findings:
        for f in sorted(findings, key=lambda x: ["critical","warning","info","ok"].index(x.get("severity","ok"))):
            sev = f.get("severity", "ok")
            sc = _severity_hex(sev)
            obs = f"{f.get('observed_value', ''):.4g} {f.get('observed_unit','')}" if f.get("observed_value") is not None else ""
            thr = f" (threshold {f.get('threshold'):.4g})" if f.get("threshold") is not None else ""
            findings_html += f"""
      <div class="finding-row py-3">
        <div class="flex items-start gap-3">
          <span class="badge-pill mt-0.5 text-white" style="background:{sc};min-width:64px;text-align:center">{sev.upper()}</span>
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 flex-wrap">
              <span class="font-semibold text-slate-200 text-sm">{f.get('code','')}</span>
              <span class="text-slate-500 text-xs">{f.get('metric','')}</span>
              {f'<span class="text-slate-400 text-xs">{obs}{thr}</span>' if obs else ''}
            </div>
            <p class="text-slate-400 text-xs mt-0.5">{f.get('message','')}</p>
            {f'<p class="text-slate-500 text-xs mt-0.5 italic">{f.get("implication","")}</p>' if f.get('implication') else ''}
          </div>
        </div>
      </div>"""
    else:
        findings_html = '<p class="text-slate-500 text-sm py-4">No findings — dataset passed all checks.</p>'

    # ── policy recommendations ────────────────────────────────────────────────
    policy_rows_html = ""
    policy_map = {
        "behavior_cloning": "Behavior Cloning",
        "act": "ACT",
        "diffusion_policy": "Diffusion Policy",
        "gr00t": "GR00T",
    }
    for key, label in policy_map.items():
        rec = recs.get(key, {})
        if isinstance(rec, str):
            status, reason = rec, ""
        else:
            status = rec.get("status", "review") if isinstance(rec, dict) else "review"
            reason = rec.get("reason", "") or "" if isinstance(rec, dict) else ""
        pc = _policy_hex(status)
        pl = _policy_label(status)
        policy_rows_html += (
            f'<tr class="border-b border-slate-800/50">'
            f'<td class="py-2.5 text-slate-300 text-sm">{label}</td>'
            f'<td class="py-2.5"><span class="badge-pill" style="color:{pc};border:1px solid {pc}">{pl}</span></td>'
            f'<td class="py-2.5 text-slate-500 text-xs">{reason}</td>'
            f'</tr>'
        )

    # ── history table ─────────────────────────────────────────────────────────
    history_html = ""
    if len(history) > 1:
        rows_h = ""
        for h in reversed(history):
            hc = _score_hex(h["score"] or 0)
            rows_h += (
                f'<tr class="border-b border-slate-800/50 text-sm">'
                f'<td class="py-2.5 font-mono text-slate-400 text-xs">{h["revision"]}</td>'
                f'<td class="py-2.5 text-slate-400 text-xs">{h["timestamp"][:10]}</td>'
                f'<td class="py-2.5 font-bold" style="color:{hc}">{h["score"]}</td>'
                f'<td class="py-2.5"><span class="badge-pill text-white" style="background:{hc}">{h["grade"]}</span></td>'
                f'<td class="py-2.5 text-slate-500 text-xs">{h["calibra_version"]}</td>'
                f'</tr>'
            )
        history_html = f"""
  <div class="card mt-6">
    <h2 class="text-base font-semibold text-white mb-4">Score History</h2>
    <table class="w-full"><thead>
      <tr class="text-slate-400 text-xs border-b border-slate-800">
        <th class="text-left py-2">Revision</th><th class="text-left py-2">Date</th>
        <th class="text-left py-2">Score</th><th class="text-left py-2">Grade</th>
        <th class="text-left py-2">Calibra</th>
      </tr>
    </thead><tbody>{rows_h}</tbody></table>
  </div>"""

    # ── badge snippet ─────────────────────────────────────────────────────────
    badge_url = "badge.svg"
    md_snippet = f"![Calibra {score:.0f} {grade}]({badge_url})"
    report_url = "index.html"

    # ── critical failures note ────────────────────────────────────────────────
    crit_note = ""
    if critical_failures:
        crit_note = (
            f'<div class="mt-3 p-3 rounded-lg border border-red-900/50 bg-red-950/30 text-xs text-red-300">'
            f'Critical failures: {", ".join(critical_failures)}</div>'
        )

    title = f"{repo_id} — Calibra"
    page = _HEAD.replace("__TITLE__", title)
    page += f"""
<div class="max-w-5xl mx-auto px-4 py-8">

  <!-- Breadcrumb -->
  <a href="../../index.html" class="text-indigo-400 hover:text-indigo-300 text-sm mb-6 inline-flex items-center gap-1">
    ← Leaderboard
  </a>

  <!-- Hero -->
  <div class="card mt-4 mb-6">
    <div class="flex items-start gap-6 flex-wrap">
      <div class="score-ring" style="border-color:{score_color}">
        <span class="text-2xl font-black" style="color:{score_color}">{score:.0f}</span>
        <span class="text-slate-500 text-xs">/ 100</span>
      </div>
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-3 flex-wrap mb-2">
          <h1 class="text-xl font-bold text-white">{repo_id}</h1>
          {f'<a href="{hf_url}" target="_blank" class="text-indigo-400 hover:text-indigo-300 text-xs">↗ HF Hub</a>' if hf_url else ''}
        </div>
        <div class="flex items-center gap-2 flex-wrap">
          <span class="badge-pill text-white" style="background:{score_color}">{grade}</span>
          <span class="badge-pill" style="color:{cert_color};border:1px solid {cert_color}">{cert_label}</span>
          <span class="text-slate-500 text-xs">confidence {confidence:.0%}</span>
        </div>
        {crit_note}
        <div class="mt-3 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-1 text-xs text-slate-400">
          <span>Episodes: <b class="text-slate-200">{ds.get('episodes_total', '—')}</b></span>
          <span>Frames: <b class="text-slate-200">{f"{ds.get('frames_total',0):,}" if ds.get('frames_total') else '—'}</b></span>
          <span>Format: <b class="text-slate-200">{ds.get('dataset_format','—')}</b></span>
          <span>Revision: <b class="text-slate-200 font-mono">{(ds.get('revision') or '—')[:8]}</b></span>
          <span>Rubric: <b class="text-slate-200">{audit.get('scoring_rubric','—')}</b></span>
          <span>Calibra: <b class="text-slate-200">{calibra_ver}</b></span>
          <span>Audited: <b class="text-slate-200">{generated_at}</b></span>
          <span>Report ID: <b class="text-slate-200 font-mono text-xs">{(meta.get('id') or '—')[:20]}…</b></span>
        </div>
      </div>
    </div>
  </div>

  <!-- Dimensions -->
  <h2 class="text-base font-semibold text-white mb-3">Quality Dimensions</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
    {dim_cards_html}
  </div>

  <!-- Findings -->
  <div class="card mb-6">
    <h2 class="text-base font-semibold text-white mb-1">Findings</h2>
    <p class="text-slate-500 text-xs mb-3">{len(findings)} finding(s) across all analyzers</p>
    <div>{findings_html}</div>
  </div>

  <!-- Policy Recommendations -->
  <div class="card mb-6">
    <h2 class="text-base font-semibold text-white mb-3">Policy Recommendations</h2>
    <table class="w-full">
      <thead><tr class="text-slate-400 text-xs border-b border-slate-800">
        <th class="text-left py-2">Policy</th>
        <th class="text-left py-2">Status</th>
        <th class="text-left py-2">Notes</th>
      </tr></thead>
      <tbody>{policy_rows_html}</tbody>
    </table>
  </div>

  <!-- Badge -->
  <div class="card mb-6">
    <h2 class="text-base font-semibold text-white mb-3">Calibra Certified Badge</h2>
    <div class="flex items-start gap-6 flex-wrap">
      <img src="{badge_url}" alt="Calibra {score:.0f} {grade}" class="h-5">
      <div class="flex-1">
        <p class="text-xs text-slate-400 mb-2">Add to your dataset README or GitHub repository:</p>
        <pre class="bg-slate-800 rounded-lg p-3 text-xs text-slate-300 overflow-x-auto">{md_snippet}</pre>
      </div>
    </div>
  </div>

  {history_html}

</div>
"""
    page += _FOOT.replace("__CALIBRA_VER__", calibra_ver or "–").replace("__GENERATED_AT__", generated_at)

    # Write
    parts = repo_id.split("/")
    page_path = out_dir
    for part in parts:
        page_path = page_path / part
    page_path.mkdir(parents=True, exist_ok=True)
    (page_path / "index.html").write_text(page, encoding="utf-8")
    print(f"  dataset page → {page_path / 'index.html'}", file=sys.stderr)


# ── orchestrator ──────────────────────────────────────────────────────────────

def build_site(
    results_dir: Path,
    out_dir: Path,
    title: str = "Calibra Leaderboard",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = _scan_results(results_dir)
    if not reports:
        print("No reports found — run 'calibra audit-all' first.", file=sys.stderr)
        return

    print(f"Building site from {len(reports)} report(s) → {out_dir}", file=sys.stderr)

    # Leaderboard
    _render_leaderboard(reports, out_dir, title=title)

    # Per-dataset pages + badges
    for report in reports:
        repo_id = report.get("dataset", {}).get("repository_id", "")
        if not repo_id:
            continue

        history = _collect_history(results_dir, repo_id)

        # Dataset page
        _render_dataset_page(report, history, out_dir)

        # Badge
        overall = report.get("results", {}).get("overall", {})
        score = overall.get("score", 0)
        grade = overall.get("grade", "?")
        cert = overall.get("certification", "")
        svg = _badge_svg(score, grade, cert)

        parts = repo_id.split("/")
        badge_dir = out_dir
        for part in parts:
            badge_dir = badge_dir / part
        badge_dir.mkdir(parents=True, exist_ok=True)
        (badge_dir / "badge.svg").write_text(svg, encoding="utf-8")

        # History JSON (for external consumers)
        if history:
            (badge_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

    n = len(reports)
    print(f"\nDone. {n} dataset(s) → {out_dir}/index.html", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_site(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra site",
        description="Generate a static leaderboard website from audit results.",
    )
    p.add_argument("--results", metavar="DIR", default="results",
                   help="Directory containing audit JSON reports (default: ./results)")
    p.add_argument("--out", metavar="DIR", default="site",
                   help="Output directory for the generated site (default: ./site)")
    p.add_argument("--title", metavar="TITLE", default="Calibra Leaderboard",
                   help='Leaderboard page title (default: "Calibra Leaderboard")')
    args = p.parse_args(argv)

    build_site(
        results_dir=Path(args.results),
        out_dir=Path(args.out),
        title=args.title,
    )
