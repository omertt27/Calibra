#!/usr/bin/env python3
"""
scripts/summarize_experiments.py
---------------------------------
Summarise all reference profiles into a single table and emit both a
Markdown and a LaTeX version.  Run from the repository root:

    python scripts/summarize_experiments.py
    python scripts/summarize_experiments.py --format latex
    python scripts/summarize_experiments.py --format both --out experiments/figures/
"""

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REF_DIR   = REPO_ROOT / "calibra" / "references"

METRIC_PREFIXES = [
    "control_smoothness/",
    "jerk_spikes/",
    "velocity_discontinuity/",
    "temporal_stability/",
    "",
]

def get_mean(dist: dict, key: str) -> float | None:
    for pfx in METRIC_PREFIXES:
        v = dist.get(pfx + key, {})
        if isinstance(v, dict) and v.get("mean") is not None:
            return v["mean"]
    return None


def load_profiles() -> list[dict]:
    rows = []
    for f in sorted(REF_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        meta = d.get("meta", {})
        dist = d.get("per_episode_distributions", {})
        rows.append(dict(
            dataset    = meta.get("dataset", f.stem),
            n_episodes = meta.get("n_episodes", "-"),
            ctrl       = meta.get("control_mode", "?"),
            is_scripted= "scripted" in f.stem,
            spike      = get_mean(dist, "per_episode_spike_rate"),
            vel_disc   = get_mean(dist, "per_episode_vel_disc_rate"),
            jitter_cv  = get_mean(dist, "per_episode_jitter_cv"),
            ldlj       = get_mean(dist, "per_episode_ldlj"),
        ))
    return rows


def fmt_pct(v: float | None) -> str:
    return f"{v * 100:.2f}%" if v is not None else "-"


def fmt_sci(v: float | None) -> str:
    return f"{v:.1e}" if v is not None else "-"


def fmt_f2(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "-"


def render_markdown(rows: list[dict]) -> str:
    header = (
        "| Dataset | Ctrl | N | Spike% | VelDisc% | Jitter CV | LDLJ |\n"
        "|---------|------|---|--------|----------|-----------|------|\n"
    )
    lines = [header]
    for r in rows:
        scripted = " *(scripted)*" if r["is_scripted"] else ""
        lines.append(
            f"| `{r['dataset']}`{scripted} | {r['ctrl']} | {r['n_episodes']} "
            f"| {fmt_pct(r['spike'])} | {fmt_pct(r['vel_disc'])} "
            f"| {fmt_sci(r['jitter_cv'])} | {fmt_f2(r['ldlj'])} |\n"
        )
    return "".join(lines)


def render_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Calibra observability metrics across 15 public imitation-learning datasets. "
        r"Scripted datasets are marked with $\dagger$.}",
        r"\label{tab:cross_dataset}",
        r"\small",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Dataset & Ctrl & $N$ & Spike\,\% & VelDisc\,\% & Jitter CV & LDLJ \\",
        r"\midrule",
    ]
    for r in rows:
        ds = r["dataset"].replace("lerobot/", "").replace("nvidia/", "").replace("_", r"\_")
        if r["is_scripted"]:
            ds += r"$^\dagger$"
        lines.append(
            f"{ds} & {r['ctrl']} & {r['n_episodes']} & "
            f"{fmt_pct(r['spike'])} & {fmt_pct(r['vel_disc'])} & "
            f"{fmt_sci(r['jitter_cv'])} & {fmt_f2(r['ldlj'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="markdown")
    parser.add_argument("--out", default=None, help="Output directory (default: stdout)")
    args = parser.parse_args()

    rows = load_profiles()
    print(f"Loaded {len(rows)} reference profiles", file=sys.stderr)

    out_dir = pathlib.Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("markdown", "both"):
        md = render_markdown(rows)
        if out_dir:
            (out_dir / "table1_cross_dataset.md").write_text(md)
            print(f"Written {out_dir / 'table1_cross_dataset.md'}", file=sys.stderr)
        else:
            print(md)

    if args.format in ("latex", "both"):
        tex = render_latex(rows)
        if out_dir:
            (out_dir / "table1_cross_dataset.tex").write_text(tex)
            print(f"Written {out_dir / 'table1_cross_dataset.tex'}", file=sys.stderr)
        else:
            print(tex)


if __name__ == "__main__":
    main()
