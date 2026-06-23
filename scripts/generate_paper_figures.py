"""
Generate all paper figures including the system overview diagram (Figure 1).

Usage:
    python scripts/generate_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_system_overview() -> None:
    """Generate Figure 1: Calibra system overview block diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    def box(x, y, w, h, label, sublabel="", color="#E3F2FD", textcolor="#1A237E",
            fontsize=10, bold=True):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#90CAF9", linewidth=1.5,
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2 + (0.12 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=textcolor)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.18, sublabel,
                    ha="center", va="center", fontsize=7.5,
                    color="#546E7A", style="italic")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#455A64",
                                   lw=1.5, mutation_scale=15))

    def label(x, y, text, fontsize=8.5, color="#455A64"):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=color, style="italic")

    # ── Row 1: Input formats ──────────────────────────────────────────────────
    formats = [
        ("LeRobot v2\n(Parquet)", 0.3),
        ("HDF5 /\nIsaac Lab", 1.85),
        ("RLDS /\nTF Datasets", 3.4),
        ("MCAP /\nROS2 bags", 4.95),
        ("HuggingFace\nHub", 6.5),
    ]
    for label_text, x in formats:
        box(x, 5.7, 1.3, 0.9, label_text, color="#FFF9C4", textcolor="#33691E",
            fontsize=8.5, bold=False)
        arrow(x + 0.65, 5.7, x + 0.65, 5.1)

    # ── Row 2: Format adapters ────────────────────────────────────────────────
    box(0.3, 4.45, 7.5, 0.55, "Format Adapters  —  metadata-first, lazy image skip",
        color="#F3E5F5", textcolor="#4A148C", fontsize=9)
    arrow(4.05, 4.45, 4.05, 3.85)

    # ── Row 3: EpisodeBatch ───────────────────────────────────────────────────
    box(0.3, 3.35, 7.5, 0.45, "EpisodeBatch  —  normalised internal representation",
        color="#E8F5E9", textcolor="#1B5E20", fontsize=9)
    arrow(4.05, 3.35, 4.05, 2.75)

    # ── Row 4: Four analyzers ─────────────────────────────────────────────────
    analyzers = [
        ("Temporal\nAnalyzer", "jitter, dropout"),
        ("Smoothness\nAnalyzer", "LDLJ, spikes, VD"),
        ("Coverage\nAnalyzer", "entropy, PCA"),
        ("Task Structure\nAnalyzer", "contact, grasp"),
    ]
    for i, (name, sub) in enumerate(analyzers):
        x = 0.3 + i * 1.9
        box(x, 2.1, 1.7, 0.6, name, sub, color="#E3F2FD", textcolor="#0D47A1",
            fontsize=8.5)
        arrow(x + 0.85, 2.1, x + 0.85, 1.55)

    label(4.05, 2.72, "parallelisable  ·  95% bootstrap CIs over episodes", fontsize=8)

    # ── Row 5: DiagnosticReport ───────────────────────────────────────────────
    box(0.3, 1.1, 7.5, 0.4, "DiagnosticReport  —  metric values · risk flags · bootstrap CIs",
        color="#FBE9E7", textcolor="#BF360C", fontsize=9)

    # ── Row 6: Six output commands ────────────────────────────────────────────
    commands = ["audit", "compare", "certify", "prune", "predict", "watch\n--stream"]
    colors_cmd = ["#ECEFF1", "#ECEFF1", "#ECEFF1", "#E8F5E9", "#FFF3E0", "#FCE4EC"]
    text_colors = ["#37474F"] * 3 + ["#1B5E20", "#E65100", "#880E4F"]
    for i, (cmd, bg, tc) in enumerate(zip(commands, colors_cmd, text_colors)):
        x = 0.3 + i * 1.27
        box(x, 0.1, 1.17, 0.75, f"`{cmd}`", color=bg, textcolor=tc, fontsize=8.5)
        arrow(0.3 + i * 1.27 + 0.585, 1.1, 0.3 + i * 1.27 + 0.585, 0.85)

    # ── Side panel: Claim registry ────────────────────────────────────────────
    box(8.3, 3.35, 3.4, 2.95,
        "Falsifiable\nClaim Registry",
        "evidence · confidence\nfalsification condition",
        color="#FFF8E1", textcolor="#E65100", fontsize=9)
    ax.annotate("", xy=(8.3, 4.5), xytext=(7.8, 4.5),
                arrowprops=dict(arrowstyle="<->", color="#E65100", lw=1.5))

    # ── Side panel: Outcome DB ────────────────────────────────────────────────
    box(8.3, 0.9, 3.4, 1.8,
        "Outcome DB\n(empirical loop)",
        "~/.calibra/outcomes.jsonl\nrecord → blend → calibrate",
        color="#F3E5F5", textcolor="#4A148C", fontsize=9)
    ax.annotate("", xy=(8.3, 1.4), xytext=(7.8, 1.4),
                arrowprops=dict(arrowstyle="<->", color="#4A148C", lw=1.5))

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(4.05, 6.8, "Calibra Pipeline",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

    fig.tight_layout(pad=0.2)
    out = FIG_DIR / "fig_system_overview.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / "fig_system_overview.png", bbox_inches="tight", dpi=200)
    print(f"  fig_system_overview → {out}")


def regenerate_correlation_figure() -> None:
    """Re-run the predictor correlation study to regenerate fig6."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "predict_correlation_study.py")],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  fig6_predict_correlation regenerated")
    else:
        print(f"  Warning: correlation study failed: {result.stderr[-200:]}")


def main() -> None:
    print("Generating paper figures ...")
    generate_system_overview()
    regenerate_correlation_figure()
    print(f"\nAll figures in {FIG_DIR}/")
    print("Paper can now be compiled with: cd paper && pdflatex main.tex")


if __name__ == "__main__":
    main()
