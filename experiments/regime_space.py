"""
Regime space visualization.

Plots each tested dataset as a point in a 2D space:
  X axis: noise score  (composite of spike_fraction + vel_disc_rate)
  Y axis: state entropy (state_space_entropy_2d from latent_dynamics)

Points are colored by which ablation condition won:
  Blue  = diversity-only dominant
  Green = quality + diversity complementary (both contribute)
  Red   = quality-only dominant  (not yet observed)

Calibra's regime diagnosis for each dataset is shown as a background
shaded region, making the classifier boundaries visible.

The ablation results embedded here come from experiments run on an
RTX 2080. Each dataset used 5 random seeds for the baseline.

Usage
-----
    python experiments/regime_space.py
    python experiments/regime_space.py --save
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Empirical results ─────────────────────────────────────────────────────────
# Each entry: dataset name, noise metrics, entropy, ablation winner, and
# the percentage gain of each component vs. random.

DATASETS = [
    {
        "name": "ALOHA mobile\n(dual-arm, real)",
        "noise_score": 0.075,       # spike=0.007, vel_disc=0.013
        "state_entropy": 6.86,      # from latent_dynamics
        "q_gain": 8.8,
        "d_gain": 13.0,
        "full_gain": 22.6,
        "winner": "both",           # both components complementary
        "n_eps": 85,
        "policy": "BC-MLP (14D)",
        "ann_offset": (0.025, 0.18),
    },
    {
        "name": "DROID-100\n(multi-robot)",
        "noise_score": 0.443,
        "state_entropy": 5.23,
        "q_gain": -5.8,             # quality-only HURTS
        "d_gain": 16.9,
        "full_gain": 16.9,
        "winner": "diversity",
        "n_eps": 100,
        "policy": "BC-MLP (7D)",
        "ann_offset": (0.025, -0.55),
    },
    {
        "name": "PushT real\n(contact-rich)",
        "noise_score": 0.416,       # vel_disc=0.29 from contact events
        "state_entropy": 6.62,      # from latent_dynamics
        "q_gain": -29.2,
        "d_gain": 48.4,
        "full_gain": -30.5,
        "winner": "diversity",
        "n_eps": 136,
        "policy": "BC-MLP (8D→7D)",
        "note": "vel_disc from contact\nevents, not control noise",
        "ann_offset": (-0.18, 0.25),
    },
]

# Calibra regime thresholds (from strategy.py)
_SPIKE_LOW = 0.025
_SPIKE_HIGH = 0.090
_DISC_LOW = 0.040
_DISC_HIGH = 0.130

# Approximate noise_score thresholds
# noise_score = 0.5*(spike/0.09) + 0.35*(disc/0.13) + ...
# LOW_NOISE → MODERATE boundary: roughly noise_score = 0.15
# MODERATE → HIGH boundary: roughly noise_score = 0.38
_NS_LOW_MED = 0.15
_NS_MED_HIGH = 0.38


def _winner_color(winner: str) -> str:
    return {
        "diversity": "#2563EB",   # blue
        "quality": "#DC2626",     # red
        "both": "#16A34A",        # green
    }.get(winner, "#6B7280")


def _winner_label(winner: str, q_gain: float, d_gain: float) -> str:
    if winner == "both":
        return f"Both contribute\n(Q:{q_gain:+.0f}%, D:{d_gain:+.0f}%)"
    elif winner == "diversity":
        return f"Diversity dominant\n(Q:{q_gain:+.0f}%, D:{d_gain:+.0f}%)"
    else:
        return f"Quality dominant\n(Q:{q_gain:+.0f}%, D:{d_gain:+.0f}%)"


def plot(save: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("matplotlib not installed. pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F8F9FA")

    # ── regime background bands ───────────────────────────────────────────────
    y_min, y_max = 3.5, 9.0
    x_min, x_max = 0.0, 0.80

    ax.axvspan(x_min, _NS_LOW_MED, ymin=0, ymax=1,
               color="#DBEAFE", alpha=0.35, zorder=0)
    ax.axvspan(_NS_LOW_MED, _NS_MED_HIGH, ymin=0, ymax=1,
               color="#FEF3C7", alpha=0.35, zorder=0)
    ax.axvspan(_NS_MED_HIGH, x_max, ymin=0, ymax=1,
               color="#FEE2E2", alpha=0.35, zorder=0)

    # Region labels
    for x_mid, label, col in [
        ((_NS_LOW_MED)/2,          "LOW NOISE\n(diversity + quality\ncomplementary)", "#1D4ED8"),
        ((_NS_LOW_MED+_NS_MED_HIGH)/2, "MODERATE NOISE\n(diversity dominant,\nquality risky)", "#92400E"),
        ((_NS_MED_HIGH+x_max)/2,   "HIGH NOISE\n(contact or sensor\nnoise — diversity\nstill dominant)", "#991B1B"),
    ]:
        ax.text(x_mid, y_max - 0.25, label, ha="center", va="top",
                fontsize=7.5, color=col, fontweight="bold", style="italic", zorder=1)

    # Regime boundary lines
    for xv in [_NS_LOW_MED, _NS_MED_HIGH]:
        ax.axvline(xv, color="#94A3B8", linewidth=1, linestyle="--", zorder=1, alpha=0.7)

    # ── dataset points ────────────────────────────────────────────────────────
    plotted_winners = set()
    for ds in DATASETS:
        x = ds["noise_score"]
        y = ds.get("state_entropy") or 5.5   # fallback if not extracted
        color = _winner_color(ds["winner"])

        # Bubble sized by n_episodes
        size = 200 + ds["n_eps"] * 1.5

        ax.scatter(x, y, s=size, c=color, alpha=0.85, zorder=5,
                   edgecolors="white", linewidths=1.5)

        # Annotation
        ox, oy = ds.get("ann_offset", (0.025, 0.18))
        ax.annotate(
            ds["name"],
            xy=(x, y),
            xytext=(x + ox, y + oy),
            fontsize=9, fontweight="bold", color="#1E293B",
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.8),
            zorder=6,
        )

        # Gain annotation near offset label
        gain_str = f"D:{ds['d_gain']:+.0f}%  Q:{ds['q_gain']:+.0f}%  Full:{ds['full_gain']:+.0f}%"
        ax.text(x + ox, y + oy - 0.30, gain_str, ha="center", va="top",
                fontsize=7, color=color, zorder=6)

        # Note (e.g. contact explanation)
        if "note" in ds:
            ax.text(x + ox, y + oy - 0.58, ds["note"], ha="center", va="top",
                    fontsize=6.5, color="#64748B", style="italic", zorder=6)

        plotted_winners.add(ds["winner"])

    # ── legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#16A34A", label="Both quality + diversity contribute"),
        mpatches.Patch(color="#2563EB", label="Diversity dominant (quality alone harmful)"),
        mpatches.Patch(color="#DC2626", label="Quality dominant (not yet observed)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#CBD5E1")

    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(
        "Noise score  (composite: 0.5 × spike_fraction + 0.35 × vel_disc_rate + ...)",
        fontsize=10,
    )
    ax.set_ylabel("State space entropy  (bits, from latent_dynamics)", fontsize=10)
    ax.set_title(
        "Dataset Regime Space: noise × entropy → optimal selection strategy\n"
        "Calibra ablation results across 3 robot datasets  |  bubble size ~ n_episodes",
        fontsize=11, fontweight="bold", pad=12,
    )

    ax.grid(True, alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── key finding callout ───────────────────────────────────────────────────
    ax.text(
        0.01, 0.01,
        "Key finding: diversity selection is robust across all tested regimes.\n"
        "Quality filtering fails in two independent ways: (1) collapses rare\n"
        "morphologies in heterogeneous datasets; (2) misclassifies contact\n"
        "events as noise in manipulation tasks.",
        transform=ax.transAxes,
        fontsize=7.5, va="bottom", color="#374151",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CBD5E1", alpha=0.9),
    )

    # ── calibration note ──────────────────────────────────────────────────────
    ax.text(
        0.99, 0.99,
        "n=3 datasets  |  5 random seeds each\nAblation: BC-MLP, 200-300 epochs, RTX 2080\n"
        "Hypothesis, not established law.",
        transform=ax.transAxes,
        fontsize=6.5, va="top", ha="right", color="#6B7280", style="italic",
    )

    fig.tight_layout()

    if save:
        out_pdf = FIG_DIR / "fig_regime_space.pdf"
        out_png = FIG_DIR / "fig_regime_space.png"
        fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")
    else:
        plt.show()

    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true", help="Save to experiments/figures/")
    args = p.parse_args()
    plot(save=args.save)


if __name__ == "__main__":
    main()
