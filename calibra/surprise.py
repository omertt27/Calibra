"""
calibra surprise — World-model surprise scoring for robot demonstrations.

Trains a RobotJEPA world model on your dataset, scores every episode by how
much it surprises the model, then cross-references with kinematic quality to
classify each episode into one of three categories:

  NOVEL      high surprise + low jerk  → genuine unexplored dynamics — KEEP
  CORRUPTED  high surprise + high jerk → noise/dropout masking as novelty — PRUNE
  REDUNDANT  low surprise              → near-duplicate of seen dynamics — safe to prune

This is the correct selection criterion for world model training:
  • IL coreset selection maximises behavioural diversity (action-space spread)
  • World model selection maximises dynamic novelty (latent prediction error)

These are different sets. An episode can be diverse in action space but
predictable in dynamics (e.g. the same grasp at a new XY position). Surprise
finds what the world model genuinely doesn't know.

Usage
-----
    calibra surprise /data/my_demos
    calibra surprise lerobot/my_dataset --top 20
    calibra surprise /data/my_demos --json
    calibra surprise /data/my_demos --epochs 30   # faster, less accurate

Requires: torch (pip install torch)
Exit codes: 0 OK, 1 if >30% of high-surprise episodes are corrupted (data quality problem)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra import __version__
from calibra.pipeline import Pipeline
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import DiagnosticReport

_WIDTH = 60
_THICK = "━" * _WIDTH
_THIN = "─" * _WIDTH

# Thresholds for episode classification
_SURPRISE_HIGH = 0.60     # above this → model is surprised
_JERK_HIGH = 0.04         # above this → episode is kinematically suspicious


# ── result schema ──────────────────────────────────────────────────────────────

@dataclass
class EpisodeSurpriseScore:
    episode_id: str
    surprise: float          # 0–1, normalised within batch
    jerk_rate: float         # fraction of steps with jerk spikes
    verdict: str             # NOVEL | CORRUPTED | REDUNDANT


@dataclass
class SurpriseResult:
    dataset_name: str
    n_episodes: int
    n_novel: int
    n_corrupted: int
    n_redundant: int
    scores: list[EpisodeSurpriseScore]
    training_epochs: int
    torch_available: bool

    def summary(self, top: int = 10) -> str:
        if not self.torch_available:
            return (
                f"{_THICK}\n"
                f"  CALIBRA WORLD-MODEL SURPRISE\n"
                f"{_THICK}\n"
                f"  torch not installed — world-model scoring requires PyTorch.\n"
                f"  Install: pip install torch\n"
                f"{_THICK}"
            )

        lines = [
            _THICK,
            "  CALIBRA WORLD-MODEL SURPRISE",
            _THICK,
            f"  Dataset : {self.dataset_name}  ·  {self.n_episodes} episodes",
            _THIN,
        ]

        novel = [s for s in self.scores if s.verdict == "NOVEL"]
        corrupted = [s for s in self.scores if s.verdict == "CORRUPTED"]
        redundant = [s for s in self.scores if s.verdict == "REDUNDANT"]

        if novel:
            lines += [
                f"  NOVEL DYNAMICS  (high surprise, low jerk — KEEP for world model training)",
                _THIN,
            ]
            for s in novel[:top]:
                lines.append(
                    f"  {s.episode_id:<20}  surprise={s.surprise:.3f}  "
                    f"jerk={s.jerk_rate:.3f}  → NOVEL"
                )
            if len(novel) > top:
                lines.append(f"  ... and {len(novel) - top} more")
            lines.append("")

        if corrupted:
            lines += [
                f"  CORRUPTED  (high surprise, high jerk — PRUNE)",
                _THIN,
            ]
            for s in corrupted[:top]:
                lines.append(
                    f"  {s.episode_id:<20}  surprise={s.surprise:.3f}  "
                    f"jerk={s.jerk_rate:.3f}  → CORRUPTED"
                )
            if len(corrupted) > top:
                lines.append(f"  ... and {len(corrupted) - top} more")
            lines.append("")

        if redundant:
            lines += [
                f"  REDUNDANT  (low surprise — near-duplicate dynamics)",
                _THIN,
                f"  {len(redundant)} episodes with surprise < {_SURPRISE_HIGH:.2f}",
                "",
            ]

        novel_pct = len(novel) / max(self.n_episodes, 1) * 100
        corrupt_pct = len(corrupted) / max(self.n_episodes, 1) * 100
        redundant_pct = len(redundant) / max(self.n_episodes, 1) * 100

        corruption_warning = ""
        if len(novel) + len(corrupted) > 0:
            corrupt_of_high = len(corrupted) / (len(novel) + len(corrupted))
            if corrupt_of_high > 0.30:
                corruption_warning = (
                    f"\n  ⚠️  WARNING: {corrupt_of_high:.0%} of high-surprise episodes are\n"
                    f"  corrupted — run `calibra certify` to diagnose data quality."
                )

        lines += [
            _THIN,
            "  SUMMARY",
            _THIN,
            f"  Novel (keep)   : {len(novel):>5}  ({novel_pct:.1f}%)",
            f"  Corrupted      : {len(corrupted):>5}  ({corrupt_pct:.1f}%)",
            f"  Redundant      : {len(redundant):>5}  ({redundant_pct:.1f}%)",
        ]

        if novel:
            lines += [
                "",
                f"  Minimal world-model training set: {len(novel)} episodes",
                f"  Run: calibra prune <path> --strategy world-model --keep "
                f"{max(0.05, novel_pct / 100):.2f}",
            ]

        if corruption_warning:
            lines.append(corruption_warning)

        lines.append(_THICK)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "n_episodes": self.n_episodes,
            "n_novel": self.n_novel,
            "n_corrupted": self.n_corrupted,
            "n_redundant": self.n_redundant,
            "torch_available": self.torch_available,
            "training_epochs": self.training_epochs,
            "scores": [
                {
                    "episode_id": s.episode_id,
                    "surprise": round(s.surprise, 4),
                    "jerk_rate": round(s.jerk_rate, 4),
                    "verdict": s.verdict,
                }
                for s in self.scores
            ],
        }


# ── core analysis ──────────────────────────────────────────────────────────────

def score_surprise(
    batch: EpisodeBatch,
    report: DiagnosticReport,
    epochs: int = 60,
) -> SurpriseResult:
    """
    Train RobotJEPA on batch and classify every episode.

    Parameters
    ----------
    batch   : loaded EpisodeBatch
    report  : DiagnosticReport for kinematic quality cross-reference
    epochs  : JEPA training epochs (fewer = faster but less accurate)
    """
    try:
        import torch as _torch  # noqa: F401
    except ImportError:
        return SurpriseResult(
            dataset_name=report.dataset_name,
            n_episodes=batch.n_episodes,
            n_novel=0, n_corrupted=0, n_redundant=0,
            scores=[], training_epochs=epochs,
            torch_available=False,
        )

    from calibra.models.robot_jepa import RobotJEPA, RobotJEPAConfig

    # Extract per-episode jerk rates from diagnostic report
    jerk_rates = _extract_jerk_rates(report, batch)

    print(
        f"  Training RobotJEPA ({epochs} epochs)...",
        file=sys.stderr, flush=True,
    )

    cfg = RobotJEPAConfig(n_epochs=epochs)
    jepa = RobotJEPA(cfg)
    jepa.fit(batch)
    surprise_scores = jepa.score_episodes(batch)

    scores: list[EpisodeSurpriseScore] = []
    for ep in batch.episodes:
        eid = ep.metadata.episode_id
        surprise = surprise_scores.get(eid, 0.0)
        jerk = jerk_rates.get(eid, 0.0)

        if surprise >= _SURPRISE_HIGH and jerk < _JERK_HIGH:
            verdict = "NOVEL"
        elif surprise >= _SURPRISE_HIGH and jerk >= _JERK_HIGH:
            verdict = "CORRUPTED"
        else:
            verdict = "REDUNDANT"

        scores.append(EpisodeSurpriseScore(
            episode_id=eid,
            surprise=round(surprise, 4),
            jerk_rate=round(jerk, 4),
            verdict=verdict,
        ))

    # Sort: novel first (by surprise desc), then corrupted, then redundant
    _ORDER = {"NOVEL": 0, "CORRUPTED": 1, "REDUNDANT": 2}
    scores.sort(key=lambda s: (_ORDER[s.verdict], -s.surprise))

    n_novel = sum(1 for s in scores if s.verdict == "NOVEL")
    n_corrupted = sum(1 for s in scores if s.verdict == "CORRUPTED")
    n_redundant = sum(1 for s in scores if s.verdict == "REDUNDANT")

    return SurpriseResult(
        dataset_name=report.dataset_name,
        n_episodes=batch.n_episodes,
        n_novel=n_novel,
        n_corrupted=n_corrupted,
        n_redundant=n_redundant,
        scores=scores,
        training_epochs=epochs,
        torch_available=True,
    )


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_jerk_rates(
    report: DiagnosticReport,
    batch: EpisodeBatch,
) -> dict[str, float]:
    """Extract per-episode jerk spike rates from diagnostic report."""
    for result in report.analyzer_results:
        if result.analyzer_name == "control_smoothness":
            per_ep = result.raw_metrics.get("jerk_spikes", {}).get("per_episode_spike_rate")
            if per_ep is not None:
                return {
                    ep.metadata.episode_id: float(per_ep[i]) if i < len(per_ep) else 0.0
                    for i, ep in enumerate(batch.episodes)
                }
    # Fallback: try per_episode_spike_rate at top level
    for result in report.analyzer_results:
        per_ep = result.raw_metrics.get("per_episode_spike_rate")
        if per_ep is not None:
            return {
                ep.metadata.episode_id: float(per_ep[i]) if i < len(per_ep) else 0.0
                for i, ep in enumerate(batch.episodes)
            }
    return {ep.metadata.episode_id: 0.0 for ep in batch.episodes}


# ── CLI ────────────────────────────────────────────────────────────────────────

def run_surprise(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra surprise",
        description=(
            "Score every episode by world-model surprise (RobotJEPA). "
            "Classifies episodes as NOVEL, CORRUPTED, or REDUNDANT. "
            "Requires: pip install torch"
        ),
    )
    p.add_argument("path", help="Dataset path or HuggingFace Hub ID")
    p.add_argument(
        "--top", "-n", type=int, default=10, metavar="N",
        help="Show top N episodes per category (default: 10)",
    )
    p.add_argument(
        "--epochs", "-e", type=int, default=60, metavar="N",
        help="JEPA training epochs (default: 60; use 20–30 for quick runs)",
    )
    p.add_argument(
        "--format", "-f",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force format adapter",
    )
    p.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    args = p.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {args.path!r} ...")

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    try:
        from calibra.ingestion.registry import load
        batch = load(args.path, reader=reader)
        report = Pipeline().run(batch)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {report.n_episodes} episodes  ·  {report.n_samples:,} steps")

    result = score_surprise(batch, report, epochs=args.epochs)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary(top=args.top))

    # Exit 1 if corruption rate among high-surprise episodes > 30%
    if result.torch_available:
        high = result.n_novel + result.n_corrupted
        if high > 0 and result.n_corrupted / high > 0.30:
            sys.exit(1)
