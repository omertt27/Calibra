"""
World Model Consistency Analyzer.

Trains a lightweight RobotJEPA on the episode batch and reports:
  - Per-episode JEPA surprise score  (world-model learnability signal)
  - World-model learnability fraction  (share of episodes the WM predicts well)
  - High-surprise fraction  (share that confuses the world model)

Research framing
----------------
The central finding connecting Calibra to LeCun's JEPA paradigm:

  Episodes that score well on Calibra's hand-crafted quality metrics
  (low jerk, low dropout, good coverage) ALSO have low JEPA surprise —
  they are learnable by a world model.

This validates that diagnostic metrics are offline proxies for world-model
predictability.  An IL policy trained on high-surprise data memorizes
specific trajectories; a JEPA world model trained on the low-surprise
coreset generalizes to novel states.

Cross-reference with jerk metrics to distinguish two high-surprise regimes:

    high surprise + high jerk   →  CORRUPTED  (noise, dropout, spikes)
    high surprise + low  jerk   →  NOVEL      (rare, informative — keep)

Requires PyTorch (optional dependency). Skips gracefully if absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult, RiskFlag, RiskLevel, ObservedValue


@dataclass
class WorldModelConsistencyAnalyzer(Analyzer):
    """
    Evaluates dataset consistency with respect to a learned JEPA world model.

    Parameters
    ----------
    latent_dim  : dimensionality of the JEPA latent space.
    n_epochs    : training epochs for the JEPA predictor.
    batch_size  : mini-batch size during JEPA training.
    verbose     : if True, print training progress to stderr.
    """

    latent_dim: int = 64
    n_epochs: int = 60
    batch_size: int = 512
    verbose: bool = False

    @property
    def name(self) -> str:
        return "world_model_consistency"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        try:
            import torch  # noqa: F401
        except ImportError:
            return AnalyzerResult(
                analyzer_name=self.name,
                raw_metrics={"skipped": "torch not installed — pip install torch"},
            )

        if batch.n_episodes < 3:
            return AnalyzerResult(analyzer_name=self.name)

        from calibra.models.robot_jepa import RobotJEPA, RobotJEPAConfig

        cfg = RobotJEPAConfig(
            latent_dim=self.latent_dim,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
        )

        try:
            import sys

            if self.verbose:
                print(
                    f"[world_model] training RobotJEPA "
                    f"(latent_dim={cfg.latent_dim}, epochs={cfg.n_epochs})",
                    file=sys.stderr,
                    flush=True,
                )
            jepa = RobotJEPA(cfg).fit(batch)
            surprise_scores = jepa.score_episodes(batch)
        except Exception as exc:
            return AnalyzerResult(
                analyzer_name=self.name,
                raw_metrics={"error": str(exc)},
            )

        scores_arr = np.array(list(surprise_scores.values()), dtype=float)
        mean_surprise = float(scores_arr.mean())
        high_surprise_frac = float((scores_arr > 0.7).mean())
        learnability = float((scores_arr < 0.3).mean())

        # ── risk flags ────────────────────────────────────────────────────────
        flags: list[RiskFlag] = []

        if high_surprise_frac > 0.30:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="jepa_high_surprise_fraction",
                    observed=ObservedValue(value=high_surprise_frac, unit="fraction"),
                    threshold=0.30,
                    interpretation=(
                        f"{high_surprise_frac:.1%} of episodes have high JEPA surprise "
                        "(world model cannot predict their dynamics)."
                    ),
                    implication=(
                        "Cross-reference with jerk_spike_rate: "
                        "high surprise + high jerk = corrupted episode (prune it); "
                        "high surprise + low jerk = genuinely novel episode (keep it). "
                        "IL policies trained on unpredictable data fail to generalise "
                        "beyond the training distribution."
                    ),
                )
            )

        if learnability < 0.50:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="jepa_world_model_learnability",
                    observed=ObservedValue(value=learnability, unit="fraction"),
                    threshold=0.50,
                    interpretation=(
                        f"Only {learnability:.1%} of episodes are well-predicted "
                        "by the JEPA world model."
                    ),
                    implication=(
                        "A dataset with low world-model learnability produces IL policies "
                        "that memorise specific trajectories but fail on novel start states. "
                        "Run `calibra prune --strategy world-model` to select the "
                        "low-surprise coreset before training."
                    ),
                )
            )

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            raw_metrics={
                "per_episode_surprise": surprise_scores,
                "mean_surprise": mean_surprise,
                "high_surprise_fraction": high_surprise_frac,
                "world_model_learnability": learnability,
                "training_loss_curve": jepa.training_loss_curve,
            },
        )
