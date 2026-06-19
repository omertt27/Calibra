"""
Influence Analyzer.

Computes a lightweight influence score per episode to identify which
demonstrations are most informative for model learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.curation.entropy import compute_trajectory_entropy
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult, RiskFlag, RiskLevel, ObservedValue


@dataclass
class InfluenceAnalyzer(Analyzer):
    """
    Computes an estimated learning influence score per episode.
    """

    novelty_weight: float = 0.4
    contact_weight: float = 0.3
    entropy_weight: float = 0.3

    @property
    def name(self) -> str:
        return "influence"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        episodes = batch.episodes
        n = len(episodes)

        # 1. Action-space stats & novelty (distance to nearest neighbor)
        # Extract action stats (mean and std per dim)
        action_feats = []
        for ep in episodes:
            acts = ep.actions
            if acts.ndim == 1:
                acts = acts[:, np.newaxis]
            mean = np.mean(acts, axis=0)
            std = np.std(acts, axis=0)
            feat = np.concatenate([mean, std])
            action_feats.append(feat)
        
        feat_matrix = np.stack(action_feats, axis=0)
        col_mins = feat_matrix.min(axis=0)
        col_maxs = feat_matrix.max(axis=0)
        scale = col_maxs - col_mins
        scale[scale == 0] = 1.0
        normalized_feats = (feat_matrix - col_mins) / scale

        # Nearest neighbor distances
        nn_dists = []
        if n > 1:
            for i in range(n):
                dists = np.linalg.norm(normalized_feats - normalized_feats[i], axis=1)
                dists[i] = np.inf
                nn_dists.append(float(np.min(dists)))
        else:
            nn_dists = [1.0]

        # 2. Entropy scores
        entropy_scores = [
            compute_trajectory_entropy(ep.actions)
            for ep in episodes
        ]

        # 3. Contact fractions (from phase balance or task structure)
        from calibra.analyzers.phase_balance import _episode_phase_fractions
        from calibra.analyzers.task_structure import _detect_gripper_dims, _collect_actions
        
        actions_all = _collect_actions(batch)
        g_dims = _detect_gripper_dims(actions_all)
        
        contact_scores = []
        for ep in episodes:
            try:
                phases = _episode_phase_fractions(
                    ep, gripper_dims=g_dims, vel_slow_threshold=0.08,
                    action_type="position", min_contact_run=3
                )
                contact_scores.append(phases["contact"])
            except Exception:
                contact_scores.append(0.15)

        # Normalize metrics to [0, 1] range to combine them fairly
        def norm_list(lst):
            arr = np.array(lst)
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-8:
                return np.ones_like(arr)
            return (arr - lo) / (hi - lo)

        norm_novelty = norm_list(nn_dists)
        norm_entropy = norm_list(entropy_scores)
        norm_contact = norm_list(contact_scores)

        influence_scores = (
            self.novelty_weight * norm_novelty +
            self.entropy_weight * norm_entropy +
            self.contact_weight * norm_contact
        )

        per_ep_dict = {
            episodes[i].metadata.episode_id: float(influence_scores[i])
            for i in range(n)
        }

        raw = {
            "per_episode_influence": per_ep_dict,
            "mean_influence": float(np.mean(influence_scores)),
            "per_episode_novelty": {episodes[i].metadata.episode_id: float(norm_novelty[i]) for i in range(n)},
            "per_episode_entropy": {episodes[i].metadata.episode_id: float(norm_entropy[i]) for i in range(n)},
            "per_episode_contact": {episodes[i].metadata.episode_id: float(norm_contact[i]) for i in range(n)},
        }

        sorted_ep = sorted(per_ep_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_ids = [ep_id for ep_id, _ in sorted_ep[:3]]
        
        flag = RiskFlag(
            level=RiskLevel.INFO,
            metric="dataset_influence_score",
            observed=ObservedValue(value=float(np.mean(influence_scores)), unit="influence"),
            interpretation=f"Computed offline learning influence. Top influential episode IDs: {', '.join(top_ids)}",
            implication="Exposing --strategy influence in calibra prune to target informative coreset selections."
        )

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=[flag],
            raw_metrics=raw
        )
