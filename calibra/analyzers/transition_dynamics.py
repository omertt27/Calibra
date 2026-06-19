"""
Transition Dynamics Analyzer.

Analyzes environmental state-action transition dynamics to measure the physical
complexity and coverage of transitions in a robot dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult, RiskFlag, RiskLevel, ObservedValue


@dataclass
class TransitionDynamicsAnalyzer(Analyzer):
    """
    Evaluates state-action transition coverage and fits a forward dynamics model.
    """

    @property
    def name(self) -> str:
        return "transition_dynamics"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        episodes = batch.episodes
        n = len(episodes)

        states_list = []
        actions_list = []
        next_states_list = []
        episode_dynamics_errors = {}
        episode_transition_entropy = {}

        # Extract transitions
        for ep in episodes:
            states = ep.observations.get("proprio")
            acts = ep.actions
            if states is None or len(states) < 2:
                continue
                
            t_max = min(len(states) - 1, len(acts))
            s_t = states[:t_max]
            a_t = acts[:t_max]
            s_next = states[1:t_max+1]
            
            states_list.append(s_t)
            actions_list.append(a_t)
            next_states_list.append(s_next)

        if not states_list:
            return AnalyzerResult(analyzer_name=self.name)

        # Concatenate for full batch
        X_s = np.concatenate(states_list, axis=0)
        X_a = np.concatenate(actions_list, axis=0)
        Y = np.concatenate(next_states_list, axis=0)

        # Fit a simple forward dynamics model: S_{t+1} = S_t + W * [S_t, A_t] + b
        state_diff = Y - X_s
        features = np.concatenate([X_s, X_a], axis=1)

        try:
            reg = 1e-4
            identity = np.eye(features.shape[1])
            W = np.linalg.solve(features.T @ features + reg * identity, features.T @ state_diff)
            predictions = X_s + features @ W
            residuals = Y - predictions
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
        except Exception:
            rmse = 0.1
            W = np.zeros((features.shape[1], Y.shape[1]))

        # Calculate per-episode transition prediction errors (energy)
        for i, ep in enumerate(episodes):
            states = ep.observations.get("proprio")
            acts = ep.actions
            if states is None or len(states) < 2:
                episode_dynamics_errors[ep.metadata.episode_id] = 0.0
                episode_transition_entropy[ep.metadata.episode_id] = 0.0
                continue
                
            t_max = min(len(states) - 1, len(acts))
            s_t = states[:t_max]
            a_t = acts[:t_max]
            s_next = states[1:t_max+1]
            
            feats = np.concatenate([s_t, a_t], axis=1)
            pred = s_t + feats @ W
            errs = np.linalg.norm(s_next - pred, axis=1)
            mean_err = float(np.mean(errs))
            episode_dynamics_errors[ep.metadata.episode_id] = mean_err
            
            # Compute direction entropy of transition vectors
            diffs = s_next - s_t
            norms = np.linalg.norm(diffs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dirs = diffs / norms
            
            quadrants = np.sign(dirs)
            _, counts = np.unique(quadrants, axis=0, return_counts=True)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log2(probs))
            episode_transition_entropy[ep.metadata.episode_id] = float(ent)

        raw = {
            "overall_transition_rmse": rmse,
            "per_episode_dynamics_error": episode_dynamics_errors,
            "per_episode_transition_entropy": episode_transition_entropy,
            "mean_transition_entropy": float(np.mean(list(episode_transition_entropy.values())))
        }

        flag = RiskFlag(
            level=RiskLevel.INFO,
            metric="dynamics_prediction_rmse",
            observed=ObservedValue(value=rmse, unit="rmse"),
            interpretation=f"Fit linear transition dynamics. Dynamics RMSE: {rmse:.4g}",
            implication="High RMSE indicates complex non-linear physics or high transition noise."
        )

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=[flag],
            raw_metrics=raw
        )
