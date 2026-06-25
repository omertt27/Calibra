"""
Latent Dynamics Analyzer.

Implements Phase 1 of the World-Model Observability spec:
- Abstraction for StateEncoder and JointStateEncoder.
- Distance-based and multi-scale transition graph construction.
- State-space and transition-space topological coverage entropy.
- Nearest-neighbor density estimation.
- Predictability scoring (Ridge regression coefficient of determination R^2)
  and Transition Energy Outliers (high-residual transitions).
- Causal Action-Effect Mutual Information (via normalized Hilbert-Schmidt Independence Criterion).
- Action Controllability (R^2 mapping of action -> future state delta).
- Redundancy evaluation: State Space Redundancy, Transition Space Redundancy.
- Pruning signal: Exclusive Trajectory (Episode) Novelty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult, RiskFlag, RiskLevel, ObservedValue


# ── State Encoder Abstractions ──────────────────────────────────────────────


class StateEncoder(ABC):
    """
    Abstract base class for mapping raw observations (proprioceptive or visual)
    into a structured latent state vector z_t.
    """

    @abstractmethod
    def encode(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            observations: Dict mapping keys (e.g. 'proprio', 'images') to arrays.
                         Each array has shape (T, ...).
        Returns:
            z: A latent representation array of shape (T, D_latent).
        """
        pass


class JointStateEncoder(StateEncoder):
    """
    Concrete encoder that concatenates selected proprioceptive keys,
    flattening extra dimensions if necessary.
    """

    def __init__(self, select_keys: list[str]):
        self.select_keys = select_keys

    def encode(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        features = []
        for k in self.select_keys:
            if k in observations:
                val = observations[k]
                if val.ndim == 1:
                    features.append(val[:, np.newaxis])
                elif val.ndim == 2:
                    features.append(val)
                else:
                    features.append(val.reshape(val.shape[0], -1))
        if not features:
            raise ValueError(f"None of the select keys {self.select_keys} found in observations.")
        return np.concatenate(features, axis=1)


# ── Numerical / Topological Estimators ───────────────────────────────────────


def _pca_project_kd(Z: np.ndarray, k: int = 3) -> tuple[np.ndarray, float]:
    """Projects Z onto its top-k PCA components. Returns projected data and variance ratio."""
    if len(Z) < k + 1:
        return Z, 1.0
    centered = Z - Z.mean(axis=0)
    try:
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        var = s**2
        total_var = var.sum()
        var_ratio = float(np.sum(var[:k]) / total_var) if total_var > 0 else 1.0
        Z_kd = centered @ Vt[:k].T
        return Z_kd, var_ratio
    except np.linalg.LinAlgError:
        return Z[:, :k], 1.0


def _compute_entropy_2d(Z_2d: np.ndarray, bins: int = 20) -> float:
    """Computes Shannon entropy of the data projected on a 2D grid (in bits)."""
    if len(Z_2d) == 0:
        return 0.0
    hist, _, _ = np.histogram2d(Z_2d[:, 0], Z_2d[:, 1], bins=bins)
    probs = hist.flatten() / hist.sum()
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


def _compute_knn_density(Z: np.ndarray, k: int = 5, max_samples: int = 2000) -> float:
    """Computes the mean distance to the k-th nearest neighbor."""
    N = len(Z)
    if N <= k:
        return 0.0
    if N > max_samples:
        indices = np.random.choice(N, max_samples, replace=False)
        Z_sub = Z[indices]
    else:
        Z_sub = Z

    # Vectorized pairwise L2 distances using (A-B)^2 = A^2 + B^2 - 2AB
    sq_norms = np.sum(Z_sub**2, axis=1)
    dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (Z_sub @ Z_sub.T)
    dists_sq = np.maximum(dists_sq, 0.0)
    dists = np.sqrt(dists_sq)

    sorted_dists = np.sort(dists, axis=1)
    return float(np.mean(sorted_dists[:, k]))


def _compute_normalized_hsic(X: np.ndarray, Y: np.ndarray, max_samples: int = 1000) -> float:
    """
    Computes the normalized Hilbert-Schmidt Independence Criterion (dHSIC) between X and Y.
    Ranges from 0.0 (independent) to 1.0 (strongly dependent).
    """
    n = len(X)
    if n < 5:
        return 0.0
    if n > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, max_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
        n = max_samples

    # Compute pairwise squared distance matrices
    def get_dists_sq(A: np.ndarray) -> np.ndarray:
        norms = np.sum(A**2, axis=1)
        d = norms[:, None] + norms[None, :] - 2.0 * (A @ A.T)
        return np.maximum(d, 0.0)

    dist_X = get_dists_sq(X)
    dist_Y = get_dists_sq(Y)

    # Use median distance heuristic for RBF bandwidth
    med_x = np.median(dist_X)
    med_y = np.median(dist_Y)
    sigma_x = np.sqrt(med_x) if med_x > 0 else 1.0
    sigma_y = np.sqrt(med_y) if med_y > 0 else 1.0

    K = np.exp(-dist_X / (2.0 * sigma_x**2))
    L = np.exp(-dist_Y / (2.0 * sigma_y**2))

    H = np.eye(n) - np.ones((n, n)) / n
    KH = K @ H
    LH = L @ H

    hsic_val = np.trace(KH @ LH)
    hsic_xx = np.trace(KH @ KH)
    hsic_yy = np.trace(LH @ LH)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom > 0:
        return float(hsic_val / denom)
    return 0.0


# ── Latent Dynamics Analyzer ────────────────────────────────────────────────


@dataclass
class LatentDynamicsAnalyzer(Analyzer):
    """
    Evaluates dataset transition graphs and state coverage for world-model predictability.
    """

    proprio_keys: tuple[str, ...] = (
        "proprio",
        "state",
        "joint_state",
        "joint_pos",
        "robot_state",
        "qpos",
        "obs",
    )
    k_nn: int = 5
    alpha: float = 3.0  # multiplier on step size median for epsilon calculation
    ridge_reg: float = 1e-4
    voxel_resolution: float = 0.5  # fraction of std dev for voxel grid sizes

    @property
    def name(self) -> str:
        return "latent_dynamics"

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        if batch.n_episodes == 0:
            return AnalyzerResult(analyzer_name=self.name)

        # 1. Identify proprioceptive key & initialize encoder
        target_key = None
        for key in self.proprio_keys:
            if any(key in ep.observations for ep in batch.episodes):
                target_key = key
                break

        if target_key is None:
            # Skip if no proprioceptive state found
            return AnalyzerResult(
                analyzer_name=self.name,
                raw_metrics={"skipped": "no proprioceptive observations found"},
            )

        encoder = JointStateEncoder([target_key])

        # 2. Extract latent trajectories & calculate self-calibrating step size
        latents_list = []
        step_distances = []
        for ep in batch.episodes:
            if ep.n_steps < 2:
                continue
            try:
                Z_ep = encoder.encode(ep.observations)
                latents_list.append(Z_ep)
                diffs = np.diff(Z_ep, axis=0)
                step_distances.append(np.linalg.norm(diffs, axis=1))
            except Exception:
                continue

        if not latents_list:
            return AnalyzerResult(analyzer_name=self.name)

        all_step_dists = np.concatenate(step_distances) if step_distances else np.array([])
        if len(all_step_dists) > 0:
            d_step = np.median(all_step_dists)
            if d_step == 0:
                d_step = 1e-5
        else:
            d_step = 1e-5

        epsilon = float(self.alpha * d_step)

        # 3. Construct Distance-based and Multi-scale transition datasets
        distance_transitions = []
        transition_episode_ids = []
        all_states = np.concatenate(latents_list, axis=0)

        for ep in batch.episodes:
            try:
                Z_ep = encoder.encode(ep.observations)
                A_ep = ep.actions
                N = len(Z_ep)
                if N < 2:
                    continue

                # A. Distance-based transitions
                t = 0
                while t < N - 1:
                    z_curr = Z_ep[t]
                    t_next = t + 1
                    while t_next < N and np.linalg.norm(z_curr - Z_ep[t_next]) <= epsilon:
                        t_next += 1
                    if t_next < N:
                        mean_a = np.mean(A_ep[t:t_next], axis=0)
                        distance_transitions.append((z_curr, mean_a, Z_ep[t_next]))
                        transition_episode_ids.append(ep.metadata.episode_id)
                    t = t_next
            except Exception:
                continue

        # Fallback to standard 1-step transitions if none exceed the distance threshold
        used_fallback = False
        if not distance_transitions:
            used_fallback = True
            for ep in batch.episodes:
                try:
                    Z_ep = encoder.encode(ep.observations)
                    A_ep = ep.actions
                    N = len(Z_ep)
                    for t in range(N - 1):
                        distance_transitions.append((Z_ep[t], A_ep[t], Z_ep[t + 1]))
                        transition_episode_ids.append(ep.metadata.episode_id)
                except Exception:
                    continue

        if not distance_transitions:
            return AnalyzerResult(
                analyzer_name=self.name,
                raw_metrics={"error": "no valid transitions could be constructed"},
            )

        # Convert transitions to structured matrices
        Z_src = np.array([x[0] for x in distance_transitions])
        A_act = np.array([x[1] for x in distance_transitions])
        Z_tgt = np.array([x[2] for x in distance_transitions])
        Delta_Z = Z_tgt - Z_src

        # 4. Compute Metrics
        # A. State Space Coverage & Redundancy
        Z_2d, state_var_ratio = _pca_project_kd(all_states, k=2)
        state_entropy = _compute_entropy_2d(Z_2d)
        knn_density = _compute_knn_density(all_states, k=self.k_nn)

        # State Redundancy via 3D Voxel Occupancy
        Z_3d, _ = _pca_project_kd(all_states, k=3)
        state_voxels = np.round(Z_3d / self.voxel_resolution).astype(int)
        unique_state_voxels = len(set(map(tuple, state_voxels)))
        state_redundancy = (
            float(1.0 - (unique_state_voxels / len(all_states))) if len(all_states) > 0 else 0.0
        )

        # B. Transition Space Coverage & Redundancy
        Delta_Z_2d, trans_var_ratio = _pca_project_kd(Delta_Z, k=2)
        trans_entropy = _compute_entropy_2d(Delta_Z_2d)

        # Transition Redundancy via 3D PCA Transition Voxel Occupancy
        trans_features = np.concatenate([Z_src, A_act, Delta_Z], axis=1)
        Z_trans_3d, _ = _pca_project_kd(trans_features, k=3)
        trans_voxels = np.round(Z_trans_3d / self.voxel_resolution).astype(int)

        unique_trans_voxels_set = set(map(tuple, trans_voxels))
        total_trans = len(trans_voxels)
        trans_redundancy = (
            float(1.0 - (len(unique_trans_voxels_set) / total_trans)) if total_trans > 0 else 0.0
        )

        # Map each unique transition voxel to the set of episodes that generated it
        voxel_to_episodes = {}
        for idx, voxel in enumerate(trans_voxels):
            v_tuple = tuple(voxel)
            ep_id = transition_episode_ids[idx]
            if v_tuple not in voxel_to_episodes:
                voxel_to_episodes[v_tuple] = set()
            voxel_to_episodes[v_tuple].add(ep_id)

        # Calculate Exclusive Trajectory (Episode) Novelty
        # count of voxels owned exclusively by each episode ID
        episode_exclusive_counts = {ep.metadata.episode_id: 0 for ep in batch.episodes}
        for v_tuple, eps_set in voxel_to_episodes.items():
            if len(eps_set) == 1:
                ep_owner = next(iter(eps_set))
                if ep_owner in episode_exclusive_counts:
                    episode_exclusive_counts[ep_owner] += 1

        total_unique_voxels = len(voxel_to_episodes)
        episode_exclusive_novelty = {}
        for ep_id, count in episode_exclusive_counts.items():
            episode_exclusive_novelty[ep_id] = (
                float(count / total_unique_voxels) if total_unique_voxels > 0 else 0.0
            )

        # C. Local Predictability (Ridge Regression fit: predict Delta_Z from [Z_src, A_act])
        X = np.concatenate([Z_src, A_act], axis=1)
        X_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        try:
            reg_identity = np.eye(X_bias.shape[1]) * self.ridge_reg
            W = np.linalg.solve(X_bias.T @ X_bias + reg_identity, X_bias.T @ Delta_Z)
            preds = X_bias @ W
            residuals = Delta_Z - preds
            res_norms = np.linalg.norm(residuals, axis=1)

            # Predictability R^2
            total_variance = np.sum((Delta_Z - np.mean(Delta_Z, axis=0)) ** 2)
            residual_variance = np.sum(residuals**2)
            if total_variance > 0:
                r2_score = float(1.0 - (residual_variance / total_variance))
            else:
                r2_score = 1.0 if residual_variance == 0 else 0.0
        except Exception:
            r2_score = 0.0
            res_norms = np.zeros(len(Z_src))

        # D. Causal Action-Effect Mutual Information & Controllability R^2
        # Controllability: how well actions explain the transition delta (ignoring current state)
        A_bias = np.concatenate([A_act, np.ones((A_act.shape[0], 1))], axis=1)
        try:
            reg_id_a = np.eye(A_bias.shape[1]) * self.ridge_reg
            W_a = np.linalg.solve(A_bias.T @ A_bias + reg_id_a, A_bias.T @ Delta_Z)
            preds_a = A_bias @ W_a
            res_variance_a = np.sum((Delta_Z - preds_a) ** 2)
            total_variance = np.sum((Delta_Z - np.mean(Delta_Z, axis=0)) ** 2)

            if total_variance > 0:
                action_controllability_r2 = float(1.0 - (res_variance_a / total_variance))
            else:
                action_controllability_r2 = 1.0 if res_variance_a == 0 else 0.0
        except Exception:
            action_controllability_r2 = 0.0

        # Action-Effect Mutual dependency proxy (normalized HSIC)
        action_effect_mi = _compute_normalized_hsic(A_act, Delta_Z)

        # E. Outlier Transitions (Energy Outliers based on residuals)
        outlier_count = 0
        if len(res_norms) > 0:
            median_res = np.median(res_norms)
            mad = np.median(np.abs(res_norms - median_res))
            # avoid dividing by zero
            mad = max(mad, 1e-6)
            outliers = res_norms > (median_res + 3.5 * mad)
            outlier_count = int(np.sum(outliers))
            outlier_fraction = float(outlier_count / len(res_norms))
        else:
            outlier_fraction = 0.0

        raw_metrics = {
            "state_space_entropy_2d": state_entropy,
            "state_knn_density": knn_density,
            "state_pca_variance_ratio": state_var_ratio,
            "state_redundancy": state_redundancy,
            "transition_entropy_2d": trans_entropy,
            "transition_pca_variance_ratio": trans_var_ratio,
            "transition_redundancy": trans_redundancy,
            "dynamics_r2_predictability": r2_score,
            "action_controllability_r2": action_controllability_r2,
            "action_effect_mi": action_effect_mi,
            "epsilon_distance_threshold": epsilon,
            "n_distance_transitions": len(distance_transitions),
            "outlier_transition_fraction": outlier_fraction,
            "outlier_transition_count": outlier_count,
            "used_distance_fallback": used_fallback,
            "per_episode_exclusive_novelty": episode_exclusive_novelty,
        }

        # 5. Formulate Risk Flags
        flags = []

        # State space warning
        if state_entropy < 2.5:
            level = RiskLevel.WARNING
            implication = "State space coverage is extremely restricted; world model will struggle to generalize outside nominal track."
        else:
            level = RiskLevel.OK
            implication = "State space displays healthy distribution across workspaces."

        flags.append(
            RiskFlag(
                level=level,
                metric="latent_state_entropy",
                observed=ObservedValue(value=state_entropy, unit="bits"),
                threshold=2.5,
                interpretation=f"Latent state coverage entropy is {state_entropy:.2f} bits.",
                implication=implication,
            )
        )

        # Transition Redundancy warning
        if trans_redundancy > 0.8:
            red_level = RiskLevel.WARNING
            red_implication = "Over 80% of demonstrations contain redundant dynamics. Consider pruning to save compute."
        else:
            red_level = RiskLevel.OK
            red_implication = (
                "Transition redundancy is low; most demonstrations contain novel dynamics."
            )

        flags.append(
            RiskFlag(
                level=red_level,
                metric="transition_redundancy",
                observed=ObservedValue(value=trans_redundancy, unit="fraction"),
                threshold=0.8,
                interpretation=f"Transition redundancy is {trans_redundancy:.2%}.",
                implication=red_implication,
            )
        )

        # Predictability warning
        if r2_score < 0.2:
            pred_level = RiskLevel.WARNING
            pred_implication = "Downstream world model learnability is low. High noise, actuators slippage, or command-feedback misalignment detected."
        else:
            pred_level = RiskLevel.OK
            pred_implication = "Transitions display high causal predictability, suitable for learning world models."

        flags.append(
            RiskFlag(
                level=pred_level,
                metric="dynamics_predictability_r2",
                observed=ObservedValue(value=r2_score, unit="R^2"),
                threshold=0.2,
                interpretation=f"Dynamics predictability R^2 is {r2_score:.4f}.",
                implication=pred_implication,
            )
        )

        # Causal Action-Effect dependency warning
        if action_effect_mi < 0.1:
            mi_level = RiskLevel.WARNING
        else:
            mi_level = RiskLevel.OK

        flags.append(
            RiskFlag(
                level=mi_level,
                metric="causal_action_effect_mi",
                observed=ObservedValue(value=action_effect_mi, unit="dHSIC"),
                threshold=0.1,
                interpretation=f"Causal action-effect dependency is {action_effect_mi:.4f}.",
                implication=mi_level.value,
            )
        )

        # Outlier transition warning
        if outlier_fraction > 0.05:
            out_level = RiskLevel.WARNING
            out_implication = "Large fraction of transitions violate physical dynamics constraints. Check for simulator/sensor glitching."
        else:
            out_level = RiskLevel.OK
            out_implication = "Transition trajectory is physically smooth and consistent."

        flags.append(
            RiskFlag(
                level=out_level,
                metric="outlier_transition_fraction",
                observed=ObservedValue(value=outlier_fraction, unit="fraction"),
                threshold=0.05,
                interpretation=f"{outlier_count} transition anomalies detected ({outlier_fraction:.2%}).",
                implication=out_implication,
            )
        )

        return AnalyzerResult(analyzer_name=self.name, flags=flags, raw_metrics=raw_metrics)
