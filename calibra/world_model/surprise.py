"""
calibra.world_model.surprise — lightweight latent prediction-error surprise scoring (v1).

This is the "no GPU, no training loop" baseline for world-model data curation:

    episode observations/actions
        -> encoder (PCA / random projection, closed-form, no torch)
        -> latent z
        -> linear next-latent predictor z_t -> z_{t+1} (closed-form least squares)
        -> surprise = MSE(z_pred, z_next), normalised to [0, 1] across the batch

It answers a different question than IL coreset selection: not "is this episode
behaviourally diverse?" but "does the dataset's own latent dynamics model fail to
predict this episode?" High surprise + a quality pass means the episode contains
dynamics the rest of the dataset doesn't explain — worth keeping for world-model
training. High surprise + a quality fail means the episode is noise, not novelty.

This module has no PyTorch dependency and never runs gradient descent — the whole
pipeline is a handful of matrix decompositions, so it is fast even on 10k+ episodes.
For a trained neural JEPA (higher fidelity, requires torch), see
``calibra.models.robot_jepa`` / ``calibra.surprise``.

Decision rule
-------------
    if quality == FAIL:
        prune
    elif surprise >= threshold:
        keep          # novel dynamics
    else:
        prune          # redundant — well explained by the dataset's own dynamics

Usage
-----
    from calibra.pipeline import Pipeline
    from calibra.ingestion.registry import load
    from calibra.world_model.surprise import curate_for_world_model

    batch = load("/data/demos")
    report = Pipeline().run(batch)
    result = curate_for_world_model(batch, report)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from calibra.schema.episode import Episode, EpisodeBatch
from calibra.schema.report import DiagnosticReport

_PROPRIO_KEYS = ("proprio", "state", "joint_state", "joint_pos", "robot_state", "qpos", "obs")

_SURPRISE_HIGH = 0.60  # normalised threshold for the keep/prune decision rule


# ── encoder + predictor (v1 lightweight baseline — no training loop) ──────────


class LatentEncoder:
    """
    Linear encoder fit in closed form — PCA by default, random projection as a
    fallback for degenerate inputs (too few samples, singular covariance).

    No gradient descent: PCA is one SVD call, random projection is one draw.
    """

    def __init__(self, latent_dim: int = 16, method: str = "pca", seed: int = 0):
        self.latent_dim = latent_dim
        self.method = method
        self.seed = seed
        self._mean: Optional[np.ndarray] = None
        self._components: Optional[np.ndarray] = None  # (in_dim, latent_dim)

    def fit(self, X: np.ndarray) -> "LatentEncoder":
        in_dim = X.shape[1]
        d = max(1, min(self.latent_dim, in_dim))
        self._mean = X.mean(axis=0)
        Xc = X - self._mean

        components: Optional[np.ndarray] = None
        if self.method == "pca" and in_dim >= 1 and len(X) >= 2:
            try:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                components = Vt[:d].T.astype(np.float64)  # (in_dim, d)
            except np.linalg.LinAlgError:
                components = None

        if components is None:
            rng = np.random.default_rng(self.seed)
            W = rng.standard_normal((in_dim, d))
            norms = np.linalg.norm(W, axis=0, keepdims=True)
            components = W / np.clip(norms, 1e-8, None)

        if components.shape[1] < self.latent_dim:
            pad = np.zeros((in_dim, self.latent_dim - components.shape[1]))
            components = np.concatenate([components, pad], axis=1)

        self._components = components
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._components is None:
            raise RuntimeError("Call fit() before transform().")
        return (X - self._mean) @ self._components


class LinearLatentPredictor:
    """
    Closed-form (ridge) least-squares regressor: no gradient descent.

    Fit once via ``np.linalg.solve`` on the normal equations — this is the
    "linear next-latent predictor" in the v1 architecture diagram.
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self._W: Optional[np.ndarray] = None  # (in_dim + 1, out_dim), bias-augmented

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LinearLatentPredictor":
        ones = np.ones((len(X), 1))
        Xb = np.concatenate([X, ones], axis=1)
        n_features = Xb.shape[1]
        A = Xb.T @ Xb + self.ridge * np.eye(n_features)
        B = Xb.T @ Y
        self._W = np.linalg.solve(A, B)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("Call fit() before predict().")
        ones = np.ones((len(X), 1))
        Xb = np.concatenate([X, ones], axis=1)
        return Xb @ self._W


def compute_surprise_scores(
    batch: EpisodeBatch,
    latent_dim: int = 16,
    method: str = "pca",
    seed: int = 0,
) -> dict[str, float]:
    """
    Score every episode by latent next-state prediction error.

    Pipeline: normalise proprio state -> PCA/random-projection encoder -> latent z
    -> closed-form linear predictor (z_t, a_t) -> z_{t+1} -> per-transition MSE,
    averaged per episode and min-max normalised to [0, 1] across the batch.

    Returns {} if there are fewer than 3 episodes or none has a usable
    proprioceptive observation — mirrors the contract of
    ``calibra.models.robot_jepa.score_by_jepa_surprise`` so callers can use
    either scorer interchangeably.
    """
    if batch.n_episodes < 3:
        return {}

    states_l, actions_l, next_l, ids_l = [], [], [], []
    for ep in batch.episodes:
        key = next((k for k in _PROPRIO_KEYS if k in ep.observations), None)
        if key is None:
            continue
        s = ep.observations[key]
        a = ep.actions
        s = s[:, np.newaxis] if s.ndim == 1 else s
        a = a[:, np.newaxis] if a.ndim == 1 else a
        T = min(len(s), len(a))
        if T < 2:
            continue
        s = s[:T].astype(np.float64)
        a = a[:T].astype(np.float64)
        states_l.append(s[:-1])
        actions_l.append(a[:-1])
        next_l.append(s[1:])
        ids_l.extend([ep.metadata.episode_id] * (T - 1))

    if not states_l:
        return {}

    S = np.concatenate(states_l, 0)
    A = np.concatenate(actions_l, 0)
    NS = np.concatenate(next_l, 0)

    s_mean, s_std = S.mean(0), np.clip(S.std(0), 1e-8, None)
    a_mean, a_std = A.mean(0), np.clip(A.std(0), 1e-8, None)
    S_n = (S - s_mean) / s_std
    NS_n = (NS - s_mean) / s_std
    A_n = (A - a_mean) / a_std

    d = max(1, min(latent_dim, S_n.shape[1]))
    encoder = LatentEncoder(latent_dim=d, method=method, seed=seed).fit(S_n)
    Z = encoder.transform(S_n)
    Z_next = encoder.transform(NS_n)

    predictor = LinearLatentPredictor().fit(np.concatenate([Z, A_n], axis=1), Z_next)
    Z_pred = predictor.predict(np.concatenate([Z, A_n], axis=1))

    per_transition_mse = np.mean((Z_pred - Z_next) ** 2, axis=1)

    ids_arr = np.array(ids_l)
    scores: dict[str, float] = {}
    for eid in dict.fromkeys(ids_l):  # preserve first-seen order, dedupe
        scores[eid] = float(per_transition_mse[ids_arr == eid].mean())

    # Episodes with no usable proprio observation score 0 (treated as unsurprising).
    for ep in batch.episodes:
        scores.setdefault(ep.metadata.episode_id, 0.0)

    vals = np.array(list(scores.values()))
    v_min, v_max = vals.min(), vals.max()
    if v_max > v_min:
        for eid in scores:
            scores[eid] = float((scores[eid] - v_min) / (v_max - v_min))

    return scores


# ── reason heuristics ──────────────────────────────────────────────────────────
#
# Cheap, self-contained signals computed from an episode's own action/state
# arrays (no dependency on other analyzers). Each top-surprise episode gets a
# one-line "why" by comparing its signals against the quality-passing population
# and naming whichever one deviates most.

_REASON_LABELS = {
    "action_jerk": "unusual contact dynamics",
    "state_range": "rare state-space excursion",
    "length": "long-horizon recovery",
    "action_energy": "high-energy motion profile",
}

_REASON_Z_THRESHOLD = 1.0  # below this, no signal stands out enough to name


def _episode_signals(ep: Episode) -> dict[str, float]:
    a = ep.actions
    a = a[:, np.newaxis] if a.ndim == 1 else a

    action_energy = float(np.mean(np.linalg.norm(a, axis=1))) if len(a) else 0.0
    action_jerk = float(np.mean(np.linalg.norm(np.diff(a, axis=0), axis=1))) if len(a) > 1 else 0.0

    key = next((k for k in _PROPRIO_KEYS if k in ep.observations), None)
    s = ep.observations.get(key) if key else None
    if s is not None and len(s):
        s = s[:, np.newaxis] if s.ndim == 1 else s
        state_range = float(np.mean(np.ptp(s, axis=0)))
    else:
        state_range = 0.0

    return {
        "action_energy": action_energy,
        "action_jerk": action_jerk,
        "state_range": state_range,
        "length": float(ep.n_steps),
    }


def infer_reason(ep: Episode, population: list[dict[str, float]]) -> str:
    """One-line heuristic explanation for why an episode scored high surprise."""
    signals = _episode_signals(ep)
    if not population:
        return "high dynamics prediction error in latent space"

    best_key, best_z = None, 0.0
    for key, value in signals.items():
        vals = np.array([p[key] for p in population])
        std = vals.std()
        if std < 1e-8:
            continue
        z = abs((value - vals.mean()) / std)
        if z > best_z:
            best_key, best_z = key, z

    if best_key is None or best_z < _REASON_Z_THRESHOLD:
        return "high dynamics prediction error in latent space"
    return _REASON_LABELS[best_key]


# ── result schema ──────────────────────────────────────────────────────────────


@dataclass
class EpisodeSurprise:
    episode_id: str
    surprise: float
    reason: str


@dataclass
class WorldModelCurationResult:
    dataset_name: str
    n_original: int
    n_quality_fail: int
    n_kept: int
    n_pruned_redundant: int
    keep_episode_ids: list[str]
    quality_fail_ids: list[str]
    redundant_ids: list[str]
    surprise_scores: dict[str, float]
    top_novel: list[EpisodeSurprise] = field(default_factory=list)
    surprise_threshold: float = _SURPRISE_HIGH
    method: str = "quality_filter + pca_linear_latent_surprise"

    def summary(self, top: int = 10) -> str:
        lines = [
            "WORLD-MODEL CURATION SUMMARY",
            "",
            f"Original episodes: {self.n_original}",
            f"Quality failures: {self.n_quality_fail}",
            f"High-surprise kept: {self.n_kept}",
            f"Low-surprise pruned: {self.n_pruned_redundant}",
            "",
            "Top novel episodes:",
        ]
        for es in self.top_novel[:top]:
            lines.append(f'  {es.episode_id}  surprise={es.surprise:.2f}  reason="{es.reason}"')
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "n_original": self.n_original,
            "n_quality_fail": self.n_quality_fail,
            "n_kept": self.n_kept,
            "n_pruned_redundant": self.n_pruned_redundant,
            "keep_episode_ids": self.keep_episode_ids,
            "quality_fail_ids": self.quality_fail_ids,
            "redundant_ids": self.redundant_ids,
            "surprise_scores": {k: round(v, 4) for k, v in self.surprise_scores.items()},
            "surprise_threshold": self.surprise_threshold,
            "top_novel": [
                {"episode_id": e.episode_id, "surprise": round(e.surprise, 4), "reason": e.reason}
                for e in self.top_novel
            ],
            "method": self.method,
        }


# ── decision rule ───────────────────────────────────────────────────────────────


def curate_for_world_model(
    batch: EpisodeBatch,
    report: DiagnosticReport,
    surprise_threshold: float = _SURPRISE_HIGH,
    latent_dim: int = 16,
    top: int = 10,
    quality_kwargs: Optional[dict] = None,
) -> WorldModelCurationResult:
    """
    Apply the world-model decision rule to every episode:

        quality FAIL          -> prune
        quality PASS + HIGH surprise -> keep (novel dynamics)
        quality PASS + LOW  surprise -> prune (redundant)

    Quality filtering reuses ``CoresetSelector``'s existing Stage 1 thresholds
    (``quality_kwargs`` forwards overrides like ``max_spike_rate``) rather than
    re-implementing them. Surprise comes from the lightweight latent predictor
    in this module — no torch, no training loop.
    """
    from calibra.pruning import CoresetSelector

    selector = CoresetSelector(keep_fraction=1.0, quality_only=True, **(quality_kwargs or {}))
    quality_result = selector.select(batch, report)
    quality_fail_ids = list(quality_result.quality_fail_ids)
    quality_fail_set = set(quality_fail_ids)

    quality_pass_ids = [
        ep.metadata.episode_id for ep in batch.episodes if ep.metadata.episode_id not in quality_fail_set
    ]

    surprise_scores = compute_surprise_scores(batch, latent_dim=latent_dim)

    keep_ids, redundant_ids = [], []
    for eid in quality_pass_ids:
        if surprise_scores.get(eid, 0.0) >= surprise_threshold:
            keep_ids.append(eid)
        else:
            redundant_ids.append(eid)

    keep_ids.sort(key=lambda eid: surprise_scores.get(eid, 0.0), reverse=True)

    by_id = {ep.metadata.episode_id: ep for ep in batch.episodes}
    population = [_episode_signals(by_id[eid]) for eid in quality_pass_ids if eid in by_id]
    top_novel = [
        EpisodeSurprise(
            episode_id=eid,
            surprise=round(surprise_scores.get(eid, 0.0), 4),
            reason=infer_reason(by_id[eid], population),
        )
        for eid in keep_ids[:top]
        if eid in by_id
    ]

    return WorldModelCurationResult(
        dataset_name=report.dataset_name,
        n_original=batch.n_episodes,
        n_quality_fail=len(quality_fail_ids),
        n_kept=len(keep_ids),
        n_pruned_redundant=len(redundant_ids),
        keep_episode_ids=keep_ids,
        quality_fail_ids=quality_fail_ids,
        redundant_ids=redundant_ids,
        surprise_scores=surprise_scores,
        top_novel=top_novel,
        surprise_threshold=surprise_threshold,
    )


def format_world_model_summary(result, batch: EpisodeBatch, top: int = 10) -> str:
    """
    Render a "WORLD-MODEL CURATION SUMMARY" banner from an existing
    ``calibra.pruning.PruningResult`` (strategy="world-model") without
    recomputing surprise — reuses ``result.diversity_scores`` as the surprise
    scores, whichever scorer produced them (trained JEPA or this lightweight
    baseline).
    """
    by_id = {ep.metadata.episode_id: ep for ep in batch.episodes}
    quality_pass_ids = result.keep_episode_ids + result.diversity_pruned_ids
    population = [_episode_signals(by_id[eid]) for eid in quality_pass_ids if eid in by_id]

    sorted_keep = sorted(
        result.keep_episode_ids,
        key=lambda eid: result.diversity_scores.get(eid, 0.0),
        reverse=True,
    )

    lines = [
        "WORLD-MODEL CURATION SUMMARY",
        "",
        f"Original episodes: {result.n_original}",
        f"Quality failures: {result.n_quality_failures}",
        f"High-surprise kept: {result.n_kept}",
        f"Low-surprise pruned: {result.n_diversity_pruned}",
        "",
        "Top novel episodes:",
    ]
    for eid in sorted_keep[:top]:
        if eid not in by_id:
            continue
        surprise = result.diversity_scores.get(eid, 0.0)
        reason = infer_reason(by_id[eid], population)
        lines.append(f'  {eid}  surprise={surprise:.2f}  reason="{reason}"')
    return "\n".join(lines)