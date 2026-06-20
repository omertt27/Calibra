"""
Self-Supervised (SSL) Trajectory Embedding Analyzer.

Projects state-action trajectories into an embedding space to measure
global trajectory novelty and detect anomalous demonstration outliers.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult, RiskFlag, RiskLevel, ObservedValue


class SSLTrajectoryEmbedderAnalyzer(Analyzer):
    """
    Trajectory-level embedder analyzing structural novelty.
    
    Uses sequence projection to map variable-length states/actions to 
    fixed-size vectors, computing pairwise cosine distance to locate
    behavioral outliers and measure dataset representation coverage.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        self.embedding_dim = embedding_dim

    @property
    def name(self) -> str:
        return "ssl_embed"

    def _embed_episode(self, actions: np.ndarray, proprio: Optional[np.ndarray]) -> np.ndarray:
        """
        Map a variable-length trajectory to a fixed embedding vector.
        Uses a deterministic projection over temporal steps.
        """
        # Combine actions and proprioception states if available
        features = actions
        if proprio is not None and proprio.ndim == actions.ndim:
            features = np.concatenate([features, proprio], axis=-1)

        # Pad or interpolate to a standard length
        T, D = features.shape
        if T == 0:
            return np.zeros(self.embedding_dim)

        # Create a deterministic random projection matrix to map features
        rng = np.random.default_rng(seed=42 + D)
        proj = rng.normal(size=(D, self.embedding_dim))
        projected = features @ proj  # (T, embedding_dim)

        # Apply temporal aggregation (mean + std deviation to capture dynamics)
        mean_emb = np.mean(projected, axis=0)
        std_emb = np.std(projected, axis=0)
        emb = mean_emb + std_emb

        # Normalize to unit length
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-8 else emb

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        embeddings = []
        for ep in batch.episodes:
            proprio = ep.observations.get("proprio")
            emb = self._embed_episode(ep.actions, proprio)
            embeddings.append(emb)

        if not embeddings or len(embeddings) < 2:
            return AnalyzerResult(analyzer_name=self.name)

        embs = np.array(embeddings)  # (N, D)

        # Calculate cosine similarity matrix
        sim_matrix = embs @ embs.T  # (N, N)
        dist_matrix = 1.0 - sim_matrix
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

        # Compute novelty score for each episode (distance to its nearest neighbor)
        # We ignore the self-distance (which is 0.0) by filling the diagonal with infinity
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = np.min(dist_matrix, axis=0)
        np.fill_diagonal(dist_matrix, 0.0) # restore diagonal

        # Compute global centroid distance to find outliers
        mean_emb = np.mean(embs, axis=0)
        mean_emb_norm = mean_emb / np.linalg.norm(mean_emb) if np.linalg.norm(mean_emb) > 1e-8 else mean_emb
        centroid_distances = 1.0 - (embs @ mean_emb_norm)

        # Calculate MAD (Median Absolute Deviation) outliers on centroid distances
        median_dist = np.median(centroid_distances)
        mad = np.median(np.abs(centroid_distances - median_dist))
        mad = max(mad, 1e-6)
        
        # Outliers are trajectories with distance > median + 3 * (1.4826 * MAD)
        threshold = median_dist + 3 * (1.4826 * mad)
        outliers = centroid_distances > threshold
        outlier_indices = np.where(outliers)[0]
        outlier_fraction = float(len(outlier_indices) / len(batch.episodes))

        flags = []
        if outlier_fraction > 0.05:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="ssl_trajectory_outliers",
                    observed=ObservedValue(value=outlier_fraction, unit="fraction"),
                    threshold=0.05,
                    interpretation=f"{outlier_fraction:.1%} of episodes are trajectory-level outliers.",
                    implication=(
                        "A high fraction of outlier trajectories suggests inconsistent "
                        "demonstration quality, hardware calibration drift, or operator errors. "
                        "Review these outlier episodes before policy training."
                    ),
                    affected_fraction=outlier_fraction,
                )
            )

        # Compute dataset coverage score (average distance to nearest neighbor)
        mean_nearest_dist = float(np.mean(nearest_distances[np.isfinite(nearest_distances)]))
        
        # If coverage distance is very high, it means episodes are sparse and fragmented
        if mean_nearest_dist > 0.6:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="ssl_dataset_sparsity",
                    observed=ObservedValue(value=mean_nearest_dist, unit="distance"),
                    threshold=0.6,
                    interpretation=f"Dataset average nearest-neighbor embedding distance is {mean_nearest_dist:.3f}.",
                    implication=(
                        "High sparsity indicates that demonstrations are too disjoint, "
                        "which makes policy interpolation difficult. Consider collecting more intermediate episodes."
                    ),
                )
            )

        raw_metrics = {
            "per_episode_ssl_novelty": nearest_distances.tolist(),
            "per_episode_ssl_centroid_dist": centroid_distances.tolist(),
            "mean_nearest_distance": mean_nearest_dist,
            "outlier_indices": outlier_indices.tolist(),
        }

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            raw_metrics=raw_metrics,
        )
