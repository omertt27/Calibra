"""
calibra.llm.select — coreset selection for SFT / instruction-tuning datasets.

The text analogue of `calibra.pruning.CoresetSelector`: a two-stage pipeline that
first drops low-quality examples (low coherence, high repetition, boilerplate,
too short), then selects a diverse coreset from the quality-passing pool via
greedy max-coverage over per-example embeddings — reusing the exact same
farthest-point algorithm the robotics pipeline uses over action-space stats.

Usage
-----
    from calibra.llm.fingerprint import compute_fingerprints
    from calibra.llm.select import SFTCoresetSelector

    fp = compute_fingerprints(instructions, outputs)
    selector = SFTCoresetSelector(keep_fraction=0.3)
    result = selector.select(fp)

    print(result.summary())
    kept_instructions = [instructions[i] for i in result.keep_indices]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibra.llm.fingerprint import FingerprintResult
from calibra.pruning import _approximate_max_coverage, _greedy_max_coverage

# ── quality thresholds (conservative defaults) ────────────────────────────────

_DEFAULT_MIN_COHERENCE = 0.10
_DEFAULT_MAX_REPETITION_RATE = 0.40
_DEFAULT_MAX_TEMPLATE_RATIO = 0.50
_DEFAULT_MIN_LENGTH_WORDS = 5


def _avg_nn_distance(embs: np.ndarray, sample_size: int = 2000) -> float:
    """Mean nearest-neighbour distance in embedding space — proxy for spread/diversity."""
    n = len(embs)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(42)
    if n > sample_size:
        embs = embs[rng.choice(n, sample_size, replace=False)]
    x2 = np.sum(embs**2, axis=1, keepdims=True)
    dists2 = np.maximum(x2 + x2.T - 2.0 * (embs @ embs.T), 0.0)
    np.fill_diagonal(dists2, np.inf)
    return float(np.sqrt(np.min(dists2, axis=1)).mean())


@dataclass
class SFTSelectionResult:
    """
    Output of SFTCoresetSelector.select().

    Attributes
    ----------
    keep_indices            : indices (into the original example list) to retain.
    quality_fail_indices    : indices removed in Stage 1 (quality threshold failures).
    diversity_pruned_indices: indices removed in Stage 2 (redundant under max-coverage).
    quality_scores          : per-index composite quality score (lower = cleaner).
    diversity_scores        : per-index min-distance-to-selected score after Stage 2.
    aggregate_fingerprint   : corpus-level fingerprint over the *kept* subset — feeds
                              `calibra.llm.predict.predict_sft_outcome` and outcome recording.
    n_original, n_kept, n_quality_failures, n_diversity_pruned, keep_fraction_actual, method
    """

    keep_indices: list[int]
    quality_fail_indices: list[int]
    diversity_pruned_indices: list[int]
    quality_scores: dict[int, float]
    diversity_scores: dict[int, float]
    aggregate_fingerprint: dict[str, float]
    n_original: int
    n_kept: int
    n_quality_failures: int
    n_diversity_pruned: int
    keep_fraction_actual: float
    method: str = "quality_filter + greedy_max_coverage"

    def summary(self) -> str:
        lines = [
            "━" * 56,
            "  CALIBRA SFT SELECTION SUMMARY",
            "━" * 56,
            f"  Original examples  : {self.n_original}",
            f"  Quality failures   : {self.n_quality_failures}  (removed in Stage 1)",
            f"  Diversity pruned   : {self.n_diversity_pruned}  (removed in Stage 2)",
            f"  Coreset size       : {self.n_kept}  ({self.keep_fraction_actual:.1%} of original)",
            f"  Method             : {self.method}",
            "─" * 56,
        ]
        for k, v in self.aggregate_fingerprint.items():
            lines.append(f"  {k:<24}: {v:.4f}")
        lines.append("━" * 56)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_original": self.n_original,
            "n_kept": self.n_kept,
            "n_quality_failures": self.n_quality_failures,
            "n_diversity_pruned": self.n_diversity_pruned,
            "keep_fraction_actual": self.keep_fraction_actual,
            "keep_indices": self.keep_indices,
            "quality_fail_indices": self.quality_fail_indices,
            "diversity_pruned_indices": self.diversity_pruned_indices,
            "quality_scores": {str(k): v for k, v in self.quality_scores.items()},
            "diversity_scores": {str(k): v for k, v in self.diversity_scores.items()},
            "aggregate_fingerprint": self.aggregate_fingerprint,
        }


def _aggregate_fingerprint(fp: FingerprintResult, keep_indices: list[int]) -> dict[str, float]:
    if not keep_indices:
        return {
            "mean_coherence": 0.0,
            "repetition_rate": 0.0,
            "template_ratio": 0.0,
            "mean_response_length": 0.0,
            "diversity_nn_dist": 0.0,
        }
    idx = np.array(keep_indices)
    return {
        "mean_coherence": float(np.mean(fp.coherence[idx])),
        "repetition_rate": float(np.mean(fp.repetition_rate[idx])),
        "template_ratio": float(np.mean(fp.template_ratio[idx])),
        "mean_response_length": float(np.mean(fp.response_length_words[idx])),
        "diversity_nn_dist": _avg_nn_distance(fp.embeddings[idx]),
    }


@dataclass
class SFTCoresetSelector:
    """
    Two-stage coreset selection engine for SFT/instruction-tuning data.

    Parameters
    ----------
    keep_fraction : target fraction of examples to keep. Applied after quality
                    filtering; the actual fraction of the original dataset will
                    be <= keep_fraction if quality failures reduce the pool.
    min_coherence : Stage 1 threshold — examples below this instruction/response
                    cosine similarity are removed. Default 0.10.
    max_repetition_rate : Stage 1 threshold — bigram repetition rate. Default 0.40.
    max_template_ratio : Stage 1 threshold — boilerplate-opener word fraction. Default 0.50.
    min_length_words : Stage 1 threshold — minimum response length in words. Default 5.
    quality_only : skip Stage 2; return all quality-passing examples without
                   diversity selection.
    use_approximate : use the MiniBatch approximate greedy coverage (handles 100k+
                      examples); exact greedy k-center otherwise.
    batch_size : batch size for the approximate selector. Default 1000.
    """

    keep_fraction: float = 0.5
    min_coherence: float = _DEFAULT_MIN_COHERENCE
    max_repetition_rate: float = _DEFAULT_MAX_REPETITION_RATE
    max_template_ratio: float = _DEFAULT_MAX_TEMPLATE_RATIO
    min_length_words: int = _DEFAULT_MIN_LENGTH_WORDS
    quality_only: bool = False
    use_approximate: bool = False
    batch_size: int = 1000

    def select(self, fp: FingerprintResult) -> SFTSelectionResult:
        n = len(fp.coherence)

        fail_mask = (
            (fp.coherence < self.min_coherence)
            | (fp.repetition_rate > self.max_repetition_rate)
            | (fp.template_ratio > self.max_template_ratio)
            | (fp.response_length_words < self.min_length_words)
        )
        quality_fail_indices = np.nonzero(fail_mask)[0].tolist()
        quality_pass_indices = np.nonzero(~fail_mask)[0].tolist()

        quality_scores = {
            i: round(
                0.4 * float(fp.repetition_rate[i])
                + 0.3 * float(fp.template_ratio[i])
                + 0.3 * max(0.0, 1.0 - float(fp.coherence[i])),
                6,
            )
            for i in range(n)
        }

        if not quality_pass_indices:
            return SFTSelectionResult(
                keep_indices=[],
                quality_fail_indices=quality_fail_indices,
                diversity_pruned_indices=[],
                quality_scores=quality_scores,
                diversity_scores={},
                aggregate_fingerprint=_aggregate_fingerprint(fp, []),
                n_original=n,
                n_kept=0,
                n_quality_failures=len(quality_fail_indices),
                n_diversity_pruned=0,
                keep_fraction_actual=0.0,
            )

        k = max(1, round(n * self.keep_fraction))

        if self.quality_only or k >= len(quality_pass_indices):
            keep_indices = quality_pass_indices
            diversity_pruned_indices: list[int] = []
            diversity_scores: dict[int, float] = {}
        else:
            features = fp.embeddings[quality_pass_indices]
            if self.use_approximate:
                selected_local = _approximate_max_coverage(features, k, self.batch_size)
            else:
                selected_local = _greedy_max_coverage(features, k)

            selected_global = [quality_pass_indices[i] for i in selected_local]
            selected_set = set(selected_global)

            keep_indices = selected_global
            diversity_pruned_indices = [i for i in quality_pass_indices if i not in selected_set]
            diversity_scores = _diversity_score_map(features, quality_pass_indices, selected_local)

        return SFTSelectionResult(
            keep_indices=keep_indices,
            quality_fail_indices=quality_fail_indices,
            diversity_pruned_indices=diversity_pruned_indices,
            quality_scores=quality_scores,
            diversity_scores=diversity_scores,
            aggregate_fingerprint=_aggregate_fingerprint(fp, keep_indices),
            n_original=n,
            n_kept=len(keep_indices),
            n_quality_failures=len(quality_fail_indices),
            n_diversity_pruned=len(diversity_pruned_indices),
            keep_fraction_actual=len(keep_indices) / max(n, 1),
            method=(
                "quality_filter + approximate_minibatch_coverage"
                if self.use_approximate
                else "quality_filter + greedy_max_coverage"
            ),
        )


def _diversity_score_map(
    features: np.ndarray,
    candidate_indices: list[int],
    selected_local: list[int],
) -> dict[int, float]:
    """Return {original_index: min_distance_to_selected_set} for all candidates."""
    selected_set = sorted(set(selected_local))
    selected_feats = features[selected_set]

    x2 = np.sum(features**2, axis=1, keepdims=True)
    y2 = np.sum(selected_feats**2, axis=1, keepdims=True).T
    xy = features @ selected_feats.T
    dists2 = np.maximum(x2 + y2 - 2 * xy, 0.0)
    min_dists = np.min(np.sqrt(dists2), axis=1)

    return {
        candidate_indices[local_idx]: float(min_dists[local_idx])
        for local_idx in range(len(candidate_indices))
    }
