"""Tests for the coverage entropy analyzer."""
from __future__ import annotations

import numpy as np
import pytest

from calibra.analyzers.coverage import (
    CoverageEntropyAnalyzer,
    _collect_actions,
    _collect_state,
    _is_bimodal_heuristic,
    _marginal_entropy_bits,
    _pca_top_k_fraction,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_episode(
    n_steps: int,
    action_range: float = 1.0,
    action_dim: int = 4,
    ep_id: str = "ep_0",
    with_proprio: bool = True,
) -> Episode:
    rng = np.random.default_rng(int(ep_id.split("_")[-1]) if "_" in ep_id else 0)
    ts = np.arange(n_steps, dtype=np.float64) * 0.1
    actions = rng.uniform(-action_range, action_range, (n_steps, action_dim)).astype(np.float32)
    obs: dict = {}
    if with_proprio:
        obs["proprio"] = rng.uniform(-1, 1, (n_steps, 6)).astype(np.float32)
    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=ts,
        observations=obs,
        actions=actions,
    )


def _make_collapsed_episode(
    n_steps: int = 200,
    action_dim: int = 4,
    ep_id: str = "ep_0",
) -> Episode:
    """Near-constant actions — very low diversity."""
    rng = np.random.default_rng(0)
    ts = np.arange(n_steps, dtype=np.float64) * 0.1
    actions = np.full((n_steps, action_dim), 0.5, dtype=np.float32)
    actions += rng.normal(0, 0.005, actions.shape).astype(np.float32)
    return Episode(
        metadata=EpisodeMetadata(episode_id=ep_id),
        timestamps=ts,
        observations={"proprio": rng.uniform(-1, 1, (n_steps, 6)).astype(np.float32)},
        actions=actions,
    )


def _batch_of(episodes) -> EpisodeBatch:
    return EpisodeBatch(episodes=list(episodes), dataset_name="test",
                        format="hdf5", source_path="/tmp/x.h5")


def _diverse_batch(n_eps: int = 10, n_steps: int = 200) -> EpisodeBatch:
    return _batch_of([_make_episode(n_steps, ep_id=f"ep_{i}") for i in range(n_eps)])


def _collapsed_batch(n_eps: int = 10, n_steps: int = 200) -> EpisodeBatch:
    return _batch_of([_make_collapsed_episode(n_steps, ep_id=f"ep_{i}")
                      for i in range(n_eps)])


# ── unit tests: marginal entropy ─────────────────────────────────────────────

class TestMarginalEntropyBits:
    def test_uniform_has_higher_entropy_than_constant(self):
        rng = np.random.default_rng(0)
        uniform = rng.uniform(0, 1, (1000, 4))
        constant = np.full((1000, 4), 0.5)
        assert _marginal_entropy_bits(uniform) > _marginal_entropy_bits(constant)

    def test_constant_has_near_zero_entropy(self):
        data = np.full((500, 3), 0.5)
        assert _marginal_entropy_bits(data) == pytest.approx(0.0)

    def test_1d_input(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, 500)
        entropy = _marginal_entropy_bits(data)
        assert entropy > 0

    def test_narrow_range_lower_entropy_with_reference(self):
        # Without a reference range, binning self-adapts → similar entropy.
        # WITH a reference range, a narrow distribution occupies far fewer
        # bins than a wide one → clear entropy separation.
        rng = np.random.default_rng(0)
        narrow = rng.uniform(0.45, 0.55, (1000, 3))  # occupies 5% of [-1, 1]
        wide   = rng.uniform(-1.0, 1.0,  (1000, 3))  # occupies 100% of [-1, 1]
        ref = (-1.0, 1.0)
        assert _marginal_entropy_bits(wide, data_range=ref) > \
               _marginal_entropy_bits(narrow, data_range=ref)


# ── unit tests: PCA variance ─────────────────────────────────────────────────

class TestPCATopKFraction:
    def test_rank1_data_has_fraction_near_1(self):
        rng = np.random.default_rng(0)
        v = rng.random(500)
        data = np.column_stack([v, 2 * v, -0.5 * v, 3 * v])
        frac, _ = _pca_top_k_fraction(data, k=2)
        assert frac > 0.99

    def test_isotropic_data_has_low_concentration(self):
        rng = np.random.default_rng(0)
        data = rng.random((1000, 8))
        frac, _ = _pca_top_k_fraction(data, k=2)
        # 2/8 dims → expect roughly 0.25 in uniform case
        assert frac < 0.5

    def test_returns_per_pc_fractions(self):
        rng = np.random.default_rng(0)
        data = rng.random((200, 4))
        _, explained = _pca_top_k_fraction(data, k=2)
        assert len(explained) == 2
        assert all(0 <= v <= 1 for v in explained)

    def test_1d_action_space(self):
        data = np.random.default_rng(0).random((100, 1))
        frac, _ = _pca_top_k_fraction(data, k=2)
        # Should gracefully return info or skip
        assert frac >= 0.0


# ── unit tests: bimodality heuristic ─────────────────────────────────────────

class TestBimodalHeuristic:
    def test_unimodal_is_not_bimodal(self):
        lengths = np.full(20, 50.0)
        assert not _is_bimodal_heuristic(lengths)

    def test_bimodal_distribution_detected(self):
        rng = np.random.default_rng(0)
        short = rng.normal(30, 2, 15)
        long_  = rng.normal(100, 3, 15)
        lengths = np.concatenate([short, long_])
        assert _is_bimodal_heuristic(lengths)

    def test_too_few_episodes_not_bimodal(self):
        lengths = np.array([30.0, 100.0, 30.0])
        assert not _is_bimodal_heuristic(lengths)

    def test_low_cv_not_bimodal(self):
        rng = np.random.default_rng(0)
        lengths = rng.normal(50, 2, 30)
        assert not _is_bimodal_heuristic(lengths)


# ── unit tests: data collection helpers ──────────────────────────────────────

class TestCollectActions:
    def test_stacks_across_episodes(self):
        eps = [_make_episode(50, ep_id=f"ep_{i}") for i in range(3)]
        batch = _batch_of(eps)
        actions = _collect_actions(batch)
        assert actions.shape == (150, 4)

    def test_empty_batch(self):
        actions = _collect_actions(_batch_of([]))
        assert actions.shape[0] == 0


class TestCollectState:
    def test_finds_proprio_key(self):
        batch = _diverse_batch(5, 50)
        key, state = _collect_state(batch, ("proprio",))
        assert key == "proprio"
        assert state is not None
        assert state.shape[0] == 5 * 50

    def test_returns_none_for_missing_key(self):
        batch = _diverse_batch(3, 30)
        key, state = _collect_state(batch, ("nonexistent_key",))
        assert key is None
        assert state is None


# ── integration: CoverageEntropyAnalyzer ─────────────────────────────────────

class TestCoverageEntropyAnalyzerDiverse:
    def test_diverse_batch_ok_flags(self):
        batch = _diverse_batch(10, 300)
        result = CoverageEntropyAnalyzer().analyze(batch)
        bad = [f for f in result.flags
               if f.level == RiskLevel.CRITICAL
               and not np.isnan(f.observed.value)]
        assert bad == [], f"Unexpected CRITICAL on diverse data: {[f.metric for f in bad]}"

    def test_analyzer_name(self):
        result = CoverageEntropyAnalyzer().analyze(_diverse_batch(3, 50))
        assert result.analyzer_name == "coverage_entropy"

    def test_raw_metrics_populated(self):
        result = CoverageEntropyAnalyzer().analyze(_diverse_batch(5, 100))
        assert "action_entropy" in result.raw_metrics
        assert "pca_variance" in result.raw_metrics


class TestCoverageEntropyAnalyzerCollapsed:
    def test_collapsed_batch_flags_low_entropy_with_reference_range(self):
        # Without action_range, entropy self-normalises to observed range
        # and cannot distinguish collapsed from diverse data.
        # WITH action_range=(-1, 1), collapsed data (all near 0.5±0.005)
        # occupies <2% of the bins → genuinely low entropy.
        batch = _collapsed_batch(10, 300)
        result = CoverageEntropyAnalyzer(
            action_range=(-1.0, 1.0),
            action_entropy_warning=3.5,
            action_entropy_critical=2.0,
        ).analyze(batch)
        entropy_flags = [
            f for f in result.flags
            if "action_entropy" in f.metric and not np.isnan(f.observed.value)
        ]
        assert entropy_flags, "Expected action_entropy flag"
        assert any(f.level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
                   for f in entropy_flags)

    def test_diverse_batch_ok_entropy_with_reference_range(self):
        batch = _diverse_batch(10, 300)
        result = CoverageEntropyAnalyzer(
            action_range=(-1.0, 1.0),
            action_entropy_warning=3.5,
        ).analyze(batch)
        entropy_flags = [
            f for f in result.flags
            if "action_entropy" in f.metric and not np.isnan(f.observed.value)
        ]
        assert entropy_flags
        assert entropy_flags[0].level == RiskLevel.OK

    def test_collapsed_batch_flags_pca(self):
        """Rank-1 action data should trigger PCA concentration flag."""
        rng = np.random.default_rng(42)
        episodes = []
        for i in range(10):
            n = 200
            t = np.arange(n, dtype=np.float64) * 0.1
            v = rng.random(n)
            # All 4 dims are linear combinations of one direction
            acts = np.column_stack([v, 2*v, -v, 0.5*v]).astype(np.float32)
            episodes.append(Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=t,
                observations={"proprio": rng.random((n, 4)).astype(np.float32)},
                actions=acts,
            ))
        batch = _batch_of(episodes)
        result = CoverageEntropyAnalyzer().analyze(batch)
        pca_flags = [f for f in result.flags if "pca" in f.metric]
        assert pca_flags
        assert pca_flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)


class TestCoverageEntropyAnalyzerBimodal:
    def test_bimodal_episodes_flagged_as_info(self):
        short_eps = [_make_episode(30, ep_id=f"ep_{i}") for i in range(10)]
        long_eps  = [_make_episode(120, ep_id=f"ep_{i+10}") for i in range(10)]
        batch = _batch_of(short_eps + long_eps)
        result = CoverageEntropyAnalyzer().analyze(batch)
        ep_len_flags = [f for f in result.flags if "episode_length" in f.metric]
        assert ep_len_flags
        # Bimodal hint is INFO; high-CV warning is WARNING
        assert ep_len_flags[0].level in (RiskLevel.INFO, RiskLevel.WARNING)


class TestCoverageEntropyEdgeCases:
    def test_empty_batch(self):
        result = CoverageEntropyAnalyzer().analyze(_batch_of([]))
        assert result.flags == []

    def test_no_proprio_graceful(self):
        eps = [_make_episode(50, ep_id=f"ep_{i}", with_proprio=False) for i in range(3)]
        result = CoverageEntropyAnalyzer().analyze(_batch_of(eps))
        assert result.analyzer_name == "coverage_entropy"
        assert "state_entropy" in result.raw_metrics
        assert result.raw_metrics["state_entropy"].get("skipped") is not None

    def test_policy_hints_diffusion(self):
        rng = np.random.default_rng(0)
        episodes = []
        for i in range(8):
            n = 200
            t = np.arange(n, dtype=np.float64) * 0.1
            v = rng.random(n)
            acts = np.column_stack([v, 2*v, -v, 0.5*v]).astype(np.float32)
            episodes.append(Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=t, observations={}, actions=acts,
            ))
        batch = _batch_of(episodes)
        result = CoverageEntropyAnalyzer().analyze(batch, policy_family="diffusion")
        dp_hints = [h for h in result.hints if "Diffusion" in h.policy_family]
        assert dp_hints
