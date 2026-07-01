"""Tests for calibra.llm — SFT coreset selection and fingerprinting."""

from __future__ import annotations

import numpy as np

from calibra.llm.fingerprint import FingerprintResult, _repetition_rate, _template_ratio
from calibra.llm.select import SFTCoresetSelector


def _make_fingerprint(n: int, seed: int = 0) -> FingerprintResult:
    """Synthetic fingerprints — no real sentence-transformers call, no network access."""
    rng = np.random.default_rng(seed)
    coherence = rng.uniform(0.3, 0.9, size=n).astype(np.float32)
    repetition_rate = rng.uniform(0.0, 0.2, size=n).astype(np.float32)
    template_ratio = np.zeros(n, dtype=np.float32)
    response_length_words = rng.integers(20, 100, size=n).astype(np.int32)
    embeddings = rng.normal(size=(n, 8)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return FingerprintResult(
        coherence=coherence,
        repetition_rate=repetition_rate,
        template_ratio=template_ratio,
        response_length_words=response_length_words,
        embeddings=embeddings,
    )


# ── text-level helpers ────────────────────────────────────────────────────────


def test_repetition_rate_detects_repeated_bigrams():
    assert _repetition_rate("go go go go go go") > 0.5


def test_repetition_rate_zero_for_short_text():
    assert _repetition_rate("hi there") == 0.0


def test_template_ratio_detects_boilerplate_opener():
    assert _template_ratio("Sure, here's the answer to your question.") > 0.0


def test_template_ratio_zero_for_direct_answer():
    assert _template_ratio("The capital of France is Paris.") == 0.0


# ── quality filtering ─────────────────────────────────────────────────────────


def test_quality_filter_removes_low_coherence():
    fp = _make_fingerprint(20)
    fp.coherence[0] = 0.01  # below default min_coherence=0.10
    selector = SFTCoresetSelector(keep_fraction=1.0, quality_only=True)
    result = selector.select(fp)
    assert 0 in result.quality_fail_indices
    assert 0 not in result.keep_indices


def test_quality_filter_removes_high_repetition():
    fp = _make_fingerprint(20)
    fp.repetition_rate[0] = 0.9  # above default max_repetition_rate=0.40
    selector = SFTCoresetSelector(keep_fraction=1.0, quality_only=True)
    result = selector.select(fp)
    assert 0 in result.quality_fail_indices


def test_quality_filter_removes_short_responses():
    fp = _make_fingerprint(20)
    fp.response_length_words[0] = 1  # below default min_length_words=5
    selector = SFTCoresetSelector(keep_fraction=1.0, quality_only=True)
    result = selector.select(fp)
    assert 0 in result.quality_fail_indices


def test_quality_filter_removes_high_template_ratio():
    fp = _make_fingerprint(20)
    fp.template_ratio[0] = 0.9  # above default max_template_ratio=0.50
    selector = SFTCoresetSelector(keep_fraction=1.0, quality_only=True)
    result = selector.select(fp)
    assert 0 in result.quality_fail_indices


# ── diversity selection ────────────────────────────────────────────────────────


def test_select_returns_requested_fraction():
    fp = _make_fingerprint(100)
    selector = SFTCoresetSelector(keep_fraction=0.3)
    result = selector.select(fp)
    assert result.n_kept == round(100 * 0.3)


def test_select_approximate_matches_exact_size():
    fp = _make_fingerprint(200, seed=1)
    exact = SFTCoresetSelector(keep_fraction=0.2, use_approximate=False).select(fp)
    approx = SFTCoresetSelector(keep_fraction=0.2, use_approximate=True, batch_size=50).select(fp)
    assert exact.n_kept == approx.n_kept


def test_select_all_failing_quality_returns_empty_coreset():
    fp = _make_fingerprint(10)
    fp.coherence[:] = 0.0  # every example fails quality
    selector = SFTCoresetSelector(keep_fraction=0.5)
    result = selector.select(fp)
    assert result.n_kept == 0
    assert result.keep_indices == []


# ── aggregate fingerprint ──────────────────────────────────────────────────────


def test_aggregate_fingerprint_keys_match_outcome_db_schema():
    from calibra.outcome_db import _DOMAIN_SCHEMAS

    fp = _make_fingerprint(50)
    result = SFTCoresetSelector(keep_fraction=0.5).select(fp)
    assert set(result.aggregate_fingerprint.keys()) == set(_DOMAIN_SCHEMAS["llm_sft"]["keys"])


def test_aggregate_fingerprint_empty_coreset_is_zeroed():
    fp = _make_fingerprint(10)
    fp.coherence[:] = 0.0
    result = SFTCoresetSelector(keep_fraction=0.5).select(fp)
    assert all(v == 0.0 for v in result.aggregate_fingerprint.values())


# ── serialisation ──────────────────────────────────────────────────────────────


def test_to_dict_is_json_serialisable():
    import json

    fp = _make_fingerprint(30)
    result = SFTCoresetSelector(keep_fraction=0.5).select(fp)
    json.dumps(result.to_dict())


def test_summary_contains_coreset_size():
    fp = _make_fingerprint(30)
    result = SFTCoresetSelector(keep_fraction=0.5).select(fp)
    assert str(result.n_kept) in result.summary()
