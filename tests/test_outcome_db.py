"""Tests for calibra.outcome_db — empirical outcome database."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from calibra.outcome_db import OutcomeDatabase, _normalize, _FINGERPRINT_KEYS


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_fp(**kwargs) -> dict:
    base = {
        "ldlj": -8.0,
        "spike_rate": 0.01,
        "vel_disc_rate": 0.01,
        "dropout_rate": 0.0,
        "jitter_cv": 0.001,
        "action_entropy": 3.5,
        "contact_phase_fraction": 0.15,
    }
    base.update(kwargs)
    return base


def _db(tmp_path: Path) -> OutcomeDatabase:
    return OutcomeDatabase(path=tmp_path / "outcomes.jsonl")


# ── normalization ─────────────────────────────────────────────────────────────

def test_normalize_returns_array_of_correct_length():
    fp = _make_fp()
    vec = _normalize(fp)
    assert vec.shape == (len(_FINGERPRINT_KEYS),)


def test_normalize_clips_to_unit_interval():
    fp = _make_fp(spike_rate=999.0, ldlj=-9999.0)
    vec = _normalize(fp)
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)


def test_normalize_missing_key_uses_midrange():
    fp = {}  # all keys missing
    vec = _normalize(fp)
    assert np.all(vec == 0.5)


# ── record ────────────────────────────────────────────────────────────────────

def test_record_persists_to_jsonl(tmp_path):
    db = _db(tmp_path)
    fp = _make_fp()
    rec = db.record(fp, predicted_score=72.0, actual_success_rate=0.78,
                    policy_family="act", dataset_name="test_ds")
    assert rec.record_id
    assert rec.actual_success_rate == 0.78

    # Reload from disk
    db2 = _db(tmp_path)
    assert len(db2._records) == 1
    assert db2._records[0].actual_success_rate == 0.78
    assert db2._records[0].dataset_name == "test_ds"


def test_record_appends_multiple(tmp_path):
    db = _db(tmp_path)
    for i in range(5):
        db.record(_make_fp(spike_rate=i * 0.01), predicted_score=80.0,
                  actual_success_rate=0.8)
    assert len(db._records) == 5
    db2 = _db(tmp_path)
    assert len(db2._records) == 5


def test_record_ids_are_unique(tmp_path):
    db = _db(tmp_path)
    recs = [db.record(_make_fp(), predicted_score=80.0, actual_success_rate=0.8)
            for _ in range(10)]
    ids = [r.record_id for r in recs]
    assert len(set(ids)) == 10


# ── find_similar ──────────────────────────────────────────────────────────────

def test_find_similar_returns_empty_when_no_records(tmp_path):
    db = _db(tmp_path)
    result = db.find_similar(_make_fp())
    assert result == []


def test_find_similar_returns_closest_first(tmp_path):
    db = _db(tmp_path)
    # Near-identical dataset
    db.record(_make_fp(spike_rate=0.02), predicted_score=70.0,
              actual_success_rate=0.7, dataset_name="close")
    # Very different dataset
    db.record(_make_fp(spike_rate=0.18, ldlj=-28.0, vel_disc_rate=0.35),
              predicted_score=30.0, actual_success_rate=0.3, dataset_name="far")

    similar = db.find_similar(_make_fp(spike_rate=0.025))
    assert len(similar) >= 1
    assert similar[0][0].dataset_name == "close"


def test_find_similar_policy_family_soft_preferred(tmp_path):
    db = _db(tmp_path)
    db.record(_make_fp(), predicted_score=70.0, actual_success_rate=0.7,
              policy_family="act", dataset_name="act_match")
    db.record(_make_fp(), predicted_score=70.0, actual_success_rate=0.7,
              policy_family="diffusion", dataset_name="diff_mismatch")

    similar = db.find_similar(_make_fp(), policy_family="act")
    # act_match should come first because distance is multiplied by 0.7
    assert similar[0][0].dataset_name == "act_match"


def test_find_similar_max_distance_filters(tmp_path):
    db = _db(tmp_path)
    db.record(_make_fp(spike_rate=0.19, ldlj=-28.0, vel_disc_rate=0.38),
              predicted_score=10.0, actual_success_rate=0.1)
    similar = db.find_similar(_make_fp(), max_distance=0.05)
    assert similar == []


def test_find_similar_returns_at_most_k(tmp_path):
    db = _db(tmp_path)
    for i in range(10):
        db.record(_make_fp(spike_rate=i * 0.005), predicted_score=70.0,
                  actual_success_rate=0.7)
    similar = db.find_similar(_make_fp(), k=3)
    assert len(similar) <= 3


# ── blend_prediction ──────────────────────────────────────────────────────────

def test_blend_prediction_no_similar_returns_heuristic(tmp_path):
    db = _db(tmp_path)
    blended, ew = db.blend_prediction(72.0, [])
    assert blended == 72.0
    assert ew == 0.0


def test_blend_prediction_with_similar_moves_toward_empirical(tmp_path):
    db = _db(tmp_path)
    fp = _make_fp()
    db.record(fp, predicted_score=72.0, actual_success_rate=0.90,
              dataset_name="ds1")
    similar = db.find_similar(fp)
    blended, ew = db.blend_prediction(72.0, similar)
    # actual=90%, heuristic=72% → blended should be > 72
    assert blended > 72.0
    assert 0.0 < ew <= 1.0


def test_blend_prediction_stays_in_0_100(tmp_path):
    db = _db(tmp_path)
    fp = _make_fp()
    db.record(fp, predicted_score=5.0, actual_success_rate=0.0)
    similar = db.find_similar(fp)
    blended, _ = db.blend_prediction(5.0, similar)
    assert 0.0 <= blended <= 100.0


def test_blend_prediction_more_matches_higher_weight(tmp_path):
    db = _db(tmp_path)
    fp = _make_fp()
    for _ in range(5):
        db.record(fp, predicted_score=72.0, actual_success_rate=0.95)
    similar = db.find_similar(fp, k=5)
    _, ew_many = db.blend_prediction(72.0, similar)

    db2 = _db(tmp_path)
    similar_one = [similar[0]]
    _, ew_one = db2.blend_prediction(72.0, similar_one)

    assert ew_many >= ew_one


# ── calibrate_weights ─────────────────────────────────────────────────────────

def test_calibrate_returns_none_with_fewer_than_10_records(tmp_path):
    db = _db(tmp_path)
    for _ in range(5):
        db.record(_make_fp(), predicted_score=70.0, actual_success_rate=0.7)
    assert db.calibrate_weights() is None


def test_calibrate_returns_dict_with_10_or_more_records(tmp_path):
    db = _db(tmp_path)
    rng = np.random.default_rng(42)
    for i in range(12):
        spike = rng.uniform(0.01, 0.10)
        db.record(
            _make_fp(spike_rate=spike),
            predicted_score=80.0 - spike * 100,
            actual_success_rate=max(0.0, 0.9 - spike),
        )
    result = db.calibrate_weights()
    assert isinstance(result, dict)
    for key in _FINGERPRINT_KEYS:
        assert key in result
        assert result[key] >= 0.0  # non-negative weights only


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_empty_db(tmp_path):
    db = _db(tmp_path)
    s = db.summary()
    assert "0 records" in s


def test_summary_nonempty_db(tmp_path):
    db = _db(tmp_path)
    db.record(_make_fp(), predicted_score=72.0, actual_success_rate=0.78)
    s = db.summary()
    assert "1 record" in s
    assert "Mean absolute error" in s


# ── list_records ──────────────────────────────────────────────────────────────

def test_list_records_serialisable(tmp_path):
    db = _db(tmp_path)
    db.record(_make_fp(), predicted_score=72.0, actual_success_rate=0.78)
    records = db.list_records()
    assert len(records) == 1
    # Must be JSON-serialisable
    json.dumps(records)


# ── corrupt / partial data resilience ────────────────────────────────────────

def test_corrupt_jsonl_line_is_skipped(tmp_path):
    p = tmp_path / "outcomes.jsonl"
    p.write_text('{"record_id":"a","timestamp":0,"fingerprint":{},'
                 '"predicted_score":70,"actual_success_rate":0.7,'
                 '"policy_family":"act","n_episodes":50,'
                 '"dataset_name":"ds","notes":""}\n'
                 'NOT VALID JSON\n')
    db = OutcomeDatabase(path=p)
    assert len(db._records) == 1  # corrupt line skipped silently
