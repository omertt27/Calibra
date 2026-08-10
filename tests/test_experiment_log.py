"""Tests for calibra.experiment_log — design-partner experiment log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibra.experiment_log import ExperimentLog, ExperimentRecord


def _log(tmp_path: Path) -> ExperimentLog:
    return ExperimentLog(path=tmp_path / "experiments.jsonl")


# ── record validation ───────────────────────────────────────────────────────


def test_record_rejects_bad_condition(tmp_path):
    log = _log(tmp_path)
    with pytest.raises(ValueError):
        log.record(experiment_id="e1", condition="bogus", retention_pct=25.0)


def test_record_rejects_out_of_range_retention(tmp_path):
    log = _log(tmp_path)
    with pytest.raises(ValueError):
        log.record(experiment_id="e1", condition="calibra", retention_pct=150.0)


def test_record_rejects_out_of_range_success_rate(tmp_path):
    log = _log(tmp_path)
    with pytest.raises(ValueError):
        log.record(
            experiment_id="e1", condition="calibra", retention_pct=25.0, eval_success_rate=1.5
        )


# ── persistence ──────────────────────────────────────────────────────────────


def test_record_persists_to_jsonl(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, n_episodes=100)
    assert log.path.exists()
    lines = log.path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["experiment_id"] == "e1"
    assert payload["condition"] == "calibra"


def test_reload_from_disk(tmp_path):
    path = tmp_path / "experiments.jsonl"
    log1 = ExperimentLog(path=path)
    log1.record(experiment_id="e1", condition="full", retention_pct=100.0, n_episodes=1000)

    log2 = ExperimentLog(path=path)
    assert len(log2.list_records()) == 1
    assert log2.list_records()[0].experiment_id == "e1"


def test_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "experiments.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json\n" + json.dumps({"bad": "record"}) + "\n")
    log = ExperimentLog(path=path)
    assert log.list_records() == []


# ── grouping / retention table ──────────────────────────────────────────────


def test_experiments_lists_unique_ids_in_order(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="full", retention_pct=100.0)
    log.record(experiment_id="e2", condition="full", retention_pct=100.0)
    log.record(experiment_id="e1", condition="random", retention_pct=25.0)
    assert log.experiments() == ["e1", "e2"]


def test_retention_table_groups_by_level_and_condition(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="full", retention_pct=100.0)
    log.record(experiment_id="e1", condition="random", retention_pct=25.0)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0)

    table = log.retention_table("e1")
    assert set(table.keys()) == {100.0, 25.0}
    assert set(table[25.0].keys()) == {"random", "calibra"}


def test_retention_table_rerecording_supersedes(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, eval_success_rate=0.5)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, eval_success_rate=0.9)
    table = log.retention_table("e1")
    assert table[25.0]["calibra"].eval_success_rate == 0.9


# ── protocol completeness ───────────────────────────────────────────────────


def test_missing_conditions_full_protocol_absent(tmp_path):
    log = _log(tmp_path)
    missing = log.missing_conditions("e1")
    # 1 "full" slot + 4 non-100% levels * 2 conditions (random, calibra)
    assert len(missing) == 1 + 4 * 2
    assert (100.0, "full") in missing
    assert (25.0, "random") in missing
    assert (25.0, "calibra") in missing


def test_missing_conditions_full_protocol_complete(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="full", retention_pct=100.0)
    for level in (10.0, 25.0, 50.0, 75.0):
        log.record(experiment_id="e1", condition="random", retention_pct=level)
        log.record(experiment_id="e1", condition="calibra", retention_pct=level)
    assert log.missing_conditions("e1") == []


# ── calibra vs random ────────────────────────────────────────────────────────


def test_calibra_vs_random_delta(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="random", retention_pct=25.0, eval_success_rate=0.70)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, eval_success_rate=0.84)
    deltas = log.calibra_vs_random("e1")
    assert deltas[25.0] == pytest.approx(0.14)


def test_calibra_vs_random_none_when_missing(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, eval_success_rate=0.84)
    deltas = log.calibra_vs_random("e1")
    assert deltas[25.0] is None


def test_calibra_vs_random_excludes_100_pct(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="full", retention_pct=100.0, eval_success_rate=0.9)
    deltas = log.calibra_vs_random("e1")
    assert 100.0 not in deltas


# ── report rendering ─────────────────────────────────────────────────────────


def test_report_empty_experiment(tmp_path):
    log = _log(tmp_path)
    text = log.report("nonexistent")
    assert "No records" in text


def test_report_includes_delta_and_missing_summary(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="random", retention_pct=25.0, eval_success_rate=0.70)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, eval_success_rate=0.84)
    text = log.report("e1")
    assert "beats random" in text
    assert "Protocol incomplete" in text


def test_record_to_dict_round_trip():
    rec = ExperimentRecord(
        record_id="abc123",
        timestamp=1.0,
        experiment_id="e1",
        condition="calibra",
        retention_pct=25.0,
        eval_success_rate=0.84,
    )
    restored = ExperimentRecord.from_dict(rec.to_dict())
    assert restored.to_dict() == rec.to_dict()
