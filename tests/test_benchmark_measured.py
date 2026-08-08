"""Tests for calibra.benchmark's --experiment-id measured-value substitution."""

from __future__ import annotations

import json

import numpy as np
import pytest

import calibra.experiment_log as experiment_log_module
from calibra.benchmark import _case_study_status, _lookup_measured, run_benchmark
from calibra.experiment_log import ExperimentLog
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata


@pytest.fixture
def mock_batch():
    episodes = [
        Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
            timestamps=np.linspace(0, 2.0, 20),
            observations={"proprio": np.random.randn(20, 3)},
            actions=np.random.randn(20, 2),
        )
        for i in range(10)
    ]
    return EpisodeBatch(
        episodes=episodes, dataset_name="mock_dataset", format="lerobot", source_path="mock_path"
    )


@pytest.fixture
def elog(tmp_path, monkeypatch):
    path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr(experiment_log_module, "_DEFAULT_DB_PATH", path)
    return ExperimentLog(path=path)


# ── _lookup_measured ─────────────────────────────────────────────────────────


def test_lookup_measured_empty_table():
    assert _lookup_measured({}, "calibra", 25.0) is None


def test_lookup_measured_within_tolerance(tmp_path):
    log = ExperimentLog(path=tmp_path / "e.jsonl")
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, gpu_hours=6.0)
    table = log.retention_table("e1")
    rec = _lookup_measured(table, "calibra", 25.4)  # within 1.0 tolerance
    assert rec is not None
    assert rec.gpu_hours == 6.0


def test_lookup_measured_outside_tolerance(tmp_path):
    log = ExperimentLog(path=tmp_path / "e.jsonl")
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, gpu_hours=6.0)
    table = log.retention_table("e1")
    assert _lookup_measured(table, "calibra", 40.0) is None


# ── CLI integration ──────────────────────────────────────────────────────────


def test_benchmark_uses_simulated_by_default(mock_batch, monkeypatch, capsys):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    run_benchmark(["mock_path", "--keep", "0.3", "--policy", "diffusion", "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["any_measured"] is False
    assert summary["status"] == "SIMULATED"
    assert summary["results"]["calibra"]["gpu_hours_source"] == "simulated"


def test_benchmark_substitutes_measured_values(mock_batch, monkeypatch, capsys, elog):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)

    elog.record(
        experiment_id="partner-a",
        condition="full",
        retention_pct=100.0,
        gpu_hours=24.0,
        eval_success_rate=0.90,
    )
    elog.record(
        experiment_id="partner-a",
        condition="calibra",
        retention_pct=30.0,
        gpu_hours=8.4,
        eval_success_rate=0.88,
    )
    # random condition intentionally left unmeasured — should stay simulated

    run_benchmark(
        [
            "mock_path",
            "--keep",
            "0.3",
            "--policy",
            "diffusion",
            "--experiment-id",
            "partner-a",
            "--json",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert summary["any_measured"] is True
    # random is still simulated, so this must NOT be presented as a validated case study
    assert summary["status"] == "PARTIAL MEASUREMENT"
    assert summary["results"]["raw"]["gpu_hours"] == 24.0
    assert summary["results"]["raw"]["gpu_hours_source"] == "measured"
    assert summary["results"]["raw"]["predicted_success_rate"] == 90.0
    assert summary["results"]["calibra"]["gpu_hours"] == 8.4
    assert summary["results"]["calibra"]["gpu_hours_source"] == "measured"
    assert summary["results"]["calibra"]["predicted_success_rate"] == 88.0
    # random wasn't recorded — stays simulated
    assert summary["results"]["random"]["gpu_hours_source"] == "simulated"

    # compute savings now reflects measured GPU-hours (8.4 / 24.0), not episode count
    assert summary["compute_savings_pct"] == pytest.approx(100.0 * (1 - 8.4 / 24.0), abs=0.1)


def test_benchmark_report_flags_partial_measurement(mock_batch, monkeypatch, capsys, elog):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    elog.record(
        experiment_id="partner-a", condition="calibra", retention_pct=30.0, gpu_hours=8.4
    )

    run_benchmark(
        ["mock_path", "--keep", "0.3", "--experiment-id", "partner-a"]
    )
    out = capsys.readouterr().out
    assert "(measured)" in out
    assert "(simulated)" in out
    assert "still simulated" in out
    assert "STATUS: PARTIAL MEASUREMENT" in out
    assert "VALIDATED" not in out


def test_benchmark_fully_measured_is_validated(mock_batch, monkeypatch, capsys, elog):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    elog.record(
        experiment_id="partner-a",
        condition="full",
        retention_pct=100.0,
        gpu_hours=24.0,
        eval_success_rate=0.90,
    )
    elog.record(
        experiment_id="partner-a",
        condition="random",
        retention_pct=30.0,
        gpu_hours=9.0,
        eval_success_rate=0.82,
    )
    elog.record(
        experiment_id="partner-a",
        condition="calibra",
        retention_pct=30.0,
        gpu_hours=8.4,
        eval_success_rate=0.88,
    )

    run_benchmark(
        ["mock_path", "--keep", "0.3", "--experiment-id", "partner-a", "--json"]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "CASE STUDY / VALIDATED"

    run_benchmark(["mock_path", "--keep", "0.3", "--experiment-id", "partner-a"])
    out = capsys.readouterr().out
    assert "STATUS: CASE STUDY / VALIDATED" in out
    assert "PARTIAL" not in out


# ── _case_study_status ───────────────────────────────────────────────────────


def _cond(gpu_source="simulated", success_source="simulated"):
    return {
        "n_episodes": 10,
        "gpu_hours": 1.0,
        "gpu_hours_source": gpu_source,
        "predicted_success_rate": 90.0,
        "success_rate_source": success_source,
    }


def test_status_all_simulated():
    assert _case_study_status([_cond(), _cond(), _cond()]) == "SIMULATED"


def test_status_all_fully_measured():
    conds = [_cond("measured", "measured") for _ in range(3)]
    assert _case_study_status(conds) == "CASE STUDY / VALIDATED"


def test_status_one_field_unmeasured_is_partial_not_validated():
    # Calibra's gpu_hours measured but success rate still simulated — must not validate.
    conds = [
        _cond("measured", "measured"),
        _cond("measured", "measured"),
        _cond("measured", "simulated"),
    ]
    assert _case_study_status(conds) == "PARTIAL MEASUREMENT"


def test_status_no_measurement_anywhere_is_not_partial():
    assert _case_study_status([_cond(), _cond()]) == "SIMULATED"


# ── --sweep ──────────────────────────────────────────────────────────────────


def test_sweep_default_fractions_all_simulated(mock_batch, monkeypatch, capsys):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    run_benchmark(["mock_path", "--sweep", "--policy", "diffusion", "--json"])
    summary = json.loads(capsys.readouterr().out)

    assert summary["fractions"] == [0.1, 0.25, 0.5, 0.75, 1.0]
    assert summary["overall_status"] == "SIMULATED"
    # 4 sub-100% levels, each with random + calibra
    assert len(summary["rows"]) == 4
    retentions = sorted(row["retention_pct"] for row in summary["rows"])
    assert retentions == pytest.approx([10.0, 25.0, 50.0, 75.0])
    for row in summary["rows"]:
        assert row["random"]["gpu_hours_source"] == "simulated"
        assert row["calibra"]["gpu_hours_source"] == "simulated"


def test_sweep_custom_fractions(mock_batch, monkeypatch, capsys):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    run_benchmark(["mock_path", "--sweep", "--fractions", "0.2,0.6,1.0", "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["fractions"] == [0.2, 0.6, 1.0]
    assert sorted(row["retention_pct"] for row in summary["rows"]) == [20.0, 60.0]


def test_sweep_rejects_out_of_range_fraction(mock_batch, monkeypatch):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    with pytest.raises(SystemExit):
        run_benchmark(["mock_path", "--sweep", "--fractions", "0.2,1.5"])


def test_sweep_fully_measured_is_validated(mock_batch, monkeypatch, capsys, elog):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    elog.record(
        experiment_id="partner-a", condition="full", retention_pct=100.0,
        gpu_hours=24.0, eval_success_rate=0.90,
    )
    for pct, r_gpu, r_succ, c_gpu, c_succ in [
        (10.0, 3.1, 0.71, 2.8, 0.74),
        (25.0, 6.5, 0.79, 6.0, 0.83),
        (50.0, 12.4, 0.86, 12.0, 0.87),
        (75.0, 18.0, 0.89, 18.0, 0.89),
    ]:
        elog.record(
            experiment_id="partner-a", condition="random", retention_pct=pct,
            gpu_hours=r_gpu, eval_success_rate=r_succ,
        )
        elog.record(
            experiment_id="partner-a", condition="calibra", retention_pct=pct,
            gpu_hours=c_gpu, eval_success_rate=c_succ,
        )

    run_benchmark(["mock_path", "--sweep", "--experiment-id", "partner-a", "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["overall_status"] == "CASE STUDY / VALIDATED"
    for row in summary["rows"]:
        assert row["status"] == "CASE STUDY / VALIDATED"
        assert row["random"]["gpu_hours_source"] == "measured"
        assert row["calibra"]["gpu_hours_source"] == "measured"


def test_sweep_partial_measurement_cannot_be_validated(mock_batch, monkeypatch, capsys, elog):
    import calibra.ingestion.registry as registry

    monkeypatch.setattr(registry, "load", lambda path, reader=None: mock_batch)
    elog.record(
        experiment_id="partner-a", condition="full", retention_pct=100.0,
        gpu_hours=24.0, eval_success_rate=0.90,
    )
    # Only the 25% level is fully measured; everything else stays simulated.
    elog.record(
        experiment_id="partner-a", condition="random", retention_pct=25.0,
        gpu_hours=6.5, eval_success_rate=0.79,
    )
    elog.record(
        experiment_id="partner-a", condition="calibra", retention_pct=25.0,
        gpu_hours=6.0, eval_success_rate=0.83,
    )

    run_benchmark(["mock_path", "--sweep", "--experiment-id", "partner-a", "--json"])
    summary = json.loads(capsys.readouterr().out)

    assert summary["overall_status"] == "PARTIAL MEASUREMENT"
    by_pct = {row["retention_pct"]: row for row in summary["rows"]}
    assert by_pct[25.0]["status"] == "CASE STUDY / VALIDATED"
    assert by_pct[10.0]["status"] == "SIMULATED"

    run_benchmark(["mock_path", "--sweep", "--experiment-id", "partner-a"])
    out = capsys.readouterr().out
    assert "OVERALL STATUS: PARTIAL MEASUREMENT" in out
    assert "Do not report this as a validated case study" in out
