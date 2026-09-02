"""Tests for `calibra experiment record` metrics/review ingestion (calibra.experiment)."""

from __future__ import annotations

import json

import pytest

from calibra.experiment import run_experiment


def _rows(path):
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def _base_argv(path, **over):
    argv = [
        "record",
        "--experiment-id",
        "e1",
        "--condition",
        "calibra",
        "--retention",
        "25",
        "--path",
        str(path),
    ]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


# ── --from-metrics ─────────────────────────────────────────────────────────


def test_record_pulls_values_from_metrics_file(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    metrics = tmp_path / "summary.json"
    metrics.write_text(
        json.dumps({"gpu_hours": 19.8, "_runtime": 71280, "success_rate": 0.84, "loss": 0.02})
    )

    run_experiment(_base_argv(log_path) + ["--from-metrics", str(metrics)])

    (row,) = _rows(log_path)
    assert row["gpu_hours"] == 19.8
    assert row["wall_clock_seconds"] == 71280
    assert row["eval_success_rate"] == 0.84
    assert row["training_loss"] == 0.02
    assert row["metrics_source"].startswith("json:")


def test_explicit_flag_overrides_metrics_file(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    metrics = tmp_path / "summary.json"
    metrics.write_text(json.dumps({"gpu_hours": 19.8, "success_rate": 0.84}))

    run_experiment(_base_argv(log_path, gpu_hours=25.0) + ["--from-metrics", str(metrics)])

    (row,) = _rows(log_path)
    assert row["gpu_hours"] == 25.0  # flag wins
    assert row["eval_success_rate"] == 0.84  # metrics still used where no flag


def test_map_flag_reaches_custom_path(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    metrics = tmp_path / "summary.json"
    metrics.write_text(json.dumps({"results": {"eval": {"sr": 0.71}}}))

    run_experiment(
        _base_argv(log_path)
        + ["--from-metrics", str(metrics), "--map", "eval_success_rate=results.eval.sr"]
    )

    (row,) = _rows(log_path)
    assert row["eval_success_rate"] == 0.71


def test_bad_metrics_path_exits(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    with pytest.raises(SystemExit):
        run_experiment(_base_argv(log_path) + ["--from-metrics", str(tmp_path / "nope.json")])
    assert not log_path.exists()


# ── --dry-run ──────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(tmp_path, capsys):
    log_path = tmp_path / "experiments.jsonl"
    metrics = tmp_path / "summary.json"
    metrics.write_text(json.dumps({"gpu_hours": 5.0}))

    run_experiment(_base_argv(log_path) + ["--from-metrics", str(metrics), "--dry-run"])

    assert not log_path.exists()
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "gpu_hours" in out and "5" in out


# ── --from-review rollup ───────────────────────────────────────────────────


def _review_file(tmp_path, n_episodes, rows):
    f = tmp_path / "review.json"
    f.write_text(json.dumps({"n_episodes": n_episodes, "assessments": rows}))
    return f


def test_from_review_rolls_up_mean_fields(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    review = _review_file(
        tmp_path,
        2,
        [
            {"episode_id": "a", "anomaly_score": 0.2, "quality_risk": 0.4, "coverage_value": 0.8},
            {"episode_id": "b", "anomaly_score": 0.6, "quality_risk": 0.8, "coverage_value": 0.2},
        ],
    )

    run_experiment(_base_argv(log_path) + ["--from-review", str(review)])

    (row,) = _rows(log_path)
    assert row["mean_anomaly_score"] == pytest.approx(0.4)
    assert row["mean_quality_risk"] == pytest.approx(0.6)
    assert row["mean_coverage_value"] == pytest.approx(0.5)


def test_from_review_rejects_partial_coverage(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    review = _review_file(
        tmp_path,
        165,
        [{"episode_id": "a", "anomaly_score": 0.2, "quality_risk": 0.4}],
    )
    with pytest.raises(SystemExit):
        run_experiment(_base_argv(log_path) + ["--from-review", str(review)])
    assert not log_path.exists()


def test_from_review_and_explicit_mean_flag_conflict(tmp_path):
    log_path = tmp_path / "experiments.jsonl"
    review = _review_file(
        tmp_path, 1, [{"episode_id": "a", "anomaly_score": 0.2, "quality_risk": 0.4}]
    )
    with pytest.raises(SystemExit):
        run_experiment(
            _base_argv(log_path) + ["--from-review", str(review), "--mean-anomaly-score", "0.5"]
        )
