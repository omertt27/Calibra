"""Tests for calibra.metrics_ingest — reading measured metrics from a finished run."""

from __future__ import annotations

import json

import pytest

from calibra.metrics_ingest import MetricsBundle, load_metrics, parse_field_map


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


# ── flat JSON parsing ───────────────────────────────────────────────────────


def test_flat_json_recognized_keys(tmp_path):
    f = _write(
        tmp_path / "metrics.json",
        {"gpu_hours": 19.8, "wall_clock_seconds": 71280, "loss": 0.012, "success_rate": 0.84},
    )
    mb = load_metrics(f)
    assert mb.gpu_hours == 19.8
    assert mb.wall_clock_seconds == 71280.0
    assert mb.training_loss == 0.012
    assert mb.eval_success_rate == 0.84
    assert mb.energy_kwh is None
    assert mb.source.startswith("json:")


def test_alias_table_matches_alternate_names(tmp_path):
    f = _write(
        tmp_path / "summary.json", {"gpu_hrs": 4.0, "train_runtime": 3600, "final_loss": 1.5}
    )
    mb = load_metrics(f)
    assert mb.gpu_hours == 4.0
    assert mb.wall_clock_seconds == 3600.0
    assert mb.training_loss == 1.5
    assert mb.matched["gpu_hours"] == "gpu_hrs"


def test_nested_keys_are_flattened(tmp_path):
    f = _write(tmp_path / "m.json", {"eval": {"success_rate": 0.9}, "train": {"loss": 0.5}})
    mb = load_metrics(f)
    assert mb.eval_success_rate == 0.9
    assert mb.training_loss == 0.5


def test_missing_keys_are_none(tmp_path):
    f = _write(tmp_path / "m.json", {"unrelated": 1})
    mb = load_metrics(f)
    assert mb == MetricsBundle(source=mb.source, raw={"unrelated": 1})


def test_booleans_are_not_treated_as_metrics(tmp_path):
    f = _write(tmp_path / "m.json", {"success_rate": True})
    mb = load_metrics(f)
    assert mb.eval_success_rate is None


def test_non_object_json_rejected(tmp_path):
    f = tmp_path / "m.json"
    f.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_metrics(f)


# ── success-rate normalization ─────────────────────────────────────────────


def test_success_rate_percent_is_normalized(tmp_path):
    f = _write(tmp_path / "m.json", {"success_rate": 84.0})
    mb = load_metrics(f)
    assert mb.eval_success_rate == pytest.approx(0.84)
    assert "percent" in mb.matched["eval_success_rate"]


def test_success_rate_fraction_left_alone(tmp_path):
    f = _write(tmp_path / "m.json", {"success_rate": 0.84})
    mb = load_metrics(f)
    assert mb.eval_success_rate == 0.84
    assert "percent" not in mb.matched["eval_success_rate"]


# ── W&B offline summary ────────────────────────────────────────────────────


def test_wandb_summary_runtime_becomes_wall_clock(tmp_path):
    f = _write(
        tmp_path / "wandb-summary.json",
        {"_runtime": 71280.4, "_step": 50000, "eval/success_rate": 0.8, "train/loss": 0.01},
    )
    mb = load_metrics(f)
    assert mb.wall_clock_seconds == pytest.approx(71280.4)
    assert mb.eval_success_rate == 0.8
    assert mb.training_loss == 0.01
    assert mb.source.startswith("wandb:")


def test_wandb_summary_without_gpu_hours_leaves_it_none(tmp_path):
    # GPU-hours is never derived from wall-clock — it must be in the source.
    f = _write(tmp_path / "wandb-summary.json", {"_runtime": 3600})
    mb = load_metrics(f)
    assert mb.gpu_hours is None


def test_directory_autodetects_wandb_summary(tmp_path):
    run = tmp_path / "wandb" / "latest-run" / "files"
    run.mkdir(parents=True)
    _write(run / "wandb-summary.json", {"_runtime": 100, "success_rate": 0.5})
    mb = load_metrics(tmp_path / "wandb" / "latest-run" / "files")
    assert mb.wall_clock_seconds == 100.0
    assert mb.source.endswith("wandb-summary.json")


def test_directory_with_no_known_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="none of"):
        load_metrics(tmp_path)


def test_missing_path_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metrics(tmp_path / "nope.json")


# ── --map field overrides ──────────────────────────────────────────────────


def test_map_override_reaches_nested_path(tmp_path):
    f = _write(tmp_path / "m.json", {"results": {"eval": {"sr": 0.77}}})
    mb = load_metrics(f, extra_map={"eval_success_rate": "results.eval.sr"})
    assert mb.eval_success_rate == 0.77
    assert mb.matched["eval_success_rate"] == "results.eval.sr"


def test_map_override_beats_alias_match(tmp_path):
    f = _write(tmp_path / "m.json", {"success_rate": 0.10, "real": {"sr": 0.90}})
    mb = load_metrics(f, extra_map={"eval_success_rate": "real.sr"})
    assert mb.eval_success_rate == 0.90


def test_map_override_missing_path_errors(tmp_path):
    f = _write(tmp_path / "m.json", {"a": 1})
    with pytest.raises(ValueError, match="no numeric value"):
        load_metrics(f, extra_map={"gpu_hours": "b.c"})


def test_parse_field_map_valid():
    assert parse_field_map(["gpu_hours=a.b", "training_loss=x/y"]) == {
        "gpu_hours": "a.b",
        "training_loss": "x/y",
    }


def test_parse_field_map_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown field"):
        parse_field_map(["frobnicate=a.b"])


def test_parse_field_map_rejects_missing_equals():
    with pytest.raises(ValueError, match="FIELD=PATH"):
        parse_field_map(["gpu_hours"])


def test_parse_field_map_rejects_empty_path():
    with pytest.raises(ValueError, match="missing key path"):
        parse_field_map(["gpu_hours="])
