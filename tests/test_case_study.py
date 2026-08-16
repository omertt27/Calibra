"""Tests for calibra.case_study — partner-facing case-study report generator."""

from __future__ import annotations

from pathlib import Path

from calibra.case_study import generate_case_study
from calibra.experiment_log import ExperimentLog


def _log(tmp_path: Path) -> ExperimentLog:
    return ExperimentLog(path=tmp_path / "experiments.jsonl")


def test_no_records_returns_placeholder(tmp_path):
    log = _log(tmp_path)
    text = generate_case_study(log, "nonexistent")
    assert "No records" in text


def test_empty_protocol_is_stamped_draft_with_all_gaps(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="calibra", retention_pct=25.0, eval_success_rate=0.84)
    text = generate_case_study(log, "e1")
    assert "DRAFT" in text
    assert "Open items" in text
    # headline withheld: no measured random counterpart at 25%
    assert "withheld" in text


def test_complete_protocol_is_validated_with_headline(tmp_path):
    log = _log(tmp_path)
    log.record(
        experiment_id="e1",
        condition="full",
        retention_pct=100.0,
        n_episodes=1000,
        gpu_hours=24.0,
        eval_success_rate=0.90,
        dataset_name="partner-a/pusht_v3",
        partner="Partner A",
    )
    for level, n in ((10.0, 100), (25.0, 250), (50.0, 500), (75.0, 750)):
        log.record(
            experiment_id="e1",
            condition="random",
            retention_pct=level,
            n_episodes=n,
            gpu_hours=24.0 * level / 100.0,
            eval_success_rate=0.60,
        )
        log.record(
            experiment_id="e1",
            condition="calibra",
            retention_pct=level,
            n_episodes=n,
            gpu_hours=24.0 * level / 100.0,
            eval_success_rate=0.84,
        )

    text = generate_case_study(log, "e1", gpu_cost_per_hour=2.0)

    assert "VALIDATED" in text
    assert "DRAFT" not in text
    assert "Open items" not in text
    assert "Partner A" in text
    # headline should pick the most aggressive complete level: 10%
    assert "At **10% retention**" in text
    assert "84.0%" in text  # calibra success at 10%
    assert "60.0%" in text  # random success at 10%
    # cost estimate present (gpu_hours * $2.00)
    assert "$" in text


def test_partner_label_overrides_recorded_partner(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="full", retention_pct=100.0, partner="Recorded Co")
    text = generate_case_study(log, "e1", partner_label="Override Co")
    assert "Override Co" in text
    assert "Recorded Co" not in text


def test_falls_back_to_experiment_id_when_no_partner(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="full", retention_pct=100.0)
    text = generate_case_study(log, "e1")
    assert "# Calibra Case Study — e1" in text


def test_gaps_list_missing_and_incomplete_slots(tmp_path):
    log = _log(tmp_path)
    log.record(experiment_id="e1", condition="full", retention_pct=100.0)  # no gpu_hours/success
    text = generate_case_study(log, "e1")
    assert "100% / full — recorded but missing" in text
    assert "25% / random — not recorded" in text


def test_never_mixes_in_benchmark_simulated_data(tmp_path):
    # Sanity: case_study module has no import of calibra.benchmark / predict_outcome.
    import calibra.case_study as mod

    src = Path(mod.__file__).read_text()
    assert "predict_outcome" not in src
    assert "CoresetSelector" not in src
