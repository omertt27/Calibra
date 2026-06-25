"""Tests for watch.py enhancements: remediation advice, stream mode, --remediate flag."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


from calibra.watch import (
    WatchSession,
    _remediation_advice,
    _stream_watch,
    _verdict_icon,
)


# ── _remediation_advice ───────────────────────────────────────────────────────


class TestRemediationAdvice:
    def test_jerk_spike_fail(self):
        msg = _remediation_advice("jerk_spike_rate", 0.08, "FAIL")
        assert "RE-RECORD" in msg
        assert "smoothly" in msg.lower() or "spike" in msg.lower()

    def test_jerk_spike_warn(self):
        msg = _remediation_advice("jerk_spike_rate", 0.03, "WARN")
        assert "RE-RECORD" not in msg

    def test_velocity_discontinuity(self):
        msg = _remediation_advice("velocity_discontinuity_rate", 0.07, "FAIL")
        assert "RE-RECORD" in msg
        assert (
            "continuous" in msg.lower() or "discontinu" in msg.lower() or "reversal" in msg.lower()
        )

    def test_ldlj(self):
        msg = _remediation_advice("ldlj", -18.0, "FAIL")
        assert "RE-RECORD" in msg
        assert "smooth" in msg.lower() or "slow" in msg.lower()

    def test_dropout(self):
        msg = _remediation_advice("timestamp_dropout_rate", 0.06, "FAIL")
        assert "RE-RECORD" in msg
        assert "connection" in msg.lower() or "dropout" in msg.lower() or "link" in msg.lower()

    def test_unknown_metric_returns_generic(self):
        msg = _remediation_advice("unknown_metric_xyz", 99.0, "FAIL")
        assert "RE-RECORD" in msg
        assert "unknown_metric_xyz" in msg

    def test_partial_metric_name_match(self):
        # "spike_rate" should match against "jerk_spike_rate"
        msg = _remediation_advice("spike_rate", 0.08, "FAIL")
        assert "RE-RECORD" in msg


# ── WatchSession.record with remediate ────────────────────────────────────────


def _make_mock_report(verdict: str):
    """Build a minimal mock DiagnosticReport."""
    from calibra.schema.report import RiskLevel, RiskFlag, ObservedValue

    mock_flag = MagicMock(spec=RiskFlag)
    mock_flag.metric = "jerk_spike_rate"
    mock_flag.observed = MagicMock(spec=ObservedValue)
    mock_flag.observed.value = 0.08
    mock_flag.threshold = 0.05

    mock_report = MagicMock()
    if verdict == "FAIL":
        mock_report.flags_at_level.side_effect = lambda lvl: (
            [mock_flag] if lvl == RiskLevel.CRITICAL else []
        )
    elif verdict == "WARN":
        mock_report.flags_at_level.side_effect = lambda lvl: (
            [mock_flag] if lvl == RiskLevel.WARNING else []
        )
    else:
        mock_report.flags_at_level.return_value = []
    return mock_report


class TestWatchSessionRecord:
    def test_pass_no_remediation(self, capsys, tmp_path):
        session = WatchSession(log_file=None, quiet=False, bell=False)
        report = _make_mock_report("PASS")
        session.record(Path("ep_001.h5"), "PASS", "all metrics OK", report=report, remediate=True)
        out = capsys.readouterr().out
        assert "↳" not in out

    def test_fail_remediation_printed(self, capsys, tmp_path):
        session = WatchSession(log_file=None, quiet=False, bell=False)
        report = _make_mock_report("FAIL")
        session.record(
            Path("ep_002.h5"), "FAIL", "jerk_spike_rate = 0.08", report=report, remediate=True
        )
        out = capsys.readouterr().out
        assert "↳" in out
        assert "RE-RECORD" in out

    def test_warn_remediation_printed(self, capsys):
        session = WatchSession(log_file=None, quiet=False, bell=False)
        report = _make_mock_report("WARN")
        session.record(
            Path("ep_003.h5"), "WARN", "jerk_spike_rate = 0.03", report=report, remediate=True
        )
        out = capsys.readouterr().out
        assert "↳" in out

    def test_remediation_stored_in_log_entry(self, tmp_path):
        log = tmp_path / "session.jsonl"
        session = WatchSession(log_file=log, quiet=True, bell=False)
        report = _make_mock_report("FAIL")
        session.record(
            Path("ep_004.h5"), "FAIL", "jerk_spike_rate = 0.08", report=report, remediate=True
        )
        lines = [json.loads(ln) for ln in log.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert "remediation" in lines[0]

    def test_no_remediation_flag_omits_advice(self, capsys):
        session = WatchSession(log_file=None, quiet=False, bell=False)
        report = _make_mock_report("FAIL")
        session.record(
            Path("ep_005.h5"), "FAIL", "jerk_spike_rate = 0.08", report=report, remediate=False
        )
        out = capsys.readouterr().out
        assert "↳" not in out

    def test_quiet_mode_suppresses_output(self, capsys):
        session = WatchSession(log_file=None, quiet=True, bell=False)
        report = _make_mock_report("FAIL")
        session.record(Path("ep_006.h5"), "FAIL", "spike", report=report, remediate=True)
        out = capsys.readouterr().out
        assert out == ""

    def test_counters_incremented_correctly(self):
        session = WatchSession(log_file=None, quiet=True, bell=False)
        session.record(Path("a.h5"), "PASS", "ok")
        session.record(Path("b.h5"), "WARN", "warn")
        session.record(Path("c.h5"), "FAIL", "fail")
        assert session.n_pass == 1
        assert session.n_warn == 1
        assert session.n_fail == 1
        assert session.total == 3


# ── _stream_watch ─────────────────────────────────────────────────────────────


def _run_stream(lines: list[str], remediate: bool = False) -> tuple[WatchSession, str]:
    """Helper: feed lines to _stream_watch via stdin mock, return session + stdout."""
    session = WatchSession(log_file=None, quiet=False, bell=False)
    stdin_text = "\n".join(lines) + "\n"

    captured_out = io.StringIO()
    with patch("sys.stdin", io.StringIO(stdin_text)), patch("sys.stdout", captured_out):
        _stream_watch(session, remediate=remediate)

    return session, captured_out.getvalue()


class TestStreamWatch:
    def test_pass_episode(self):
        session, out = _run_stream(
            [
                json.dumps(
                    {
                        "file": "ep_001.h5",
                        "spike_rate": 0.01,
                        "vel_disc_rate": 0.01,
                        "dropout_rate": 0.0,
                        "ldlj": -5.0,
                    }
                )
            ]
        )
        assert session.n_pass == 1
        assert session.n_fail == 0
        assert "PASS" in out

    def test_warn_episode_spike(self):
        session, out = _run_stream(
            [json.dumps({"file": "ep_002.h5", "spike_rate": 0.03, "ldlj": -5.0})]
        )
        assert session.n_warn == 1
        assert "WARN" in out

    def test_fail_episode_spike(self):
        session, out = _run_stream([json.dumps({"file": "ep_003.h5", "spike_rate": 0.08})])
        assert session.n_fail == 1
        assert "FAIL" in out

    def test_fail_episode_vel_disc(self):
        session, out = _run_stream([json.dumps({"file": "ep_004.h5", "vel_disc_rate": 0.07})])
        assert session.n_fail == 1

    def test_fail_episode_dropout(self):
        session, out = _run_stream([json.dumps({"file": "ep_005.h5", "dropout_rate": 0.06})])
        assert session.n_fail == 1

    def test_fail_episode_ldlj(self):
        session, out = _run_stream([json.dumps({"file": "ep_006.h5", "ldlj": -18.0})])
        assert session.n_fail == 1

    def test_multiple_episodes_tallied(self):
        lines = [json.dumps({"file": f"ep_{i:03d}.h5", "spike_rate": 0.01}) for i in range(3)] + [
            json.dumps({"file": "ep_bad.h5", "spike_rate": 0.09})
        ]
        session, _ = _run_stream(lines)
        assert session.total == 4
        assert session.n_pass == 3
        assert session.n_fail == 1

    def test_invalid_json_skipped(self):
        session, out = _run_stream(
            [
                "NOT JSON",
                json.dumps({"file": "ep_ok.h5", "spike_rate": 0.01}),
            ]
        )
        assert session.total == 1
        assert session.n_pass == 1

    def test_empty_line_skipped(self):
        session, _ = _run_stream(
            [
                "",
                "   ",
                json.dumps({"file": "ep_ok.h5", "spike_rate": 0.01}),
            ]
        )
        assert session.total == 1

    def test_remediate_prints_advice_on_fail(self):
        session, out = _run_stream(
            [json.dumps({"file": "ep_bad.h5", "spike_rate": 0.08})],
            remediate=True,
        )
        assert "↳" in out
        assert "RE-RECORD" in out

    def test_remediate_no_advice_on_pass(self):
        session, out = _run_stream(
            [json.dumps({"file": "ep_ok.h5", "spike_rate": 0.01})],
            remediate=True,
        )
        assert "↳" not in out

    def test_stream_log_file_written(self, tmp_path):
        log = tmp_path / "stream.jsonl"
        session = WatchSession(log_file=log, quiet=True, bell=False)
        lines_str = json.dumps({"file": "ep_001.h5", "spike_rate": 0.01}) + "\n"
        with patch("sys.stdin", io.StringIO(lines_str)):
            _stream_watch(session, remediate=False)
        entries = [json.loads(ln) for ln in log.read_text().strip().splitlines()]
        assert len(entries) == 1
        assert entries[0]["file"] == "ep_001.h5"

    def test_missing_metric_fields_treated_as_pass(self):
        # A JSON object with no metric fields → no threshold crossed → PASS
        session, out = _run_stream([json.dumps({"file": "ep_nometrics.h5"})])
        assert session.n_pass == 1

    def test_verdict_icon_mapping(self):
        assert "✅" in _verdict_icon("PASS")
        assert "❌" in _verdict_icon("FAIL")
        assert "⚠" in _verdict_icon("WARN")
