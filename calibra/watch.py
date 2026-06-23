"""
calibra watch — real-time dataset quality monitoring during teleoperation.

Watches a directory for new episode files and scores each one as it arrives,
giving operators immediate feedback during data collection sessions.

Features
---------
  • Detects new HDF5 / NPZ / MCAP files as they appear in the watched directory.
  • Scores each episode for jerk spikes, velocity discontinuities, temporal
    dropout, and LDLJ smoothness.
  • Prints a one-line verdict per episode: ✅ PASS / ⚠ WARN / ❌ FAIL.
  • Keeps a running session summary (pass rate, worst episodes).
  • Writes a machine-readable session log to --log-file (JSON Lines).
  • Rings the terminal bell on a FAIL episode (unless --no-bell).

Thresholds
-----------
  FAIL  — any CRITICAL flag (spike rate > 5%, vel_disc > 5%, dropout > 5%)
  WARN  — any WARNING flag  (spike rate > 2%, vel_disc > 2%, LDLJ < -10)
  PASS  — all metrics within acceptable range

Usage
------
    calibra watch /data/collection_session/
    calibra watch /data/collection_session/ --format hdf5 --log-file session.jsonl
    calibra watch /data/session/ --policy pi0 --no-bell --quiet

Exit codes
----------
    0  Session ended cleanly (Ctrl+C or --max-episodes reached)
    1  Error starting the watcher
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ── thresholds (mirrors ControlSmoothnessAnalyzer / TemporalAnalyzer) ─────────

_SPIKE_WARN     = 0.02
_SPIKE_FAIL     = 0.05
_VEL_DISC_WARN  = 0.02
_VEL_DISC_FAIL  = 0.05
_DROPOUT_WARN   = 0.01
_DROPOUT_FAIL   = 0.05
_LDLJ_WARN      = -10.0
_LDLJ_FAIL      = -15.0

# File extensions to watch for new episode files
_WATCH_EXTENSIONS = frozenset([".h5", ".hdf5", ".npz", ".mcap", ".bag"])


def _episode_verdict(report) -> tuple[str, str]:
    """
    Returns (verdict, details) for a single-episode DiagnosticReport.
    verdict is one of 'PASS', 'WARN', 'FAIL'.
    """
    from calibra.schema.report import RiskLevel
    if report.flags_at_level(RiskLevel.CRITICAL):
        top = report.flags_at_level(RiskLevel.CRITICAL)[0]
        return "FAIL", f"{top.metric} = {top.observed.value:.4g} (threshold {top.threshold})"
    if report.flags_at_level(RiskLevel.WARNING):
        top = report.flags_at_level(RiskLevel.WARNING)[0]
        return "WARN", f"{top.metric} = {top.observed.value:.4g}"
    return "PASS", "all metrics OK"


def _verdict_icon(verdict: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(verdict, "?")


def _remediation_advice(metric: str, value: float, verdict: str) -> str:
    """Return a one-sentence operator instruction for the top failure metric."""
    is_crit = verdict == "FAIL"
    prefix = "RE-RECORD: " if is_crit else "Consider: "
    advice = {
        "jerk_spike_rate": (
            f"{prefix}Move more smoothly — avoid abrupt stops and direction changes. "
            f"Spike rate {value:.1%} exceeds threshold."
        ),
        "spike_rate": (
            f"{prefix}Move more smoothly — avoid abrupt stops and direction changes. "
            f"Spike rate {value:.1%} exceeds threshold."
        ),
        "velocity_discontinuity_rate": (
            f"{prefix}Keep wrist/arm motion continuous. Avoid hesitation or sudden reversal. "
            f"Discontinuity rate {value:.1%} is too high."
        ),
        "vel_disc_rate": (
            f"{prefix}Keep wrist/arm motion continuous. Avoid hesitation or sudden reversal. "
            f"Discontinuity rate {value:.1%} is too high."
        ),
        "mean_ldlj": (
            f"{prefix}Slow down and smooth the trajectory — especially during approach and release. "
            f"LDLJ {value:.2f} indicates excessive jerk."
        ),
        "ldlj": (
            f"{prefix}Slow down and smooth the trajectory — especially during approach and release. "
            f"LDLJ {value:.2f} indicates excessive jerk."
        ),
        "timestamp_dropout_rate": (
            f"{prefix}Check robot connection — frame dropout detected ({value:.1%} of frames). "
            "Retry after verifying USB/Ethernet link."
        ),
        "dropout": (
            f"{prefix}Check robot connection — frame dropout detected ({value:.1%} of frames). "
            "Retry after verifying USB/Ethernet link."
        ),
        "jitter_cv": (
            f"{prefix}Control loop timing is irregular (CV={value:.4f}). "
            "Check for background processes throttling the control loop."
        ),
        "action_entropy": (
            f"{'Consider varying the demonstration — ' if not is_crit else 'RE-RECORD: '}"
            f"action entropy {value:.2f} bits/dim is very low. "
            "Try a different grasp or approach path."
        ),
    }
    metric_lower = metric.lower()
    for key, msg in advice.items():
        if key in metric_lower:
            return msg
    return (
        f"{prefix}Episode failed on '{metric}' = {value:.4g}. "
        "Review the motion and re-record if it felt incorrect."
    )


def _score_episode_file(path: Path, policy_family: Optional[str], reader=None):
    """Load a single-file episode and run the pipeline on it."""
    from calibra.ingestion.registry import load as _load
    from calibra.pipeline import Pipeline

    batch = _load(str(path), reader=reader)
    report = Pipeline().run(batch, policy_family=policy_family)
    return report


# ── session state ─────────────────────────────────────────────────────────────

class WatchSession:
    def __init__(self, log_file: Optional[Path], quiet: bool, bell: bool):
        self.log_file  = log_file
        self.quiet     = quiet
        self.bell      = bell
        self.n_pass    = 0
        self.n_warn    = 0
        self.n_fail    = 0
        self.episodes:  list[dict] = []
        self._start    = time.monotonic()

    @property
    def total(self) -> int:
        return self.n_pass + self.n_warn + self.n_fail

    def record(self, path: Path, verdict: str, details: str,
               report=None, remediate: bool = False) -> None:
        icon = _verdict_icon(verdict)
        elapsed = time.monotonic() - self._start

        entry: dict = {
            "t": round(elapsed, 2),
            "file": str(path.name),
            "verdict": verdict,
            "details": details,
        }

        remediation = ""
        if report is not None:
            from calibra.score import compute_score
            from calibra.schema.report import RiskLevel
            try:
                sc = compute_score(report)
                entry["calibra_score"] = sc["total_score"]
            except Exception:
                pass

            if remediate and verdict in ("FAIL", "WARN"):
                flags = (
                    report.flags_at_level(RiskLevel.CRITICAL)
                    or report.flags_at_level(RiskLevel.WARNING)
                )
                if flags:
                    top = flags[0]
                    remediation = _remediation_advice(
                        top.metric, float(top.observed.value), verdict
                    )
                    entry["remediation"] = remediation

        self.episodes.append(entry)

        if verdict == "PASS":
            self.n_pass += 1
        elif verdict == "WARN":
            self.n_warn += 1
        else:
            self.n_fail += 1

        if not self.quiet:
            score_str = (
                f"  score={entry['calibra_score']:.0f}"
                if "calibra_score" in entry else ""
            )
            print(
                f"  {icon} [{self.total:>4}] {path.name:<40} "
                f"{verdict:<4} — {details}{score_str}"
            )
            if remediation:
                print(f"       ↳ {remediation}")
            if verdict == "FAIL" and self.bell:
                print("\a", end="", flush=True)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def print_summary(self) -> None:
        elapsed = time.monotonic() - self._start
        print()
        print("━" * 60)
        print("  SESSION SUMMARY")
        print("━" * 60)
        print(f"  Duration : {elapsed:.0f}s")
        print(f"  Episodes : {self.total}  "
              f"(✅ {self.n_pass}  ⚠️  {self.n_warn}  ❌ {self.n_fail})")
        if self.total > 0:
            pass_rate = self.n_pass / self.total * 100
            print(f"  Pass rate: {pass_rate:.0f}%")
        if self.n_fail > 0:
            failed = [e["file"] for e in self.episodes if e["verdict"] == "FAIL"]
            print(f"  Failed   : {', '.join(failed[:5])}"
                  + (" ..." if len(failed) > 5 else ""))
        if self.log_file:
            print(f"  Log      : {self.log_file}")
        print("━" * 60)


# ── polling watcher ────────────────────────────────────────────────────────────

def _poll_watch(
    directory: Path,
    session: WatchSession,
    poll_interval: float,
    policy_family: Optional[str],
    reader,
    max_episodes: Optional[int],
    remediate: bool = False,
) -> None:
    """Poll for new episode files. Works on all OS without extra dependencies."""
    known: set[Path] = set()
    for ext in _WATCH_EXTENSIONS:
        known.update(directory.glob(f"*{ext}"))

    print(f"  Watching: {directory}")
    print(f"  Poll interval: {poll_interval:.1f}s  |  Ctrl+C to stop")
    if remediate:
        print("  Remediation advice: ON")
    print()

    while True:
        if max_episodes is not None and session.total >= max_episodes:
            break

        current: set[Path] = set()
        for ext in _WATCH_EXTENSIONS:
            current.update(directory.glob(f"*{ext}"))

        new_files = sorted(current - known)
        for path in new_files:
            time.sleep(0.5)
            try:
                report = _score_episode_file(path, policy_family, reader)
                verdict, details = _episode_verdict(report)
                session.record(path, verdict, details, report, remediate=remediate)
            except Exception as exc:
                session.record(path, "FAIL", f"load error: {exc}")
            known.add(path)

        time.sleep(poll_interval)


# ── stream mode ────────────────────────────────────────────────────────────────

def _stream_watch(session: WatchSession, remediate: bool = False) -> None:
    """
    Read episode metric JSON from stdin (one line per episode).

    Expected fields: file, ldlj, spike_rate, vel_disc_rate, dropout_rate, jitter_cv.

    Example usage:
        python examples/lerobot_watch_integration.py | calibra watch --stream --remediate
    """
    print("  Stream mode: reading episode metrics from stdin (one JSON per line)")
    print("  Expected fields: file, ldlj, spike_rate, vel_disc_rate, dropout_rate, jitter_cv")
    print()

    _SPIKE_FAIL_S   = 0.05
    _SPIKE_WARN_S   = 0.02
    _VD_FAIL_S      = 0.05
    _VD_WARN_S      = 0.02
    _DROPOUT_FAIL_S = 0.05
    _DROPOUT_WARN_S = 0.01
    _LDLJ_FAIL_S    = -15.0
    _LDLJ_WARN_S    = -10.0

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            print(f"  [skip] invalid JSON: {raw_line[:60]}", flush=True)
            continue

        fname = data.get("file", "episode")
        path = Path(fname)

        verdict = "PASS"
        top_metric = ""
        top_value = 0.0
        details = "all metrics OK"

        checks = [
            ("spike_rate",    data.get("spike_rate"),    _SPIKE_WARN_S,   _SPIKE_FAIL_S,   "higher"),
            ("vel_disc_rate", data.get("vel_disc_rate"), _VD_WARN_S,      _VD_FAIL_S,      "higher"),
            ("dropout_rate",  data.get("dropout_rate"),  _DROPOUT_WARN_S, _DROPOUT_FAIL_S, "higher"),
            ("ldlj",          data.get("ldlj"),          _LDLJ_WARN_S,    _LDLJ_FAIL_S,    "lower"),
        ]
        for metric, val, warn_t, fail_t, direction in checks:
            if val is None:
                continue
            is_fail = val >= fail_t if direction == "higher" else val <= fail_t
            is_warn = val >= warn_t if direction == "higher" else val <= warn_t
            if is_fail and verdict != "FAIL":
                verdict = "FAIL"
                top_metric, top_value = metric, val
                details = f"{metric} = {val:.4g}"
            elif is_warn and verdict == "PASS":
                verdict = "WARN"
                top_metric, top_value = metric, val
                details = f"{metric} = {val:.4g}"

        icon = _verdict_icon(verdict)
        elapsed = time.monotonic() - session._start
        entry: dict = {"t": round(elapsed, 2), "file": fname, "verdict": verdict, "details": details}

        remediation = ""
        if remediate and verdict in ("FAIL", "WARN") and top_metric:
            remediation = _remediation_advice(top_metric, top_value, verdict)
            entry["remediation"] = remediation

        if verdict == "PASS":
            session.n_pass += 1
        elif verdict == "WARN":
            session.n_warn += 1
        else:
            session.n_fail += 1
        session.episodes.append(entry)

        if not session.quiet:
            print(f"  {icon} [{session.total:>4}] {fname:<40} {verdict:<4} — {details}")
            if remediation:
                print(f"       ↳ {remediation}")
            if verdict == "FAIL" and session.bell:
                print("\a", end="", flush=True)

        if session.log_file:
            with open(session.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")


# ── optional watchdog integration ─────────────────────────────────────────────

def _watchdog_available() -> bool:
    try:
        import watchdog  # noqa: F401
        return True
    except ImportError:
        return False


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_watch(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra watch",
        description=(
            "Watch a directory for new episode files and score each one in real time. "
            "Designed for use during teleoperation data collection."
        ),
    )
    p.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Directory to watch for new episode files (not required with --stream)",
    )
    p.add_argument(
        "--format", "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect from extension)",
    )
    p.add_argument(
        "--policy", "-p",
        metavar="FAMILY",
        help="Target policy family for conditioned hints (e.g. 'pi0', 'gr00t')",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="How often to check for new files (default: 1.0s)",
    )
    p.add_argument(
        "--log-file",
        metavar="PATH",
        help="Write per-episode verdicts as JSON Lines to this file",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        metavar="N",
        help="Stop after scoring N episodes",
    )
    p.add_argument(
        "--no-bell",
        action="store_true",
        help="Suppress terminal bell on FAIL episodes",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print summary, not per-episode lines",
    )
    p.add_argument(
        "--remediate",
        action="store_true",
        help=(
            "Print a specific operator instruction on FAIL/WARN episodes: "
            "what caused the failure and how to fix it. "
            "Recommended for live teleoperation sessions."
        ),
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Read episode metric JSON from stdin instead of watching a directory. "
            "Each line: JSON object with fields file, ldlj, spike_rate, vel_disc_rate, "
            "dropout_rate, jitter_cv. "
            "Example: python collect.py | calibra watch --stream --remediate"
        ),
    )
    args = p.parse_args(argv)

    log_file = Path(args.log_file) if args.log_file else None
    session = WatchSession(log_file=log_file, quiet=args.quiet, bell=not args.no_bell)

    print("━" * 60)
    print("  CALIBRA WATCH — real-time data quality monitor")
    print("━" * 60)

    if args.stream:
        try:
            _stream_watch(session, remediate=args.remediate)
        except KeyboardInterrupt:
            pass
        finally:
            session.print_summary()
        sys.exit(0)

    if args.directory is None:
        print("error: a directory argument is required unless --stream is used", file=sys.stderr)
        sys.exit(1)

    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        print(f"error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    try:
        _poll_watch(
            directory=directory,
            session=session,
            poll_interval=args.poll_interval,
            policy_family=args.policy,
            reader=reader,
            max_episodes=args.max_episodes,
            remediate=args.remediate,
        )
    except KeyboardInterrupt:
        pass
    finally:
        session.print_summary()

    sys.exit(0)
