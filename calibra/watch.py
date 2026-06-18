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
               report=None) -> None:
        icon = _verdict_icon(verdict)
        elapsed = time.monotonic() - self._start

        entry: dict = {
            "t": round(elapsed, 2),
            "file": str(path.name),
            "verdict": verdict,
            "details": details,
        }

        if report is not None:
            from calibra.score import compute_score
            try:
                sc = compute_score(report)
                entry["calibra_score"] = sc["total_score"]
            except Exception:
                pass

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


# ── polling watcher (no third-party dependency) ───────────────────────────────

def _poll_watch(
    directory: Path,
    session: WatchSession,
    poll_interval: float,
    policy_family: Optional[str],
    reader,
    max_episodes: Optional[int],
) -> None:
    """
    Poll-based watcher. Detects new files by tracking the set of known paths.
    Works on all OS without extra dependencies.
    """
    known: set[Path] = set()
    # Pre-populate with existing files so we don't score old data on startup
    for ext in _WATCH_EXTENSIONS:
        known.update(directory.glob(f"*{ext}"))

    print(f"  Watching: {directory}")
    print(f"  Poll interval: {poll_interval:.1f}s  |  Ctrl+C to stop")
    print()

    while True:
        if max_episodes is not None and session.total >= max_episodes:
            break

        current: set[Path] = set()
        for ext in _WATCH_EXTENSIONS:
            current.update(directory.glob(f"*{ext}"))

        new_files = sorted(current - known)
        for path in new_files:
            # Wait a moment to ensure the write is complete
            time.sleep(0.5)
            try:
                report = _score_episode_file(path, policy_family, reader)
                verdict, details = _episode_verdict(report)
                session.record(path, verdict, details, report)
            except Exception as exc:
                session.record(path, "FAIL", f"load error: {exc}")
            known.add(path)

        time.sleep(poll_interval)


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
        help="Directory to watch for new episode files",
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
    args = p.parse_args(argv)

    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        print(f"error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    log_file = Path(args.log_file) if args.log_file else None

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    session = WatchSession(
        log_file=log_file,
        quiet=args.quiet,
        bell=not args.no_bell,
    )

    print("━" * 60)
    print("  CALIBRA WATCH — real-time data quality monitor")
    print("━" * 60)

    try:
        _poll_watch(
            directory=directory,
            session=session,
            poll_interval=args.poll_interval,
            policy_family=args.policy,
            reader=reader,
            max_episodes=args.max_episodes,
        )
    except KeyboardInterrupt:
        pass
    finally:
        session.print_summary()

    sys.exit(0)
