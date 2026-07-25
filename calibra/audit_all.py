"""
calibra audit-all — bulk dataset auditor.

Five jobs per dataset, nothing else:
  1. discover  — list datasets in HF org or accept explicit repo IDs
  2. resolve   — fetch current revision SHA from HF Hub
  3. audit     — Pipeline().analyze_path()
  4. validate  — assemble_public_report() → schema-validated CalibraReport
  5. write     — persist to results/<org>/<slug>/<revision-sha[:8]>/<timestamp>.json
                 and update results/<org>/<slug>/latest.json

Does NOT generate HTML, badges, or leaderboard pages.
Those are independent consumers of the JSON files this command produces.

Usage:
    calibra audit-all --org lerobot
    calibra audit-all --org lerobot --out ./results --workers 8
    calibra audit-all --dataset lerobot/pusht lerobot/aloha_sim_insertion_human
    calibra audit-all --org lerobot --force
    calibra audit-all --org lerobot --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── internal data classes ─────────────────────────────────────────────────────


@dataclass
class _Entry:
    repo_id: str
    revision: Optional[str]
    index: int
    total: int
    fmt: Optional[str]  # forced adapter name, or None = auto-detect


@dataclass
class _Result:
    repo_id: str
    revision: Optional[str]
    status: str  # "ok" | "skipped" | "failed"
    score: Optional[float] = None
    grade: Optional[str] = None
    certification: Optional[str] = None
    duration_s: float = 0.0
    error: Optional[str] = None
    report_path: Optional[str] = None


# ── discovery ─────────────────────────────────────────────────────────────────


def _hf_api(token: Optional[str]):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "error: huggingface_hub is required for audit-all.\n"
            "       Install with: pip install huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def _discover(
    org: Optional[str],
    explicit: Optional[list[str]],
    token: Optional[str],
    limit: Optional[int],
) -> list[tuple[str, Optional[str]]]:
    """Return (repo_id, revision_sha_or_None) pairs."""
    api = _hf_api(token)
    entries: list[tuple[str, Optional[str]]] = []

    if org:
        for ds in api.list_datasets(author=org):
            sha = getattr(ds, "sha", None)
            entries.append((ds.id, sha))
            if limit and len(entries) >= limit:
                break

    if explicit:
        for repo_id in explicit:
            if limit and len(entries) >= limit:
                break
            try:
                info = api.dataset_info(repo_id=repo_id)
                sha = getattr(info, "sha", None)
            except Exception:
                sha = None
            entries.append((repo_id, sha))

    return entries


# ── file layout helpers ───────────────────────────────────────────────────────


def _revision_dir(out_dir: Path, repo_id: str, revision: Optional[str]) -> Path:
    """results/<org>/<slug>/<revision[:8]>/"""
    rev = revision[:8] if revision else "unknown"
    return out_dir / repo_id / rev


def _latest_path(out_dir: Path, repo_id: str) -> Path:
    """results/<org>/<slug>/latest.json"""
    return out_dir / repo_id / "latest.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _already_audited(out_dir: Path, repo_id: str, revision: Optional[str]) -> bool:
    d = _revision_dir(out_dir, repo_id, revision)
    return d.exists() and any(f for f in d.glob("*.json") if "FAILED" not in f.name)


# ── single-dataset audit ──────────────────────────────────────────────────────


def _audit_one(entry: _Entry, out_dir: Path, force: bool) -> _Result:
    repo_id = entry.repo_id
    revision = entry.revision
    label = f"[{entry.index}/{entry.total}] {repo_id}"

    if not force and _already_audited(out_dir, repo_id, revision):
        rev_short = revision[:8] if revision else "unknown"
        print(f"{label}  skipped ({rev_short} cached)", file=sys.stderr, flush=True)
        return _Result(repo_id=repo_id, revision=revision, status="skipped")

    t0 = time.monotonic()
    print(f"{label}  auditing ...", file=sys.stderr, flush=True)

    try:
        from calibra.pipeline import Pipeline
        from calibra.report_json import assemble_public_report
        from calibra.schema.public_report import DatasetInfo, SamplingConfig

        reader = None
        if entry.fmt:
            from calibra.__main__ import _get_reader

            reader = _get_reader(entry.fmt)

        diag = Pipeline().analyze_path(repo_id, reader=reader)

        dataset_info = DatasetInfo(
            provider="huggingface",
            repository_id=repo_id,
            revision=revision,
            dataset_format=diag.format,
            episodes_total=diag.n_episodes,
            episodes_audited=diag.n_episodes,
            frames_total=diag.n_samples,
        )

        public = assemble_public_report(
            diag,
            dataset_info=dataset_info,
            sampling=SamplingConfig(mode="full"),
        )

        # Write revision-stamped report
        rev_dir = _revision_dir(out_dir, repo_id, revision)
        ts = _timestamp()
        report_path = rev_dir / f"{ts}.json"
        public.write(str(report_path))

        # Update latest.json (always overwrites)
        public.write(str(_latest_path(out_dir, repo_id)))

        duration = time.monotonic() - t0
        score = public.results.overall.score
        grade = public.results.overall.grade
        cert = public.results.overall.certification

        print(
            f"{label}  OK  score={score:.1f} grade={grade} cert={cert}  {duration:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return _Result(
            repo_id=repo_id,
            revision=revision,
            status="ok",
            score=score,
            grade=grade,
            certification=cert,
            duration_s=round(duration, 2),
            report_path=str(report_path),
        )

    except Exception as exc:
        duration = time.monotonic() - t0
        err = str(exc)
        print(f"{label}  FAILED  {err[:100]}  {duration:.1f}s", file=sys.stderr, flush=True)

        # Write failure marker so the run can be resumed with --force
        fail_dir = _revision_dir(out_dir, repo_id, revision)
        fail_dir.mkdir(parents=True, exist_ok=True)
        ts = _timestamp()
        (fail_dir / f"{ts}_FAILED.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "status": "failed",
                    "repository_id": repo_id,
                    "revision": revision,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "error": err,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return _Result(
            repo_id=repo_id,
            revision=revision,
            status="failed",
            duration_s=round(duration, 2),
            error=err,
        )


# ── manifest ──────────────────────────────────────────────────────────────────


def _write_manifest(results: list[_Result], out_dir: Path, args_summary: dict) -> Path:
    n_ok = sum(1 for r in results if r.status == "ok")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_failed = sum(1 for r in results if r.status == "failed")
    scores = [r.score for r in results if r.score is not None]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": args_summary,
        "summary": {
            "total": len(results),
            "audited": n_ok,
            "skipped": n_skipped,
            "failed": n_failed,
            "mean_score": round(sum(scores) / len(scores), 1) if scores else None,
            "grade_distribution": _grade_dist(results),
        },
        "datasets": [
            {
                "repository_id": r.repo_id,
                "revision": r.revision,
                "status": r.status,
                "score": r.score,
                "grade": r.grade,
                "certification": r.certification,
                "duration_s": r.duration_s,
                "error": r.error,
                "report_path": r.report_path,
            }
            for r in sorted(results, key=lambda r: r.score or 0, reverse=True)
        ],
    }

    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


def _grade_dist(results: list[_Result]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in results:
        if r.grade:
            dist[r.grade] = dist.get(r.grade, 0) + 1
    return dist


# ── CLI ───────────────────────────────────────────────────────────────────────


def run_audit_all(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra audit-all",
        description="Bulk-audit a HuggingFace org or explicit dataset list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  calibra audit-all --org lerobot\n"
            "  calibra audit-all --org lerobot --workers 8 --out ./results\n"
            "  calibra audit-all --dataset lerobot/pusht lerobot/aloha_sim_insertion_human\n"
            "  calibra audit-all --org lerobot --force\n"
            "  calibra audit-all --org lerobot --limit 5 --dry-run\n"
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--org", metavar="ORG", help="HuggingFace org name (e.g. 'lerobot')")
    src.add_argument(
        "--dataset",
        nargs="+",
        metavar="REPO_ID",
        help="Explicit HF repo IDs (e.g. lerobot/pusht lerobot/aloha_sim_insertion_human)",
    )

    p.add_argument(
        "--out",
        metavar="DIR",
        default="results",
        help="Output directory for JSON reports (default: ./results)",
    )
    p.add_argument(
        "--workers", type=int, default=4, metavar="N", help="Parallel audit workers (default: 4)"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-audit datasets even when a cached revision report exists",
    )
    p.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Cap number of datasets (useful for dry-runs and testing)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and resolve revisions without running audits",
    )
    p.add_argument(
        "--format",
        "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force adapter for all datasets (default: auto-detect per dataset)",
    )
    p.add_argument(
        "--token", metavar="TOKEN", help="HuggingFace API token (or set HF_TOKEN env var)"
    )

    args = p.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── discover ──────────────────────────────────────────────────────────────
    source_label = f"org={args.org}" if args.org else f"{len(args.dataset)} explicit dataset(s)"
    print(f"Discovering datasets ({source_label}) ...", file=sys.stderr, flush=True)

    raw = _discover(
        org=args.org,
        explicit=args.dataset,
        token=args.token,
        limit=args.limit,
    )

    if not raw:
        print("No datasets found.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(raw)} dataset(s).", file=sys.stderr, flush=True)

    if args.dry_run:
        for repo_id, revision in raw:
            cached = _already_audited(out_dir, repo_id, revision)
            status = "cached" if cached else "pending"
            rev_short = revision[:8] if revision else "unknown"
            print(f"  {repo_id}  rev={rev_short}  {status}")
        sys.exit(0)

    # ── audit ─────────────────────────────────────────────────────────────────
    entries = [
        _Entry(repo_id=rid, revision=rev, index=i + 1, total=len(raw), fmt=args.format)
        for i, (rid, rev) in enumerate(raw)
    ]

    results: list[_Result] = []

    if args.workers == 1:
        for entry in entries:
            results.append(_audit_one(entry, out_dir, force=args.force))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_audit_one, e, out_dir, args.force): e for e in entries}
            for fut in as_completed(futures):
                results.append(fut.result())

    # Restore discovery order for manifest readability
    order = {e.repo_id: e.index for e in entries}
    results.sort(key=lambda r: order.get(r.repo_id, 9999))

    # ── manifest + summary ────────────────────────────────────────────────────
    args_summary = {
        "org": args.org,
        "datasets": args.dataset,
        "workers": args.workers,
        "force": args.force,
        "format": args.format,
        "limit": args.limit,
    }
    manifest_path = _write_manifest(results, out_dir, args_summary)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_failed = sum(1 for r in results if r.status == "failed")
    scores = [r.score for r in results if r.score is not None]
    mean = f"{sum(scores) / len(scores):.1f}" if scores else "n/a"

    print(f"\nDone.  audited={n_ok}  skipped={n_skipped}  failed={n_failed}  mean_score={mean}")
    print(f"Manifest: {manifest_path}")
    if n_failed:
        print(f"Failed datasets: {[r.repo_id for r in results if r.status == 'failed']}")

    sys.exit(1 if n_failed > 0 else 0)
