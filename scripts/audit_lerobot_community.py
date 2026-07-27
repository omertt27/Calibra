"""
audit_lerobot_community.py — Calibra community dataset quality benchmark.

Audits a curated list of public LeRobot datasets and writes:
  results/community/manifest.json       — machine-readable summary
  results/community/leaderboard.md      — markdown table for HF dataset card

Usage:
    python scripts/audit_lerobot_community.py
    python scripts/audit_lerobot_community.py --out ./my-results --workers 4
    python scripts/audit_lerobot_community.py --limit 5          # quick test
    python scripts/audit_lerobot_community.py --force            # re-audit cached

Output leaderboard.md is ready to paste into a Hugging Face dataset card or README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── curated dataset list ──────────────────────────────────────────────────────
#
# Priority selection: widely-cited, diverse embodiments and tasks.
# Sorted roughly by size / prominence.

COMMUNITY_DATASETS: list[str] = [
    # PushT (classic 2D task, ~200 episodes)
    "lerobot/pusht",
    # ALOHA simulated tasks
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_insertion_scripted",
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_transfer_cube_scripted",
    # ALOHA static manipulation tasks
    "lerobot/aloha_static_battery",
    "lerobot/aloha_static_candy",
    "lerobot/aloha_static_coffee",
    "lerobot/aloha_static_cups_open",
    "lerobot/aloha_static_fork_pick_up",
    "lerobot/aloha_static_pingpong_test",
    "lerobot/aloha_static_screw_driver",
    "lerobot/aloha_static_tape",
    "lerobot/aloha_static_thread_velcro",
    "lerobot/aloha_static_towel",
    "lerobot/aloha_static_ziploc_slide",
    # ALOHA mobile tasks
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_mobile_chair",
    "lerobot/aloha_mobile_elevator",
    "lerobot/aloha_mobile_shrimp",
    "lerobot/aloha_mobile_wash_pan",
    "lerobot/aloha_mobile_wipe_wine",
    # xArm tasks
    "lerobot/xarm_lift_medium",
    "lerobot/xarm_lift_medium_replay",
    "lerobot/xarm_push_medium",
    "lerobot/xarm_push_medium_replay",
    "lerobot/xarm_lift_medium_unlabeled",
    # Unitree H1 humanoid tasks
    "lerobot/unitreeh1_fold_clothes",
    "lerobot/unitreeh1_rearrange_objects",
    "lerobot/unitreeh1_two_robot_greeting",
    "lerobot/unitreeh1_warehouse",
]

# ── audit helpers ─────────────────────────────────────────────────────────────


def _hf_revision(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo_id=repo_id)
        return getattr(info, "sha", None)
    except Exception:
        return None


def _result_dir(out: Path, repo_id: str, revision: Optional[str]) -> Path:
    rev = revision[:8] if revision else "unknown"
    return out / repo_id / rev


def _already_cached(out: Path, repo_id: str, revision: Optional[str]) -> bool:
    d = _result_dir(out, repo_id, revision)
    return d.exists() and any(f for f in d.glob("*.json") if "FAILED" not in f.name)


def _audit_one(
    index: int,
    total: int,
    repo_id: str,
    out: Path,
    force: bool,
) -> dict:
    label = f"[{index}/{total}] {repo_id}"
    revision = _hf_revision(repo_id)

    if not force and _already_cached(out, repo_id, revision):
        short = revision[:8] if revision else "unknown"
        print(f"{label}  skipped ({short} cached)", flush=True)
        latest = out / repo_id / "latest.json"
        if latest.exists():
            data = json.loads(latest.read_text(encoding="utf-8"))
            return {
                "repository_id": repo_id,
                "status": "skipped",
                "score": data.get("results", {}).get("overall", {}).get("score"),
                "grade": data.get("results", {}).get("overall", {}).get("grade"),
                "certification": data.get("results", {}).get("overall", {}).get("certification"),
                "n_episodes": data.get("dataset", {}).get("episodes_total"),
                "n_frames": data.get("dataset", {}).get("frames_total"),
                "n_critical": len(
                    [f for f in data.get("results", {}).get("findings", [])
                     if f.get("severity") == "critical"]
                ),
                "revision": revision,
                "error": None,
            }
        return {"repository_id": repo_id, "status": "skipped", "score": None, "grade": None,
                "certification": None, "n_episodes": None, "n_frames": None,
                "n_critical": None, "revision": revision, "error": None}

    t0 = time.monotonic()
    print(f"{label}  auditing ...", flush=True)

    try:
        from calibra.pipeline import Pipeline
        from calibra.report_json import assemble_public_report
        from calibra.schema.public_report import DatasetInfo, SamplingConfig

        diag = Pipeline().analyze_path(repo_id)

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

        rev_dir = _result_dir(out, repo_id, revision)
        rev_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        report_path = rev_dir / f"{ts}.json"
        public.write(str(report_path))
        (out / repo_id / "latest.json").parent.mkdir(parents=True, exist_ok=True)
        public.write(str(out / repo_id / "latest.json"))

        overall = public.results.overall
        n_critical = len([f for f in public.results.findings if f.severity == "critical"])
        duration = time.monotonic() - t0

        print(
            f"{label}  OK  score={overall.score:.1f} grade={overall.grade} "
            f"cert={overall.certification}  {duration:.1f}s",
            flush=True,
        )

        dim_scores = {
            name: {"score": dim.score, "weight": dim.weight}
            for name, dim in public.results.dimensions.items()
        }

        return {
            "repository_id": repo_id,
            "status": "ok",
            "score": overall.score,
            "grade": overall.grade,
            "certification": overall.certification,
            "n_episodes": diag.n_episodes,
            "n_frames": diag.n_samples,
            "n_critical": n_critical,
            "dimensions": dim_scores,
            "revision": revision,
            "duration_s": round(duration, 1),
            "error": None,
        }

    except Exception as exc:
        duration = time.monotonic() - t0
        err = str(exc)[:200]
        print(f"{label}  FAILED  {err}  {duration:.1f}s", flush=True)

        fail_dir = _result_dir(out, repo_id, revision)
        fail_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        (fail_dir / f"{ts}_FAILED.json").write_text(
            json.dumps({"repository_id": repo_id, "error": str(exc)}), encoding="utf-8"
        )
        return {
            "repository_id": repo_id,
            "status": "failed",
            "score": None,
            "grade": None,
            "certification": None,
            "n_episodes": None,
            "n_frames": None,
            "n_critical": None,
            "revision": revision,
            "duration_s": round(duration, 1),
            "error": err,
        }


# ── leaderboard renderer ──────────────────────────────────────────────────────

_CERT_LABEL = {
    "pass": "✓ Certified",
    "provisional": "~ Provisional",
    "fail": "✗ Not Certified",
}


def _render_leaderboard(results: list[dict], generated_at: str) -> str:
    ok_results = sorted(
        [r for r in results if r["score"] is not None],
        key=lambda r: r["score"],
        reverse=True,
    )
    failed = [r for r in results if r["status"] == "failed"]

    lines = [
        "# Calibra Community Dataset Quality Benchmark",
        "",
        f"*Generated {generated_at} by [Calibra](https://github.com/omertt27/Calibra)*",
        "",
        f"Audited **{len(ok_results)}** public LeRobot datasets. "
        f"{'**' + str(len(failed)) + ' datasets failed to load.**' if failed else ''}",
        "",
        "## Leaderboard",
        "",
        "| # | Dataset | Score | Grade | Certification | Episodes | Frames | Critical |",
        "|---|---------|------:|-------|---------------|--------:|-------:|--------:|",
    ]

    for rank, r in enumerate(ok_results, 1):
        cert = _CERT_LABEL.get(r.get("certification", ""), "—")
        ep = f"{r['n_episodes']:,}" if r.get("n_episodes") else "—"
        fr = f"{r['n_frames']:,}" if r.get("n_frames") else "—"
        crit = str(r["n_critical"]) if r.get("n_critical") is not None else "—"
        short_id = r["repository_id"].split("/", 1)[-1]
        hf_link = f"[{short_id}](https://huggingface.co/datasets/{r['repository_id']})"
        lines.append(
            f"| {rank} | {hf_link} | **{r['score']:.1f}** | {r['grade']} "
            f"| {cert} | {ep} | {fr} | {crit} |"
        )

    if failed:
        lines += [
            "",
            "## Failed to Load",
            "",
            "| Dataset | Error |",
            "|---------|-------|",
        ]
        for r in failed:
            short_id = r["repository_id"].split("/", 1)[-1]
            err = (r.get("error") or "unknown error")[:80]
            lines.append(f"| {short_id} | {err} |")

    lines += [
        "",
        "## Score Interpretation",
        "",
        "| Score | Grade | Certification |",
        "|-------|-------|---------------|",
        "| 90–100 | A | ✓ Certified |",
        "| 75–89 | B | ✓ Certified |",
        "| 60–74 | C | ~ Provisional |",
        "| 40–59 | D | ✗ Not Certified |",
        "| 0–39 | F | ✗ Not Certified |",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install 'calibra-robotics[lerobot]'",
        "python scripts/audit_lerobot_community.py",
        "```",
    ]

    return "\n".join(lines)


# ── manifest writer ───────────────────────────────────────────────────────────


def _write_manifest(results: list[dict], out: Path, generated_at: str) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    scores = [r["score"] for r in results if r.get("score") is not None]

    manifest = {
        "generated_at": generated_at,
        "calibra_version": _calibra_version(),
        "summary": {
            "total": len(results),
            "audited": len(ok),
            "skipped": len(skipped),
            "failed": len(failed),
            "mean_score": round(sum(scores) / len(scores), 1) if scores else None,
            "min_score": round(min(scores), 1) if scores else None,
            "max_score": round(max(scores), 1) if scores else None,
        },
        "datasets": results,
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # community_stats.json — loaded by the HF Space for percentile comparison
    if scores:
        import statistics
        community_stats = {
            "generated_at": generated_at,
            "n_datasets": len(scores),
            "scores": sorted(scores),
            "mean": round(sum(scores) / len(scores), 1),
            "median": round(statistics.median(scores), 1),
            "p25": round(sorted(scores)[len(scores) // 4], 1),
            "p75": round(sorted(scores)[3 * len(scores) // 4], 1),
            "dimensions": _community_dimension_stats(results),
        }
        (out / "community_stats.json").write_text(
            json.dumps(community_stats, indent=2), encoding="utf-8"
        )


def _community_dimension_stats(results: list[dict]) -> dict:
    """Aggregate per-dimension mean scores across all audited datasets."""
    dim_scores: dict[str, list[float]] = {}
    for r in results:
        for dim_name, dim_data in r.get("dimensions", {}).items():
            s = dim_data.get("score") if isinstance(dim_data, dict) else None
            if s is not None:
                dim_scores.setdefault(dim_name, []).append(s)
    return {
        name: round(sum(vals) / len(vals), 1)
        for name, vals in dim_scores.items()
        if vals
    }


def _calibra_version() -> str:
    try:
        from calibra import __version__
        return __version__
    except Exception:
        return "unknown"


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Calibra community dataset quality benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--out", default="results/community", metavar="DIR",
                   help="Output directory (default: results/community)")
    p.add_argument("--workers", type=int, default=2, metavar="N",
                   help="Parallel workers (default: 2; HF Hub rate-limits concurrent requests)")
    p.add_argument("--force", action="store_true",
                   help="Re-audit even when a cached report exists")
    p.add_argument("--limit", type=int, metavar="N",
                   help="Audit only the first N datasets (for testing)")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = COMMUNITY_DATASETS[: args.limit] if args.limit else COMMUNITY_DATASETS
    total = len(datasets)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"Calibra community benchmark — {total} datasets — {generated_at}")
    print(f"Output: {out.resolve()}")
    print()

    results: list[dict] = []

    if args.workers == 1:
        for i, repo_id in enumerate(datasets, 1):
            results.append(_audit_one(i, total, repo_id, out, args.force))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_audit_one, i, total, repo_id, out, args.force): repo_id
                for i, repo_id in enumerate(datasets, 1)
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    # restore discovery order
    order = {repo_id: i for i, repo_id in enumerate(datasets)}
    results.sort(key=lambda r: order.get(r["repository_id"], 9999))

    _write_manifest(results, out, generated_at)

    leaderboard_md = _render_leaderboard(results, generated_at)
    (out / "leaderboard.md").write_text(leaderboard_md, encoding="utf-8")

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    scores = [r["score"] for r in results if r.get("score") is not None]
    mean = f"{sum(scores) / len(scores):.1f}" if scores else "n/a"

    print()
    print(f"Done.  audited={ok}  skipped={skipped}  failed={failed}  mean_score={mean}")
    print(f"Manifest:    {out / 'manifest.json'}")
    print(f"Leaderboard: {out / 'leaderboard.md'}")

    if failed:
        fail_ids = [r["repository_id"] for r in results if r["status"] == "failed"]
        print(f"\nFailed datasets ({len(fail_ids)}):")
        for fid in fail_ids:
            print(f"  {fid}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
