"""
Orchestration scaffold for the ADR-011 metadata-conditioning benchmark
(METADATA_CONDITIONING_BENCHMARK.md).

This does **not** train anything. It iterates the frozen matrix
(datasets × architectures × arms × seeds), calls a `train_and_eval` callable
you supply, logs every run through `calibra experiment record`, and aggregates
the results into the protocol's table.

    from run_metadata_benchmark import run

    def train_and_eval(spec, arch, seed):
        # spec: metadata_conditioning_reference.ArmSpec
        #   spec.episode_ids, spec.cond[id], spec.weight[id]
        # return a dict with at least: success (0-1), gpu_hours,
        #   wall_clock_seconds; optionally steps_to_90pct, seed_var,
        #   generalization_success, rare_slice_bottom_q_success,
        #   worst_slice_success, slice_spread
        ...

    run(
        datasets={"droid": "meta/droid", "aloha_mt": "meta/aloha_mt"},
        architectures=["act", "diffusion"],
        seeds=[0, 1, 2],
        train_and_eval=train_and_eval,
        experiment_id="partner-x-metadata",
    )

`--dry-run` (CLI) prints the plan — arm membership, nominal vs. actual
retention, conditioning dim — without a training callable, so the partner can
sanity-check the matrix before spending GPU.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from metadata_conditioning_reference import ARMS, ArmSpec, prepare_arm

from calibra.experiment_log import ExperimentLog

# condition label for `calibra experiment record` (the data-selection axis;
# metadata_conditioning is orthogonal)
_ARM_CONDITION = {
    "A": "full",
    "B": "calibra",
    "C": "full",
    "D": "calibra",
    "D0": "calibra",
    "R": "random",
    "R+": "random",
}

_METRIC_FIELDS = (
    "success",
    "gpu_hours",
    "wall_clock_seconds",
    "steps_to_90pct",
    "generalization_success",
    "rare_slice_bottom_q_success",
    "worst_slice_success",
    "slice_spread",
)

TrainAndEval = Callable[[ArmSpec, str, int], dict]


def run(
    *,
    datasets: dict[str, str],
    architectures: list[str],
    seeds: list[int],
    train_and_eval: Optional[TrainAndEval],
    experiment_id: str,
    arms: Optional[list[str]] = None,
    out_dir: str = "metadata_benchmark_results",
    log_path: Optional[str] = None,
    dry_run: bool = False,
) -> list[dict]:
    arms = arms or ["A", "B", "C", "D", "R", "R+"]
    log = ExperimentLog(path=Path(log_path) if log_path else None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for ds_name, sidecar_dir in datasets.items():
        for arch in architectures:
            for arm in arms:
                base = prepare_arm(sidecar_dir, arm, seed=seeds[0])
                _print_plan(ds_name, arch, arm, base)
                if dry_run:
                    continue
                if train_and_eval is None:
                    raise SystemExit("no train_and_eval callable — pass one, or use dry_run")

                for seed in seeds:
                    spec = prepare_arm(sidecar_dir, arm, seed=seed)
                    t0 = time.time()
                    metrics = train_and_eval(spec, arch, seed)
                    metrics.setdefault("wall_clock_seconds", time.time() - t0)

                    log.record(
                        experiment_id=experiment_id,
                        condition=_ARM_CONDITION[arm],
                        arm=arm,
                        metadata_conditioning=ARMS[arm]["metadata"],
                        retention_pct=spec.nominal_keep_pct if arm in _ARM_CONDITION and _ARM_CONDITION[arm] != "full" else 100.0,
                        actual_retention_pct=spec.actual_retention_pct,
                        dataset_name=ds_name,
                        policy_family=arch,
                        n_episodes=len(spec.episode_ids),
                        seed=seed,
                        gpu_hours=metrics.get("gpu_hours"),
                        wall_clock_seconds=metrics.get("wall_clock_seconds"),
                        energy_kwh=metrics.get("energy_kwh"),
                        eval_success_rate=metrics.get("success"),
                        notes=metrics.get("notes", ""),
                    )
                    rows.append(
                        dict(
                            dataset=ds_name, arch=arch, arm=arm,
                            metadata=ARMS[arm]["metadata"], seed=seed,
                            nominal_retention_pct=spec.nominal_keep_pct,
                            actual_retention_pct=spec.actual_retention_pct,
                            n_episodes=len(spec.episode_ids),
                            **{k: metrics.get(k) for k in _METRIC_FIELDS},
                        )
                    )

    if rows:
        _write_runs(out / "runs.csv", rows)
        _write_summary(out / "summary.md", rows)
        print(f"\nwrote {out/'runs.csv'} and {out/'summary.md'}")
    return rows


def _print_plan(ds: str, arch: str, arm: str, spec: ArmSpec) -> None:
    print(
        f"{ds:12} {arch:10} arm {arm:3}  "
        f"n={len(spec.episode_ids):5}  nominal_keep={spec.nominal_keep_pct:5.1f}%  "
        f"actual={spec.actual_retention_pct:5.1f}%  cond_dim={spec.cond_dim}  "
        f"meta={ARMS[arm]['metadata']}"
    )


def _write_runs(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _agg(vals: list) -> str:
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return "—"
    return f"{st.mean(v):.3f}" + (f" ± {st.pstdev(v):.3f}" if len(v) > 1 else "")


def _write_summary(path: Path, rows: list[dict]) -> None:
    keys = sorted({(r["dataset"], r["arch"]) for r in rows})
    lines = ["# Metadata-conditioning benchmark — summary", ""]
    for ds, arch in keys:
        lines += [f"## {ds} · {arch}", "",
                  "| arm | meta | actual ret. | success | gpu_hours | "
                  "rare-slice bottom-q | worst slice |",
                  "|---|---|---|---|---|---|---|"]
        for arm in ["A", "B", "C", "D", "R", "R+", "D0"]:
            sub = [r for r in rows if r["dataset"] == ds and r["arch"] == arch and r["arm"] == arm]
            if not sub:
                continue
            lines.append(
                f"| {arm} | {sub[0]['metadata']} | {sub[0]['actual_retention_pct']:.0f}% | "
                f"{_agg([r['success'] for r in sub])} | {_agg([r['gpu_hours'] for r in sub])} | "
                f"{_agg([r['rare_slice_bottom_q_success'] for r in sub])} | "
                f"{_agg([r['worst_slice_success'] for r in sub])} |"
            )
        lines.append("")
    lines += ["> Read against arm A. See METADATA_CONDITIONING_BENCHMARK.md §6 "
              "for the decision-rule table."]
    path.write_text("\n".join(lines))


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar", action="append", metavar="NAME=DIR", required=True,
                   help="dataset name and its --annotate output dir (repeatable)")
    p.add_argument("--arch", action="append", default=None, help="default: act, diffusion")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--experiment-id", default="metadata-benchmark")
    p.add_argument("--arms", default="A,B,C,D,R,R+")
    p.add_argument("--out-dir", default="metadata_benchmark_results")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan without training (no callable needed)")
    args = p.parse_args()

    datasets = dict(s.split("=", 1) for s in args.sidecar)
    if not args.dry_run:
        print("This scaffold does not train. Import run() and pass train_and_eval, "
              "or use --dry-run.", file=sys.stderr)
        sys.exit(2)
    run(
        datasets=datasets,
        architectures=args.arch or ["act", "diffusion"],
        seeds=[int(s) for s in args.seeds.split(",")],
        train_and_eval=None,
        experiment_id=args.experiment_id,
        arms=args.arms.split(","),
        out_dir=args.out_dir,
        dry_run=True,
    )


if __name__ == "__main__":
    _cli()
