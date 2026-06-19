"""
calibra prune — coreset selection for robot imitation learning datasets.

Two-stage pruning pipeline:
  Stage 1: Quality filtering — remove episodes that fail kinematic/temporal thresholds.
  Stage 2: Greedy max-coverage — from quality-passing pool, select the most
           behaviorally diverse subset of the requested size.

Usage:
    calibra prune /path/to/dataset --keep 0.3
    calibra prune lerobot/pusht --keep 0.5 --quality-only --format lerobot
    calibra prune /data/my_ds --keep 0.4 --out coreset_index.json
    calibra prune /data/my_ds --keep 0.3 --max-spike-rate 0.05 --json

Exit codes:
    0  Pruning completed successfully.
    1  Error loading dataset or running pipeline.

Output
------
The pruning result is written as JSON to --out (default: coreset_index.json).
The JSON contains:
  keep_episode_ids     — episode IDs to retain
  quality_fail_ids     — episode IDs removed by Stage 1
  diversity_pruned_ids — episode IDs removed by Stage 2
  quality_scores       — per-episode composite quality score (lower = cleaner)
  diversity_scores     — per-episode min-distance-to-selected score
  n_original, n_kept, keep_fraction_actual, method

To apply to a LeRobot v2 dataset, use the episode IDs to filter your Parquet shards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from calibra.pipeline import Pipeline
from calibra.pruning import ApproximateCoresetSelector, CoresetSelector

_APPROXIMATE_AUTO_THRESHOLD = 50_000  # auto-enable approximate mode above this episode count


def run_prune(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra prune",
        description="Select a high-quality, behaviorally diverse coreset from a robot dataset.",
    )
    p.add_argument("path", help="Path or Hub ID of the dataset to prune")
    p.add_argument(
        "--keep", "-k",
        type=float,
        default=0.5,
        metavar="FRACTION",
        help="Target fraction of episodes to keep (default: 0.5)",
    )
    p.add_argument(
        "--out", "-o",
        metavar="PATH",
        default="coreset_index.json",
        help="Output JSON file path (default: coreset_index.json)",
    )
    p.add_argument(
        "--quality-only",
        action="store_true",
        help="Stage 1 only — filter quality failures but skip diversity selection",
    )
    p.add_argument(
        "--format", "-f",
        metavar="FMT",
        choices=["hdf5", "isaac_lab", "lerobot", "rlds", "mcap"],
        help="Force a format adapter (default: auto-detect)",
    )
    p.add_argument(
        "--policy",
        metavar="FAMILY",
        help=(
            "Target policy family. Use 'gr00t' to apply GR00T-optimised defaults: "
            "stricter quality thresholds and entropy-weighted diversity selection."
        ),
    )

    # Stage 1 quality thresholds
    q = p.add_argument_group("Stage 1 quality thresholds (override defaults)")
    q.add_argument("--max-spike-rate",    type=float, default=None,
                   help="Max jerk spike fraction per episode (default: 0.10; "
                        "auto-raised to 0.30 when scripted motion signature is detected)")
    q.add_argument("--max-vel-disc-rate", type=float, default=0.25,
                   help="Max velocity discontinuity fraction (default: 0.25)")
    q.add_argument("--max-dropout",       type=float, default=0.10,
                   help="Max frame dropout fraction (default: 0.10)")
    q.add_argument("--min-ldlj",          type=float, default=-30.0,
                   help="Min LDLJ value (more negative = worse, default: -30.0)")
    q.add_argument("--min-length",        type=int,   default=10,
                   help="Min episode length in steps (default: 10)")

    # Stage 2 diversity
    d = p.add_argument_group("Stage 2 diversity selection")
    d.add_argument("--diversity-weight", type=float, default=0.7,
                   help="Weight of action-space features vs quality features "
                        "in diversity computation (0–1, default: 0.7)")
    d.add_argument("--entropy-weight", type=float, default=0.0,
                   help="Weight of per-trajectory Shannon entropy in the diversity "
                        "feature vector (0–1, default: 0). Set > 0 to preferentially "
                        "retain high-entropy (informative) episodes. "
                        "Automatically set to 0.4 when --policy gr00t is used.")
    d.add_argument("--strategy", choices=["diversity", "influence"], default="diversity",
                   help="Coreset selection strategy (default: diversity)")
    d.add_argument("--approximate", action="store_true",
                   help="Use approximate MiniBatch diversity selection (O(N×B) instead of "
                        "O(N×K)). Auto-enabled when N > 50 000. Suitable for 100k+ episodes.")
    d.add_argument("--batch-size", type=int, default=1000,
                   help="MiniBatch size for --approximate mode (default: 1000)")

    p.add_argument(
        "--export-dataset",
        metavar="DIR",
        default=None,
        help=(
            "Materialise the coreset as a ready-to-train dataset in DIR. "
            "Supports LeRobot v2 (Parquet), LeRobot v1, and HDF5 sources. "
            "Hub IDs must be downloaded locally first."
        ),
    )
    p.add_argument("--json", "-j", action="store_true",
                   help="Print full JSON result to stdout in addition to writing --out")
    args = p.parse_args(argv)

    if not (0.0 < args.keep <= 1.0):
        print("error: --keep must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    dataset_path = args.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[len("hf://"):]

    reader = None
    if args.format:
        from calibra.__main__ import _get_reader
        reader = _get_reader(args.format)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {dataset_path!r} ...")

    try:
        from calibra.ingestion.registry import load
        batch = load(dataset_path, reader=reader)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {batch.n_episodes} episodes  ·  {batch.n_samples} steps")
    log("Running diagnostic pipeline ...")

    try:
        report = Pipeline().run(batch)
    except Exception as exc:
        print(f"error running pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    log("Running coreset selection ...")

    # Apply GR00T-specific defaults before building the selector.
    user_set_spike_rate = args.max_spike_rate is not None
    max_spike_rate    = args.max_spike_rate if user_set_spike_rate else 0.10
    max_vel_disc_rate = args.max_vel_disc_rate
    max_dropout       = args.max_dropout
    diversity_weight  = args.diversity_weight
    entropy_weight    = args.entropy_weight

    if args.policy and "gr00t" in args.policy.lower():
        # GR00T fine-tuning is sensitive to jerk and discontinuities.
        max_spike_rate    = min(max_spike_rate,    0.05)
        max_vel_disc_rate = min(max_vel_disc_rate, 0.10)
        max_dropout       = min(max_dropout,       0.05)
        if entropy_weight == 0.0:
            entropy_weight = 0.4   # prefer informative episodes by default
        log("  [--policy gr00t] Applying GR00T quality thresholds and entropy weighting.")

    # Auto-adjust spike threshold for scripted/planner datasets.
    # Scripted demos have spike_rate ~22–25% (planner waypoint transitions),
    # which would discard the entire dataset under the default threshold of 0.10.
    # When the scripted motion signature is detected and the user has not
    # explicitly set --max-spike-rate, raise it to 0.30 automatically.
    _SCRIPTED_AUTO_SPIKE = 0.30
    if not user_set_spike_rate:
        is_scripted = any(
            f.metric == "motion_collection_signature"
            for r in report.analyzer_results
            for f in r.flags
        )
        if is_scripted and max_spike_rate < _SCRIPTED_AUTO_SPIKE:
            log(
                f"  [scripted data] Scripted motion signature detected. "
                f"Auto-raised --max-spike-rate from {max_spike_rate:.2f} to "
                f"{_SCRIPTED_AUTO_SPIKE:.2f}. Use --max-spike-rate to override."
            )
            max_spike_rate = _SCRIPTED_AUTO_SPIKE

    # Auto-enable approximate mode for very large datasets.
    use_approximate = args.approximate or batch.n_episodes > _APPROXIMATE_AUTO_THRESHOLD
    if use_approximate and batch.n_episodes > _APPROXIMATE_AUTO_THRESHOLD and not args.approximate:
        log(
            f"  [large dataset] {batch.n_episodes} episodes exceeds {_APPROXIMATE_AUTO_THRESHOLD:,}. "
            "Auto-enabling --approximate mode. Use --approximate to silence this message."
        )

    selector_cls = ApproximateCoresetSelector if use_approximate else CoresetSelector
    selector_kwargs: dict = dict(
        keep_fraction=args.keep,
        max_spike_rate=max_spike_rate,
        max_vel_disc_rate=max_vel_disc_rate,
        max_dropout_fraction=max_dropout,
        min_ldlj=args.min_ldlj,
        min_length=args.min_length,
        quality_only=args.quality_only,
        diversity_weight=diversity_weight,
        entropy_weight=entropy_weight,
        strategy=args.strategy,
    )
    if use_approximate:
        selector_kwargs["batch_size"] = args.batch_size

    selector = selector_cls(**selector_kwargs)

    result = selector.select(batch, report)

    # Write JSON output
    out_path = Path(args.out)
    out_data = result.to_dict()
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    log(f"Coreset index written to {out_path}")

    # Human-readable summary to stdout
    print(result.summary())

    if args.json:
        print(json.dumps(out_data, indent=2))

    if args.export_dataset:
        from calibra.curation.export import export_dataset
        log(f"Exporting coreset dataset to {args.export_dataset!r} ...")
        try:
            exported = export_dataset(
                result,
                dataset_path,
                args.export_dataset,
                log=log,
            )
            log(f"Dataset exported to {exported}")
            print(
                f"\n  Ready to train:\n"
                f"    python train.py --dataset {exported}"
            )
        except Exception as exc:
            print(f"error exporting dataset: {exc}", file=sys.stderr)
            sys.exit(1)
