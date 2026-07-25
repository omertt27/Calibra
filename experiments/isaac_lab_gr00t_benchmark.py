"""
Isaac Lab → GR00T benchmark: Calibra coreset selection on synthetic 7-DOF arm demos.

Validates that the Calibra pipeline (quality filter + diversity selection) produces
a coreset that trains better BC policies than random subsampling, on 7-DOF arm
kinematic data matching Isaac Lab's HDF5 demo format.

Pipeline
--------
    1. Generate synthetic Isaac Lab-style HDF5 demos (7-DOF arm, varied quality)
    2. Run Calibra prune with --policy gr00t (applies GR00T quality thresholds)
    3. Train a BC MLP on: (a) Calibra coreset, (b) random subset, (c) full dataset
    4. Evaluate on held-out test demos via trajectory-level MSE
    5. Export GR00T manifest and plot results

Usage
-----
    python experiments/isaac_lab_gr00t_benchmark.py
    python experiments/isaac_lab_gr00t_benchmark.py --n-demos 300 --keep 0.3 0.5
    python experiments/isaac_lab_gr00t_benchmark.py --no-plots --report results.json

Requirements
------------
    pip install calibra-robotics numpy torch matplotlib h5py

Outputs
-------
    experiments/figures/isaac_lab_mse_vs_keep.pdf
    experiments/figures/isaac_lab_mse_vs_keep.png
    experiments/figures/isaac_lab_results.json
    experiments/figures/gr00t_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

FIGURES_DIR = Path(__file__).parent / "figures"

# ── synthetic data generation ─────────────────────────────────────────────────

DOF = 7  # 7-DOF arm (Franka Emika / similar)
OBS_DIM = 14  # joint positions + joint velocities
ACTION_DIM = 7  # joint position targets
DT = 0.02  # 50 Hz control


def _smooth_trajectory(n_steps: int, dof: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a smooth joint trajectory via Gaussian Process-like interpolation."""
    n_waypoints = max(3, n_steps // 15)
    waypoints = rng.uniform(-1.5, 1.5, size=(n_waypoints, dof))
    # Cubic interpolation between waypoints
    xs = np.linspace(0, 1, n_waypoints)
    ts = np.linspace(0, 1, n_steps)
    traj = np.zeros((n_steps, dof))
    for d in range(dof):
        traj[:, d] = np.interp(ts, xs, waypoints[:, d])
    return traj


def generate_demos(
    n_demos: int,
    rng: Optional[np.random.Generator] = None,
    hdf5_path: Optional[Path] = None,
) -> tuple[list[dict], Optional[Path]]:
    """
    Generate synthetic Isaac Lab HDF5-style demos with varied quality.

    About 20% of demos are injected with quality problems:
      - short episodes (< 20 steps)
      - jerk spikes (sudden waypoint jumps)
      - velocity discontinuities
      - timestamp dropouts

    Parameters
    ----------
    n_demos    : total number of demos to generate
    rng        : random generator (seeded for reproducibility)
    hdf5_path  : if provided, also writes an HDF5 file in Isaac Lab format

    Returns
    -------
    (demos_list, hdf5_path_or_None)
    where demos_list is a list of {obs, actions, timestamps} dicts.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    demos = []
    for i in range(n_demos):
        quality = rng.random()
        if quality < 0.08:
            n_steps = rng.integers(5, 15)  # too short
            problem = "short"
        elif quality < 0.15:
            n_steps = rng.integers(60, 150)
            problem = "jerk"
        elif quality < 0.20:
            n_steps = rng.integers(60, 150)
            problem = "disc"
        elif quality < 0.25:
            n_steps = rng.integers(60, 150)
            problem = "dropout"
        else:
            n_steps = rng.integers(80, 200)
            problem = None

        timestamps = np.arange(n_steps) * DT

        if problem == "short":
            actions = rng.uniform(-1, 1, (n_steps, ACTION_DIM))
        elif problem == "jerk":
            actions = _smooth_trajectory(n_steps, ACTION_DIM, rng)
            spike_steps = rng.choice(n_steps, size=n_steps // 5, replace=False)
            actions[spike_steps] += rng.uniform(3, 6, size=(len(spike_steps), ACTION_DIM))
        elif problem == "disc":
            actions = _smooth_trajectory(n_steps, ACTION_DIM, rng)
            disc_steps = rng.choice(n_steps - 1, size=n_steps // 8, replace=False)
            actions[disc_steps + 1] = rng.uniform(-3, 3, size=(len(disc_steps), ACTION_DIM))
        elif problem == "dropout":
            actions = _smooth_trajectory(n_steps, ACTION_DIM, rng)
            drop_mask = rng.random(n_steps) > 0.85
            timestamps = timestamps.copy()
            timestamps[drop_mask] = np.nan
        else:
            actions = _smooth_trajectory(n_steps, ACTION_DIM, rng)

        # obs = [joint_pos, joint_vel]
        vel = np.gradient(actions, DT, axis=0)
        obs = np.concatenate([actions, vel], axis=-1)

        demos.append(
            {
                "episode_id": str(i),
                "obs": obs,
                "actions": actions,
                "timestamps": np.nan_to_num(timestamps),
            }
        )

    if hdf5_path is not None:
        _write_hdf5(demos, hdf5_path)

    return demos, hdf5_path


def _write_hdf5(demos: list[dict], path: Path) -> None:
    """Write demos in Isaac Lab HDF5 format."""
    try:
        import h5py
    except ImportError:
        print("[warn] h5py not installed — skipping HDF5 export", file=sys.stderr)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["n_demos"] = len(demos)
        f.attrs["format"] = "isaac_lab"
        data_grp = f.require_group("data")
        for demo in demos:
            grp = data_grp.require_group(f"demo_{demo['episode_id']}")
            obs_grp = grp.require_group("obs")
            obs_grp.create_dataset("robot0_joint_pos", data=demo["obs"][:, :DOF])
            obs_grp.create_dataset("robot0_joint_vel", data=demo["obs"][:, DOF:])
            grp.create_dataset("actions", data=demo["actions"])
            grp.create_dataset("timestamps", data=demo["timestamps"])


# ── Calibra EpisodeBatch conversion ──────────────────────────────────────────


def demos_to_episode_batch(demos: list[dict], dataset_name: str):
    """Convert generated demos to a Calibra EpisodeBatch."""
    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    episodes = []
    for demo in demos:
        meta = EpisodeMetadata(
            episode_id=demo["episode_id"],
            task_name="isaac_lab_arm",
            extra={},
        )
        ep = Episode(
            metadata=meta,
            observations=demo["obs"],
            actions=demo["actions"],
            timestamps=demo["timestamps"],
        )
        episodes.append(ep)

    return EpisodeBatch(
        dataset_name=dataset_name,
        source_path=dataset_name,
        format="hdf5",
        episodes=episodes,
    )


# ── Calibra pipeline + pruning ────────────────────────────────────────────────


def run_calibra(
    batch,
    keep_fraction: float,
    report_path: Optional[Path] = None,
    cache=None,
) -> tuple:
    """
    Run Calibra pipeline + GR00T-tuned coreset selection.

    Returns (pruning_result, keep_episode_ids).
    """
    from calibra.cache import batch_episode_hashes
    from calibra.pipeline import Pipeline
    from calibra.pruning import CoresetSelector

    diag = Pipeline().run(batch, policy_family="gr00t", cache=cache)

    selector = CoresetSelector(
        keep_fraction=keep_fraction,
        max_spike_rate=0.05,
        max_vel_disc_rate=0.10,
        max_dropout_fraction=0.05,
        diversity_weight=0.7,
        entropy_weight=0.4,
        strategy="diversity",
    )
    result = selector.select(batch, diag)

    if report_path is not None:
        from calibra.report_json import assemble_public_report, dataset_info_from_report

        ds_info = dataset_info_from_report(diag)
        ep_hashes = batch_episode_hashes(batch)
        public = assemble_public_report(
            diag,
            dataset_info=ds_info,
            pruning_result=result,
            episode_hashes=ep_hashes,
        )
        public.write(str(report_path))

    return result, result.keep_episode_ids


def random_subset(demos: list[dict], keep_fraction: float, seed: int) -> list[str]:
    """Select a random subset of episode IDs."""
    rng = np.random.default_rng(seed)
    n = max(1, int(len(demos) * keep_fraction))
    indices = rng.choice(len(demos), size=n, replace=False)
    return [demos[i]["episode_id"] for i in sorted(indices)]


# ── BC MLP training ───────────────────────────────────────────────────────────


def _build_model(hidden: int = 256):
    """Build a simple BC MLP: obs → action prediction."""
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for BC training.\nInstall it with: pip install torch"
        )
    return nn.Sequential(
        nn.Linear(OBS_DIM, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, ACTION_DIM),
    )


def _demos_to_tensors(demos_subset: list[dict]):
    """Stack demos into flat (obs, action) arrays."""
    obs_all = np.concatenate([d["obs"] for d in demos_subset], axis=0)
    act_all = np.concatenate([d["actions"] for d in demos_subset], axis=0)
    return obs_all.astype(np.float32), act_all.astype(np.float32)


def train_bc(
    demos: list[dict],
    keep_ids: list[str],
    n_epochs: int = 30,
    lr: float = 3e-4,
    hidden: int = 256,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    """
    Train a BC MLP on the selected demos subset.

    Returns {'model': nn.Sequential, 'train_loss': float}.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("pip install torch")

    torch.manual_seed(seed)
    keep_set = set(keep_ids)
    subset = [d for d in demos if d["episode_id"] in keep_set]
    if not subset:
        raise ValueError("No demos selected.")

    obs_arr, act_arr = _demos_to_tensors(subset)
    obs_t = torch.from_numpy(obs_arr)
    act_t = torch.from_numpy(act_arr)
    n = len(obs_t)

    model = _build_model(hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    model.train()
    train_loss = float("inf")
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(obs_t[idx])
            loss = nn.functional.mse_loss(pred, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss = epoch_loss / max(n_batches, 1)

    return {"model": model, "train_loss": train_loss}


def evaluate_bc(model_dict: dict, test_demos: list[dict]) -> float:
    """
    Evaluate a trained BC model on held-out test demos.

    Returns mean trajectory MSE across test demos.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("pip install torch")

    model = model_dict["model"]
    model.eval()
    mses = []
    with torch.no_grad():
        for demo in test_demos:
            obs_t = torch.from_numpy(demo["obs"].astype(np.float32))
            act_t = torch.from_numpy(demo["actions"].astype(np.float32))
            pred = model(obs_t)
            mse = nn.functional.mse_loss(pred, act_t).item()
            mses.append(mse)
    return float(np.mean(mses)) if mses else float("nan")


# ── benchmark sweep ───────────────────────────────────────────────────────────


def run_benchmark(
    n_demos: int = 200,
    keep_fractions: list[float] = None,
    n_epochs: int = 30,
    n_seeds: int = 3,
    test_fraction: float = 0.2,
    report_dir: Path = FIGURES_DIR,
    no_plots: bool = False,
) -> dict:
    """
    Full benchmark sweep: Calibra coreset vs. random vs. full at each keep fraction.

    Returns results dict with MSEs for all conditions.
    """
    if keep_fractions is None:
        keep_fractions = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]

    print(f"\n{'━' * 60}")
    print("  Isaac Lab → GR00T Coreset Benchmark")
    print(f"{'━' * 60}")
    print(f"  Demos       : {n_demos}")
    print(f"  Keep %      : {[f'{k:.0%}' for k in keep_fractions]}")
    print(f"  Seeds       : {n_seeds}")
    print(f"  Test frac   : {test_fraction:.0%}")
    print(f"{'━' * 60}\n")

    rng = np.random.default_rng(42)
    all_demos, _ = generate_demos(n_demos, rng=rng)

    # Train/test split by demo index (deterministic)
    n_test = max(1, int(n_demos * test_fraction))
    test_indices = set(range(n_demos - n_test, n_demos))
    train_demos = [d for d in all_demos if int(d["episode_id"]) not in test_indices]
    test_demos = [d for d in all_demos if int(d["episode_id"]) in test_indices]

    print(f"  Train demos : {len(train_demos)}")
    print(f"  Test demos  : {len(test_demos)}\n")

    # Convert to EpisodeBatch for Calibra
    train_batch = demos_to_episode_batch(train_demos, "isaac_lab_arm_train")

    results: dict = {
        "n_demos": n_demos,
        "n_train": len(train_demos),
        "n_test": n_test,
        "keep_fractions": keep_fractions,
        "calibra": [],
        "random": [],
        "full": [],
    }

    # Full-dataset baseline (train once, evaluate)
    print("Training full-dataset baseline...")
    full_ids = [d["episode_id"] for d in train_demos]
    full_mses = []
    for seed in range(n_seeds):
        m = train_bc(train_demos, full_ids, n_epochs=n_epochs, seed=seed)
        mse = evaluate_bc(m, test_demos)
        full_mses.append(mse)
    full_mean = float(np.mean(full_mses))
    full_std = float(np.std(full_mses))
    results["full"] = {"mean": full_mean, "std": full_std, "n_demos": len(train_demos)}
    print(f"  Full dataset  MSE = {full_mean:.4f} ± {full_std:.4f}  ({len(train_demos)} demos)")

    for keep in keep_fractions:
        if keep >= 1.0:
            calibra_mse_list = full_mses.copy()
            random_mse_list = full_mses.copy()
        else:
            # ── Calibra coreset ───────────────────────────────────────────────
            t0 = time.perf_counter()
            report_path = report_dir / f"isaac_lab_report_keep{int(keep * 100):03d}.json"
            report_dir.mkdir(parents=True, exist_ok=True)
            result, calibra_ids = run_calibra(
                train_batch,
                keep_fraction=keep,
                report_path=report_path,
            )
            calibra_time = time.perf_counter() - t0
            n_calibra = len(calibra_ids)

            calibra_mse_list = []
            for seed in range(n_seeds):
                m = train_bc(train_demos, calibra_ids, n_epochs=n_epochs, seed=seed)
                mse = evaluate_bc(m, test_demos)
                calibra_mse_list.append(mse)

            # ── Random baseline ───────────────────────────────────────────────
            random_mse_list = []
            for seed in range(n_seeds):
                rand_ids = random_subset(train_demos, keep, seed=seed + 100)
                m = train_bc(train_demos, rand_ids, n_epochs=n_epochs, seed=seed)
                mse = evaluate_bc(m, test_demos)
                random_mse_list.append(mse)

            n_random = max(1, int(len(train_demos) * keep))
            print(
                f"  keep={keep:.0%}  "
                f"Calibra {np.mean(calibra_mse_list):.4f}±{np.std(calibra_mse_list):.4f} "
                f"(n={n_calibra}, Δt={calibra_time:.1f}s)  |  "
                f"Random  {np.mean(random_mse_list):.4f}±{np.std(random_mse_list):.4f} "
                f"(n={n_random})"
            )

            # Export GR00T manifest for the 30% keep run
            if abs(keep - 0.30) < 0.01 and report_path.exists():
                from calibra.integrations.isaac_lab import export_gr00t_manifest

                manifest_path = report_dir / "gr00t_manifest.json"
                export_gr00t_manifest(
                    report_path=report_path,
                    demos_path="demos/isaac_lab_arm.hdf5",
                    out_path=manifest_path,
                )
                print(f"    GR00T manifest → {manifest_path}")

        results["calibra"].append(
            {
                "keep": keep,
                "mean": float(np.mean(calibra_mse_list)),
                "std": float(np.std(calibra_mse_list)),
                "n_demos": len(calibra_ids) if keep < 1.0 else len(train_demos),
            }
        )
        results["random"].append(
            {
                "keep": keep,
                "mean": float(np.mean(random_mse_list)),
                "std": float(np.std(random_mse_list)),
                "n_demos": max(1, int(len(train_demos) * keep)),
            }
        )

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'━' * 60}")
    for i, keep in enumerate(keep_fractions):
        c = results["calibra"][i]
        r = results["random"][i]
        delta = r["mean"] - c["mean"]
        pct = (delta / r["mean"]) * 100 if r["mean"] > 0 else 0
        sign = "+" if delta >= 0 else ""
        print(
            f"  keep={keep:.0%}  Calibra {c['mean']:.4f}  Random {r['mean']:.4f}  "
            f"→ Calibra {sign}{pct:.1f}% better MSE"
        )
    print(f"{'━' * 60}\n")

    # Save results JSON
    out_json = report_dir / "isaac_lab_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results → {out_json}")

    if not no_plots:
        _plot_results(results, report_dir)

    return results


# ── plotting ──────────────────────────────────────────────────────────────────


def _plot_results(results: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except ImportError:
        print("[info] matplotlib not installed — skipping plots")
        return

    keeps = results["keep_fractions"]
    c_means = [r["mean"] for r in results["calibra"]]
    c_stds = [r["std"] for r in results["calibra"]]
    r_means = [r["mean"] for r in results["random"]]
    r_stds = [r["std"] for r in results["random"]]
    full_mean = results["full"]["mean"]

    fig, ax = plt.subplots(figsize=(8, 5))

    xs = [k * 100 for k in keeps]
    ax.plot(xs, c_means, "o-", color="#2563EB", label="Calibra coreset", linewidth=2)
    ax.fill_between(
        xs,
        [m - s for m, s in zip(c_means, c_stds)],
        [m + s for m, s in zip(c_means, c_stds)],
        alpha=0.2,
        color="#2563EB",
    )
    ax.plot(xs, r_means, "s--", color="#DC2626", label="Random subset", linewidth=2)
    ax.fill_between(
        xs,
        [m - s for m, s in zip(r_means, r_stds)],
        [m + s for m, s in zip(r_means, r_stds)],
        alpha=0.2,
        color="#DC2626",
    )
    ax.axhline(
        full_mean,
        color="#16A34A",
        linestyle=":",
        linewidth=1.5,
        label=f"Full dataset  MSE={full_mean:.4f}",
    )

    ax.set_xlabel("Keep fraction (%)", fontsize=12)
    ax.set_ylabel("Trajectory MSE (lower is better)", fontsize=12)
    ax.set_title(
        "Isaac Lab → GR00T  |  Calibra Coreset vs. Random Subsampling\n"
        "7-DOF Arm, BC-MLP, Held-out Test MSE",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig_path = out_dir / f"isaac_lab_mse_vs_keep.{ext}"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure → {fig_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Isaac Lab → GR00T benchmark: Calibra coreset vs. random subsampling",
    )
    p.add_argument(
        "--n-demos",
        type=int,
        default=200,
        help="Number of synthetic demos to generate (default: 200)",
    )
    p.add_argument(
        "--keep",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.50, 0.70, 1.00],
        help="Keep fractions to sweep (default: 0.10 0.20 0.30 0.50 0.70 1.00)",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=30,
        help="BC training epochs per run (default: 30)",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Random seeds for averaging (default: 3)",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of demos to hold out for evaluation (default: 0.2)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figure generation",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES_DIR,
        help=f"Directory for results and figures (default: {FIGURES_DIR})",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "error: PyTorch is required.\nInstall: pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)

    run_benchmark(
        n_demos=args.n_demos,
        keep_fractions=sorted(set(args.keep)),
        n_epochs=args.n_epochs,
        n_seeds=args.n_seeds,
        test_fraction=args.test_fraction,
        report_dir=args.out_dir,
        no_plots=args.no_plots,
    )
