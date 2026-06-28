"""
Real IL Benchmark: Calibra vs. Random pruning on a real robotics dataset.
=========================================================================

Trains behaviour-cloning (BC) MLP policies on three dataset versions and
measures actual policy performance — not predicted scores.

Strategies compared
-------------------
  1. Full dataset   — all training episodes
  2. Random 30%     — uniform random subset (averaged over 3 seeds)
  3. Calibra 30%    — quality-filtered + max-coverage coreset

Evaluation metrics
------------------
  • Held-out test-set MSE (action prediction error) — always available.
    Strongly predictive of rollout success for BC policies.
  • Rollout success rate — available if ``gym_pusht`` is installed:
      pip install gym-pusht

Dataset default: lerobot/pusht (HuggingFace Hub, ~200 episodes)
  Any LeRobot-format dataset (Hub ID or local path) is accepted.

Usage
-----
    python experiments/real_il_benchmark.py
    python experiments/real_il_benchmark.py --dataset lerobot/pusht --keep 0.3
    python experiments/real_il_benchmark.py --dataset /path/to/local/pusht
    python experiments/real_il_benchmark.py --n-epochs 200 --n-rollouts 100
    python experiments/real_il_benchmark.py --save-fig --json results/pusht.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Optional

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── optional gym environment ───────────────────────────────────────────────────
try:
    import gymnasium as gym
    import gym_pusht  # noqa: F401 — registers the env

    HAS_GYM_PUSHT = True
except ImportError:
    HAS_GYM_PUSHT = False


# ── data helpers ───────────────────────────────────────────────────────────────


def load_dataset(path: str) -> "EpisodeBatch":
    from calibra.ingestion.registry import load

    # Auto-detection handles both HuggingFace Hub IDs ("lerobot/pusht")
    # and local LeRobot v2 directories.
    return load(path)


def _obs_key(ep: "Episode") -> Optional[str]:
    """Return whichever observation key holds the proprioceptive state vector."""
    for k in ("state", "proprio", "joint_position", "eef_position"):
        if k in ep.observations and ep.observations[k].ndim == 2:
            return k
    return None


def train_test_split(
    batch: "EpisodeBatch", test_fraction: float = 0.20, seed: int = 0
) -> tuple["EpisodeBatch", "EpisodeBatch"]:
    """Deterministic train/test split by episode index."""
    from calibra.schema.episode import EpisodeBatch

    eps = list(batch.episodes)
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_test = max(1, round(len(eps) * test_fraction))
    test_eps = eps[:n_test]
    train_eps = eps[n_test:]
    return (
        EpisodeBatch(train_eps, batch.dataset_name + "_train", batch.format, batch.source_path),
        EpisodeBatch(test_eps, batch.dataset_name + "_test", batch.format, batch.source_path),
    )


def random_subset(
    batch: "EpisodeBatch", keep_fraction: float, seed: int = 42
) -> "EpisodeBatch":
    from calibra.schema.episode import EpisodeBatch

    eps = list(batch.episodes)
    rng = random.Random(seed)
    k = max(1, round(len(eps) * keep_fraction))
    chosen = rng.sample(eps, k)
    return EpisodeBatch(chosen, batch.dataset_name + "_random", batch.format, batch.source_path)


def calibra_subset(
    batch: "EpisodeBatch", keep_fraction: float
) -> tuple["EpisodeBatch", dict]:
    """Run Calibra pipeline + coreset selection. Returns (subset_batch, stats)."""
    from calibra.pipeline import Pipeline
    from calibra.pruning import CoresetSelector
    from calibra.schema.episode import EpisodeBatch

    print("  Running Calibra diagnostics pipeline ...", flush=True)
    t0 = time.perf_counter()
    report = Pipeline().run(batch)
    diag_s = time.perf_counter() - t0
    print(f"  Pipeline done in {diag_s:.1f}s", flush=True)

    selector = CoresetSelector(keep_fraction=keep_fraction, strategy="diversity")
    result = selector.select(batch, report)

    keep_set = set(result.keep_episode_ids)
    chosen = [ep for ep in batch.episodes if ep.metadata.episode_id in keep_set]
    subset = EpisodeBatch(
        chosen, batch.dataset_name + "_calibra", batch.format, batch.source_path
    )
    stats = {
        "n_original": result.n_original,
        "n_kept": result.n_kept,
        "n_quality_failures": result.n_quality_failures,
        "n_diversity_pruned": result.n_diversity_pruned,
        "keep_fraction_actual": result.keep_fraction_actual,
        "diag_seconds": round(diag_s, 1),
    }
    return subset, stats


# ── BC policy ──────────────────────────────────────────────────────────────────


def _collect_arrays(batch: "EpisodeBatch") -> tuple[np.ndarray, np.ndarray]:
    """Stack states and actions from all episodes into flat arrays."""
    states_all, actions_all = [], []
    for ep in batch.episodes:
        key = _obs_key(ep)
        if key is None:
            continue
        s = ep.observations[key]
        a = ep.actions
        min_len = min(len(s), len(a))
        if min_len < 2:
            continue
        states_all.append(s[:min_len])
        actions_all.append(a[:min_len])
    if not states_all:
        raise ValueError(
            "No usable observation arrays found. "
            "Expected a 'state', 'proprio', or 'joint_position' key in observations."
        )
    return np.concatenate(states_all, axis=0), np.concatenate(actions_all, axis=0)


def train_bc(
    batch: "EpisodeBatch",
    n_epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 256,
    hidden: int = 256,
) -> dict:
    """Train a BC MLP on *batch*. Returns a dict of model artifacts."""
    import torch
    import torch.nn as nn

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    S_np, A_np = _collect_arrays(batch)
    state_dim = S_np.shape[1]
    action_dim = A_np.shape[1]

    S = torch.from_numpy(S_np).float().to(device)
    A = torch.from_numpy(A_np).float().to(device)

    s_mean = S.mean(0)
    s_std = S.std(0).clamp(min=1e-6)
    a_mean = A.mean(0)
    a_std = A.std(0).clamp(min=1e-6)

    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std

    net = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, action_dim),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = len(S_n)

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            loss = ((net(S_n[idx]) - A_n[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                val_loss = ((net(S_n) - A_n) ** 2).mean().item()
            print(f"    epoch {epoch+1:>4}/{n_epochs}  train_mse={val_loss:.5f}", flush=True)

    return dict(net=net, s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std, device=device)


def eval_test_mse(artifacts: dict, test_batch: "EpisodeBatch") -> float:
    """Compute normalised action-prediction MSE on the held-out test set."""
    import torch

    net = artifacts["net"]
    s_mean = artifacts["s_mean"]
    s_std = artifacts["s_std"]
    a_mean = artifacts["a_mean"]
    a_std = artifacts["a_std"]
    device = artifacts["device"]

    S_np, A_np = _collect_arrays(test_batch)
    S = torch.from_numpy(S_np).float().to(device)
    A = torch.from_numpy(A_np).float().to(device)

    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std

    with torch.no_grad():
        pred = net(S_n)
        mse = ((pred - A_n) ** 2).mean().item()
    return mse


def eval_rollout_success(
    artifacts: dict,
    n_rollouts: int = 50,
    max_steps: int = 300,
    seed_offset: int = 0,
) -> float:
    """
    Roll out the BC policy in the PushT gym environment.
    Returns success rate (fraction of episodes that reach IoU > 0.9).

    Requires: pip install gym-pusht
    """
    import torch

    if not HAS_GYM_PUSHT:
        return float("nan")

    net = artifacts["net"]
    s_mean = artifacts["s_mean"]
    s_std = artifacts["s_std"]
    a_mean = artifacts["a_mean"]
    a_std = artifacts["a_std"]
    device = artifacts["device"]

    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    trained_dim = int(s_mean.shape[0])

    # Check that gym obs matches training obs dimensionality.
    # lerobot/pusht stores only agent_pos (2D) but gym returns full state (5D).
    # A policy trained on partial obs cannot complete the block-pushing task.
    probe_obs, _ = env.reset(seed=0)
    if probe_obs.shape[0] != trained_dim:
        env.close()
        print(
            f"  [gym] Skipping rollouts: gym obs_dim={probe_obs.shape[0]} != "
            f"training state_dim={trained_dim}. "
            f"The dataset stores partial observations (agent_pos only); "
            f"a policy trained on partial obs cannot complete block-pushing. "
            f"Use test-set MSE as the primary metric."
        )
        return float("nan")

    successes = 0
    for i in range(n_rollouts):
        obs, _ = env.reset(seed=seed_offset + i)
        for _ in range(max_steps):
            s_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            s_n = (s_t - s_mean) / s_std
            with torch.no_grad():
                a_n = net(s_n).squeeze(0)
            action = (a_n * a_std + a_mean).cpu().numpy()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get("is_success", False):
                    successes += 1
                break

    env.close()
    return successes / n_rollouts


# ── printing ───────────────────────────────────────────────────────────────────

_W = 72
_THICK = "=" * _W
_THIN = "-" * _W


def _fmt_mse(v: float) -> str:
    return f"{v:.5f}"


def _fmt_pct(v: float) -> str:
    if v != v:  # nan
        return "skipped"
    return f"{v:.1%}"


def print_results(dataset_name: str, keep: float, rows: list[dict]) -> None:
    print(f"\n{_THICK}")
    print(f"  CALIBRA REAL IL BENCHMARK - {dataset_name.upper()}")
    print(_THICK)
    print(f"  Keep fraction : {keep:.0%}")
    print(f"  Evaluation    : held-out test MSE + gym rollout success" if HAS_GYM_PUSHT else
          f"  Evaluation    : held-out test MSE (install gym-pusht for rollout success)")
    print(_THIN)
    hdr = f"  {'Strategy':<22} {'Episodes':>8}  {'Test MSE':>10}"
    if HAS_GYM_PUSHT:
        hdr += f"  {'Success':>9}"
    print(hdr)
    print(_THIN)
    for r in rows:
        line = f"  {r['label']:<22} {r['n_episodes']:>8}  {_fmt_mse(r['test_mse']):>10}"
        if HAS_GYM_PUSHT:
            line += f"  {_fmt_pct(r.get('success_rate', float('nan'))):>9}"
        print(line)
    print(_THIN)

    # Highlight Calibra vs random
    calibra = next(r for r in rows if "Calibra" in r["label"])
    randoms = [r for r in rows if "Random" in r["label"]]
    full = next(r for r in rows if r["label"] == "Full dataset")
    mean_random_mse = float(np.mean([r["test_mse"] for r in randoms]))
    mse_delta = mean_random_mse - calibra["test_mse"]
    mse_vs_full = calibra["test_mse"] - full["test_mse"]

    print(f"  MSE delta  (Calibra vs. Random):    {mse_delta:+.5f}  ({'lower=better' if mse_delta > 0 else 'Calibra worse'})")
    print(f"  MSE delta  (Calibra vs. Full  ):    {mse_vs_full:+.5f}")
    print(f"  Compute savings (vs. Full)     :    {keep:.0%} of data used")

    calibra_sr = calibra.get("success_rate", float("nan"))
    if HAS_GYM_PUSHT and calibra_sr == calibra_sr:  # not nan
        mean_random_sr = float(np.mean([r.get("success_rate", 0) for r in randoms]))
        sr_delta = calibra_sr - mean_random_sr
        print(f"  Success delta (Calibra vs. Random): {sr_delta:+.1%}")
    print(_THICK)


# ── optional figure ────────────────────────────────────────────────────────────


def save_figure(rows: list[dict], dataset_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipping figure)")
        return

    labels = [r["label"] for r in rows]
    mses = [r["test_mse"] for r in rows]
    colors = []
    for r in rows:
        if "Full" in r["label"]:
            colors.append("#6B7280")
        elif "Random" in r["label"]:
            colors.append("#EF4444")
        else:
            colors.append("#2563EB")

    fig, axes = plt.subplots(1, 2 if HAS_GYM_PUSHT else 1, figsize=(12 if HAS_GYM_PUSHT else 7, 5))
    if not HAS_GYM_PUSHT:
        axes = [axes]

    ax = axes[0]
    bars = ax.bar(range(len(labels)), mses, color=colors, width=0.5, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Test MSE (action prediction, lower=better)", fontsize=10)
    ax.set_title(f"Calibra vs. Baselines — {dataset_name}\nHeld-out Test MSE", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{mse:.4f}", ha="center", va="bottom", fontsize=8)

    if HAS_GYM_PUSHT:
        ax2 = axes[1]
        success_rates = [r.get("success_rate", 0) * 100 for r in rows]
        bars2 = ax2.bar(range(len(labels)), success_rates, color=colors, width=0.5, zorder=3)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax2.set_ylabel("Rollout success rate (%)", fontsize=10)
        ax2.set_title(f"Calibra vs. Baselines — {dataset_name}\nPushT Rollout Success", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3, zorder=0)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for bar, sr in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{sr:.1f}%", ha="center", va="bottom", fontsize=8)

    # Legend patch
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="#6B7280", label="Full dataset"),
        mpatches.Patch(color="#EF4444", label="Random pruned"),
        mpatches.Patch(color="#2563EB", label="Calibra coreset"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_pdf = FIG_DIR / "fig_real_il_benchmark.pdf"
    out_png = FIG_DIR / "fig_real_il_benchmark.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"  Figure saved: {out_pdf}")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="real_il_benchmark",
        description="Calibra real IL training benchmark on a LeRobot dataset.",
    )
    p.add_argument(
        "--dataset",
        default="lerobot/pusht",
        help="LeRobot Hub ID or local path (default: lerobot/pusht)",
    )
    p.add_argument(
        "--keep", "-k", type=float, default=0.30,
        help="Fraction of training episodes to retain in pruned coresets (default: 0.30)",
    )
    p.add_argument(
        "--test-fraction", type=float, default=0.20,
        help="Fraction of dataset held out for evaluation (default: 0.20)",
    )
    p.add_argument(
        "--n-epochs", type=int, default=150,
        help="BC training epochs per policy (default: 150)",
    )
    p.add_argument(
        "--n-rollouts", type=int, default=50,
        help="Gym rollout episodes per strategy when gym_pusht is available (default: 50)",
    )
    p.add_argument(
        "--random-seeds", type=int, default=3,
        help="Number of random pruning seeds to average (default: 3)",
    )
    p.add_argument(
        "--save-fig", action="store_true",
        help="Save comparison bar charts to experiments/figures/",
    )
    p.add_argument(
        "--json", metavar="PATH",
        help="Write raw results dict to a JSON file",
    )
    args = p.parse_args(argv)

    print("=" * _W)
    print("  CALIBRA REAL IL BENCHMARK")
    print("=" * _W)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Keep frac : {args.keep:.0%}")
    print(f"  BC epochs : {args.n_epochs}")
    print(f"  gym_pusht : {'available' if HAS_GYM_PUSHT else 'not installed (rollouts skipped)'}")
    print()

    # 1. Load dataset
    print(f"[1/6] Loading dataset '{args.dataset}' ...")
    t0 = time.perf_counter()
    batch = load_dataset(args.dataset)
    print(f"      {batch.n_episodes} episodes, {batch.n_samples} steps  ({time.perf_counter()-t0:.1f}s)")

    if batch.n_episodes < 10:
        sys.exit(f"Error: dataset has only {batch.n_episodes} episodes -- need >= 10.")

    # Detect obs key
    sample_ep = batch.episodes[0]
    obs_key = _obs_key(sample_ep)
    if obs_key is None:
        sys.exit(
            f"Error: no proprioceptive observation key found.\n"
            f"Available keys: {list(sample_ep.observations.keys())}"
        )
    state_dim = sample_ep.observations[obs_key].shape[1]
    action_dim = sample_ep.actions.shape[1]
    print(f"      obs_key={obs_key!r}  state_dim={state_dim}  action_dim={action_dim}")

    # 2. Train/test split
    print(f"\n[2/6] Splitting dataset (train {1-args.test_fraction:.0%} / test {args.test_fraction:.0%}) ...")
    train_batch, test_batch = train_test_split(batch, test_fraction=args.test_fraction)
    print(f"      train={train_batch.n_episodes} eps  test={test_batch.n_episodes} eps")

    # 3. Calibra coreset
    print(f"\n[3/6] Building Calibra coreset (keep={args.keep:.0%}) ...")
    t0 = time.perf_counter()
    calibra_batch, calibra_stats = calibra_subset(train_batch, keep_fraction=args.keep)
    print(f"      Coreset: {calibra_batch.n_episodes} eps  "
          f"(quality fails={calibra_stats['n_quality_failures']}, "
          f"diversity pruned={calibra_stats['n_diversity_pruned']})  "
          f"[{time.perf_counter()-t0:.1f}s total]")

    # 4. Train BC policies
    rows: list[dict] = []

    print(f"\n[4/6] Training BC on FULL dataset ({train_batch.n_episodes} eps) ...")
    t0 = time.perf_counter()
    full_artifacts = train_bc(train_batch, n_epochs=args.n_epochs)
    full_mse = eval_test_mse(full_artifacts, test_batch)
    full_success = eval_rollout_success(full_artifacts, n_rollouts=args.n_rollouts, seed_offset=0)
    rows.append({
        "label": "Full dataset",
        "n_episodes": train_batch.n_episodes,
        "test_mse": full_mse,
        "success_rate": full_success,
        "train_seconds": round(time.perf_counter() - t0, 1),
    })
    print(f"      test_mse={full_mse:.5f}" +
          (f"  success={full_success:.1%}" if HAS_GYM_PUSHT else ""))

    print(f"\n[5/6] Training BC on RANDOM subsets ({args.random_seeds} seeds) ...")
    random_seed_rows = []
    for seed in range(args.random_seeds):
        rnd_batch = random_subset(train_batch, keep_fraction=args.keep, seed=seed * 17 + 42)
        t0 = time.perf_counter()
        rnd_artifacts = train_bc(rnd_batch, n_epochs=args.n_epochs)
        rnd_mse = eval_test_mse(rnd_artifacts, test_batch)
        rnd_success = eval_rollout_success(rnd_artifacts, n_rollouts=args.n_rollouts,
                                           seed_offset=100 + seed * 50)
        random_seed_rows.append({
            "label": f"Random (seed={seed})",
            "n_episodes": rnd_batch.n_episodes,
            "test_mse": rnd_mse,
            "success_rate": rnd_success,
            "train_seconds": round(time.perf_counter() - t0, 1),
        })
        print(f"      seed={seed}  n={rnd_batch.n_episodes}  "
              f"test_mse={rnd_mse:.5f}" +
              (f"  success={rnd_success:.1%}" if HAS_GYM_PUSHT else ""))

    rows.extend(random_seed_rows)

    print(f"\n[6/6] Training BC on CALIBRA coreset ({calibra_batch.n_episodes} eps) ...")
    t0 = time.perf_counter()
    calibra_artifacts = train_bc(calibra_batch, n_epochs=args.n_epochs)
    calibra_mse = eval_test_mse(calibra_artifacts, test_batch)
    calibra_success = eval_rollout_success(calibra_artifacts, n_rollouts=args.n_rollouts,
                                           seed_offset=200)
    rows.append({
        "label": "Calibra coreset",
        "n_episodes": calibra_batch.n_episodes,
        "test_mse": calibra_mse,
        "success_rate": calibra_success,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "calibra_stats": calibra_stats,
    })
    print(f"      n={calibra_batch.n_episodes}  test_mse={calibra_mse:.5f}" +
          (f"  success={calibra_success:.1%}" if HAS_GYM_PUSHT else ""))

    # 5. Summary
    print_results(
        dataset_name=batch.dataset_name,
        keep=args.keep,
        rows=rows,
    )

    # 6. Spearman correlation: Calibra quality signal vs. test MSE
    #    (across the 3 strategies — full, random, calibra — ranked by n_episodes and mse)
    all_mses = [full_mse] + [r["test_mse"] for r in random_seed_rows] + [calibra_mse]
    all_labels = ["Full"] + [f"Rand-{i}" for i in range(len(random_seed_rows))] + ["Calibra"]
    if len(all_mses) >= 3:
        data_fracs = [1.0] + [args.keep] * len(random_seed_rows) + [calibra_stats["keep_fraction_actual"]]
        x = np.argsort(np.argsort(data_fracs))
        y = np.argsort(np.argsort(all_mses))
        rho = float(np.corrcoef(x, y)[0, 1])
        print(f"\n  Spearman rho (data fraction vs. test MSE, higher fraction -> lower MSE): {rho:+.3f}")
        print("  (rho close to 1 means more data = lower error; Calibra ideally breaks this by")
        print("   achieving low MSE with a small fraction -- shown in the table above.)")

    # 7. Save JSON
    output = {
        "dataset": args.dataset,
        "keep_fraction": args.keep,
        "n_epochs": args.n_epochs,
        "n_rollouts": args.n_rollouts if HAS_GYM_PUSHT else 0,
        "gym_pusht_available": HAS_GYM_PUSHT,
        "train_episodes": train_batch.n_episodes,
        "test_episodes": test_batch.n_episodes,
        "results": [
            {k: v for k, v in r.items() if k not in ("calibra_stats",)}
            for r in rows
        ],
        "calibra_diagnostics": calibra_stats,
    }

    if args.json:
        out_path = pathlib.Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    if args.save_fig:
        save_figure(rows, batch.dataset_name)

    return output


if __name__ == "__main__":
    main()
