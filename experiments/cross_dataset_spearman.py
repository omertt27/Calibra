"""
Cross-dataset Spearman Correlation Benchmark
=============================================
Tests whether Calibra's offline quality score predicts downstream BC-MLP
performance across multiple datasets.

For each dataset:
  1. Load demonstrations (LeRobot Hub ID or local path).
  2. Run Calibra pipeline → get per-dataset quality score.
  3. Train BC-MLP on three conditions (multiple seeds for variance):
       - Full dataset
       - Calibra 30% coreset
       - Random 30% baseline (n_seeds)
  4. Record test-MSE per condition.

Across datasets compute:
  • Spearman ρ: Calibra quality score vs. full-dataset test MSE
  • Spearman ρ: Calibra improvement margin vs. random-baseline test MSE

This gives an end-to-end empirical check that is fully reproducible
on CPU/MPS (M2 Pro), at the cost of wall-clock time.

Usage
-----
    # Quick smoke test (2 datasets, 3 seeds, 100 epochs)
    PYTHONPATH=. python experiments/cross_dataset_spearman.py \\
        --datasets lerobot/aloha_mobile_cabinet lerobot/droid_100 \\
        --seeds 3 --n-epochs 100

    # Full run (default datasets, 7 seeds, 300 epochs)
    PYTHONPATH=. python experiments/cross_dataset_spearman.py --seeds 7

    # Load from pre-computed JSON results
    PYTHONPATH=. python experiments/cross_dataset_spearman.py \\
        --from-json results/ablation_aloha.json results/ablation_droid.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATASETS = [
    "lerobot/aloha_mobile_cabinet",
    "lerobot/droid_100",
    "lerobot/pusht",
]

_W = 90


# ── reuse helpers from ablation_benchmark ────────────────────────────────────

def _load(path: str):
    from calibra.ingestion.registry import load
    return load(path)


def _obs_key(ep):
    for k in ("state", "proprio", "joint_position", "eef_position"):
        if k in ep.observations and ep.observations[k].ndim == 2:
            return k
    return None


def _split(batch, test_fraction: float = 0.20, seed: int = 0):
    from calibra.schema.episode import EpisodeBatch
    eps = list(batch.episodes)
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_test = max(1, round(len(eps) * test_fraction))
    test_eps, train_eps = eps[:n_test], eps[n_test:]
    return (
        EpisodeBatch(train_eps, batch.dataset_name + "_train", batch.format, batch.source_path),
        EpisodeBatch(test_eps,  batch.dataset_name + "_test",  batch.format, batch.source_path),
    )


def _collect(batch):
    S, A = [], []
    for ep in batch.episodes:
        key = _obs_key(ep)
        if key is None:
            continue
        s, a = ep.observations[key], ep.actions
        ml = min(len(s), len(a))
        if ml < 2:
            continue
        S.append(s[:ml])
        A.append(a[:ml])
    if not S:
        raise ValueError("No usable observations found in batch.")
    return np.concatenate(S), np.concatenate(A)


def _train_bc(batch, n_epochs: int = 200, lr: float = 1e-3,
              batch_size: int = 256, hidden: int = 256, seed: int = 0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    S_np, A_np = _collect(batch)
    S = torch.from_numpy(S_np).float().to(device)
    A = torch.from_numpy(A_np).float().to(device)
    s_mean, s_std = S.mean(0), S.std(0).clamp(min=1e-6)
    a_mean, a_std = A.mean(0), A.std(0).clamp(min=1e-6)
    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std

    state_dim, action_dim = S_np.shape[1], A_np.shape[1]
    net = nn.Sequential(
        nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, action_dim),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = len(S_n)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device, generator=rng)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            loss = ((net(S_n[idx]) - A_n[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    return dict(net=net, s_mean=s_mean, s_std=s_std,
                a_mean=a_mean, a_std=a_std, device=device)


def _eval_mse(artifacts, test_batch) -> float:
    import torch
    net = artifacts["net"]
    s_mean, s_std = artifacts["s_mean"], artifacts["s_std"]
    a_mean, a_std = artifacts["a_mean"], artifacts["a_std"]
    device = artifacts["device"]

    S_np, A_np = _collect(test_batch)
    S = torch.from_numpy(S_np).float().to(device)
    A = torch.from_numpy(A_np).float().to(device)
    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std
    with torch.no_grad():
        return ((net(S_n) - A_n) ** 2).mean().item()


def _random_subset(batch, k: int, seed: int = 42):
    from calibra.schema.episode import EpisodeBatch
    eps = list(batch.episodes)
    rng = random.Random(seed)
    chosen = rng.sample(eps, min(k, len(eps)))
    return EpisodeBatch(chosen, batch.dataset_name + "_rnd", batch.format, batch.source_path)


def _calibra_coreset(batch, report, keep_fraction: float):
    from calibra.pruning import CoresetSelector
    from calibra.schema.episode import EpisodeBatch
    sel = CoresetSelector(keep_fraction=keep_fraction, strategy="diversity")
    result = sel.select(batch, report)
    keep_set = set(result.keep_episode_ids)
    chosen = [ep for ep in batch.episodes if ep.metadata.episode_id in keep_set]
    return EpisodeBatch(chosen, batch.dataset_name + "_cal", batch.format, batch.source_path)


def _calibra_quality_score(batch) -> tuple[object, float]:
    from calibra.pipeline import Pipeline
    from calibra.predict import predict_outcome

    report = Pipeline().run(batch)
    obs_key = _obs_key(batch.episodes[0]) if batch.episodes else None
    action_dim = batch.episodes[0].actions.shape[1] if batch.episodes else 6
    policy_family = "act" if action_dim >= 6 else "diffusion"
    pred = predict_outcome(report, policy_family=policy_family, use_outcome_db=False)
    return report, float(pred["predicted_score"])


# ── per-dataset run ───────────────────────────────────────────────────────────

def run_dataset(
    dataset_path: str,
    n_seeds: int = 5,
    n_epochs: int = 200,
    keep_fraction: float = 0.30,
) -> dict:
    """
    For one dataset: run Calibra + BC-MLP across n_seeds train/test splits.

    Returns a dict with:
      dataset, calibra_score, full_mse_mean, full_mse_std,
      calibra_mse_mean, calibra_mse_std,
      random_mse_mean, random_mse_std,
      calibra_vs_random_pct, n_episodes
    """
    print(f"\n{'─'*_W}")
    print(f"  Dataset: {dataset_path}")
    print(f"{'─'*_W}")

    t_load = time.perf_counter()
    batch = _load(dataset_path)
    print(f"  Loaded {batch.n_episodes} episodes in {time.perf_counter()-t_load:.1f}s")

    # Calibra quality score on full dataset (deterministic)
    print("  Running Calibra pipeline ...", flush=True)
    t_cal = time.perf_counter()
    report, cal_score = _calibra_quality_score(batch)
    print(f"  Calibra quality score: {cal_score:.1f}  ({time.perf_counter()-t_cal:.1f}s)")

    full_mses, cal_mses, rnd_mses = [], [], []

    for seed in range(n_seeds):
        print(f"  [seed {seed+1}/{n_seeds}] splitting & training ...", flush=True)
        train_batch, test_batch = _split(batch, seed=seed)
        k = max(1, round(len(train_batch.episodes) * keep_fraction))

        # Full dataset
        art_full = _train_bc(train_batch, n_epochs=n_epochs, seed=seed)
        full_mses.append(_eval_mse(art_full, test_batch))

        # Calibra coreset (coreset re-selected per split so episodes are valid)
        cal_sub = _calibra_coreset(train_batch, report, keep_fraction)
        art_cal = _train_bc(cal_sub, n_epochs=n_epochs, seed=seed)
        cal_mses.append(_eval_mse(art_cal, test_batch))

        # Random baseline
        rnd_sub = _random_subset(train_batch, k, seed=seed * 17 + 42)
        art_rnd = _train_bc(rnd_sub, n_epochs=n_epochs, seed=seed)
        rnd_mses.append(_eval_mse(art_rnd, test_batch))

        print(f"    full={full_mses[-1]:.5f}  calibra={cal_mses[-1]:.5f}  random={rnd_mses[-1]:.5f}")

    full_mse_mean  = float(np.mean(full_mses))
    cal_mse_mean   = float(np.mean(cal_mses))
    rnd_mse_mean   = float(np.mean(rnd_mses))
    cal_vs_rnd_pct = (rnd_mse_mean - cal_mse_mean) / rnd_mse_mean * 100 if rnd_mse_mean else 0.0

    return {
        "dataset": dataset_path,
        "n_episodes": batch.n_episodes,
        "calibra_score": cal_score,
        "full_mse_mean":    full_mse_mean,
        "full_mse_std":     float(np.std(full_mses)),
        "calibra_mse_mean": cal_mse_mean,
        "calibra_mse_std":  float(np.std(cal_mses)),
        "random_mse_mean":  rnd_mse_mean,
        "random_mse_std":   float(np.std(rnd_mses)),
        "calibra_vs_random_pct": cal_vs_rnd_pct,
        "n_seeds": n_seeds,
    }


# ── ingest pre-computed ablation JSON ────────────────────────────────────────

def ingest_from_json(json_paths: list[str]) -> list[dict]:
    """
    Re-use results already produced by ablation_benchmark.py.

    Extracts calibra_score via Calibra pipeline (fast, no training needed)
    from the dataset listed in each JSON.
    """
    rows = []
    for p in json_paths:
        data = json.loads(pathlib.Path(p).read_text())
        dataset_path = data["dataset"]

        abl = {r["condition"]: r for r in data.get("ablation", [])}
        rnd_row = next((r for r in abl.values() if "Random" in r["condition"]), None)
        cal_row = abl.get("Calibra full")
        if not rnd_row or not cal_row:
            print(f"  [skip] {p}: ablation data incomplete")
            continue

        rnd_mse = rnd_row["test_mse"]
        cal_mse = cal_row["test_mse"]
        full_row = abl.get("Full dataset")
        full_mse = full_row["test_mse"] if full_row else float("nan")

        # Re-derive Calibra quality score
        print(f"  Loading {dataset_path} for quality score ...", flush=True)
        try:
            batch = _load(dataset_path)
            _, cal_score = _calibra_quality_score(batch)
        except Exception as e:
            print(f"  [skip] Could not score {dataset_path}: {e}")
            continue

        cal_vs_rnd_pct = (rnd_mse - cal_mse) / rnd_mse * 100 if rnd_mse else 0.0
        rows.append({
            "dataset": dataset_path,
            "n_episodes": data.get("train_episodes", "?"),
            "calibra_score": cal_score,
            "full_mse_mean":    full_mse,
            "full_mse_std":     float("nan"),
            "calibra_mse_mean": cal_mse,
            "calibra_mse_std":  float(cal_row.get("test_mse_std") or "nan"),
            "random_mse_mean":  rnd_mse,
            "random_mse_std":   float(rnd_row.get("test_mse_std") or "nan"),
            "calibra_vs_random_pct": cal_vs_rnd_pct,
            "n_seeds": data.get("n_seeds", "?"),
        })
    return rows


# ── Spearman ─────────────────────────────────────────────────────────────────

def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman ρ and p-value."""
    from scipy.stats import spearmanr
    return spearmanr(x, y)


# ── reporting ─────────────────────────────────────────────────────────────────

def print_report(rows: list[dict]) -> None:
    print(f"\n{'='*_W}")
    print("  CROSS-DATASET SPEARMAN CORRELATION — Calibra BC-MLP")
    print(f"{'='*_W}")
    print(f"  {'Dataset':<35} {'N':>5}  {'CalScore':>8}  {'FullMSE':>10}  "
          f"{'CalMSE':>10}  {'RndMSE':>10}  {'Cal vs Rnd':>11}")
    print(f"  {'─'*(_W-2)}")
    for r in rows:
        ds = r["dataset"].replace("lerobot/", "")
        print(f"  {ds:<35} {str(r['n_episodes']):>5}  {r['calibra_score']:>8.1f}  "
              f"{r['full_mse_mean']:>10.5f}  {r['calibra_mse_mean']:>10.5f}  "
              f"{r['random_mse_mean']:>10.5f}  {r['calibra_vs_random_pct']:>+10.1f}%")
    print(f"{'='*_W}")

    if len(rows) < 3:
        print(f"\n  ⚠️  Only {len(rows)} datasets — need ≥3 for meaningful Spearman.")
        return

    scores   = [r["calibra_score"]        for r in rows]
    full_mse = [r["full_mse_mean"]         for r in rows]
    cal_mse  = [r["calibra_mse_mean"]      for r in rows]
    rnd_mse  = [r["random_mse_mean"]       for r in rows]
    improvement = [r["calibra_vs_random_pct"] for r in rows]

    rho1, p1 = spearman(scores, full_mse)
    rho2, p2 = spearman(scores, improvement)
    rho3, p3 = spearman(rnd_mse, cal_mse)

    print(f"\n  Spearman correlations (N={len(rows)} datasets):")
    print(f"    ρ(CalScore, FullDatasetMSE) = {rho1:+.4f}  p={p1:.4g}"
          f"  {'✅' if abs(rho1) > 0.60 else '─ '} "
          f"{'negative = higher-quality data trains better' if rho1 < 0 else ''}")
    print(f"    ρ(CalScore, CalVsRnd%)      = {rho2:+.4f}  p={p2:.4g}"
          f"  {'✅' if abs(rho2) > 0.60 else '─ '} "
          f"{'positive = Calibra helps most where data is lowest quality' if rho2 < 0 else ''}")
    print(f"    ρ(RndMSE,   CalMSE)         = {rho3:+.4f}  p={p3:.4g}"
          f"  (sanity: Calibra tracks dataset difficulty)")

    print(f"\n  Target: |ρ| > 0.65 for publication claim")
    print(f"  {'✅ PASS' if abs(rho1) > 0.65 or abs(rho2) > 0.65 else '⚠️  below target'}")
    print(f"{'='*_W}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(prog="cross_dataset_spearman",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   metavar="DATASET",
                   help="LeRobot Hub IDs or local paths to evaluate")
    p.add_argument("--seeds", type=int, default=5,
                   help="Train/test split seeds per dataset (default: 5)")
    p.add_argument("--n-epochs", type=int, default=200,
                   help="BC-MLP training epochs (default: 200)")
    p.add_argument("--keep", type=float, default=0.30,
                   help="Coreset keep fraction (default: 0.30)")
    p.add_argument("--from-json", nargs="+", metavar="PATH",
                   help="Load pre-computed ablation_benchmark.py JSON results instead of rerunning")
    p.add_argument("--json", metavar="PATH",
                   help="Save per-dataset results to this JSON path")
    p.add_argument("--save-fig", action="store_true",
                   help="Save Spearman scatter plot to experiments/figures/")
    args = p.parse_args(argv)

    print(f"{'='*_W}")
    print("  CALIBRA — Cross-Dataset Spearman Correlation Benchmark")
    print(f"{'='*_W}")

    if args.from_json:
        print(f"  Mode: loading {len(args.from_json)} pre-computed JSON result(s)")
        rows = ingest_from_json(args.from_json)
    else:
        print(f"  Datasets   : {args.datasets}")
        print(f"  Seeds/split: {args.seeds}")
        print(f"  Epochs     : {args.n_epochs}")
        print(f"  Keep frac  : {args.keep:.0%}")
        rows = []
        for ds in args.datasets:
            try:
                row = run_dataset(ds, n_seeds=args.seeds,
                                  n_epochs=args.n_epochs,
                                  keep_fraction=args.keep)
                rows.append(row)
            except Exception as e:
                print(f"  [ERROR] {ds}: {e}")
                import traceback; traceback.print_exc()

    print_report(rows)

    if args.json and rows:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"  Results saved → {out}")

    if args.save_fig and len(rows) >= 3:
        _save_figure(rows)

    return rows


def _save_figure(rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping figure)")
        return

    scores      = [r["calibra_score"]        for r in rows]
    improvement = [r["calibra_vs_random_pct"] for r in rows]
    labels      = [r["dataset"].replace("lerobot/", "") for r in rows]

    from scipy.stats import spearmanr
    rho, p_val = spearmanr(scores, improvement)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4, color="#dddddd")

    ax.scatter(scores, improvement, s=120, c="#2563EB", edgecolors="black",
               linewidths=0.7, zorder=4)
    for x, y, lbl in zip(scores, improvement, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#333333")

    if len(rows) >= 2:
        m, b = np.polyfit(scores, improvement, 1)
        xs = np.linspace(min(scores) - 3, max(scores) + 3, 100)
        ax.plot(xs, m * xs + b, "--", color="#EF4444", linewidth=1.5, zorder=2)

    ax.set_xlabel("Calibra Quality Score", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Calibra vs. Random Improvement (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Cross-Dataset Spearman Correlation  (N={len(rows)})\n"
        f"Spearman ρ = {rho:.3f}   p = {p_val:.4g}",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.tight_layout()

    out = FIG_DIR / "fig_cross_dataset_spearman.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_cross_dataset_spearman.png", bbox_inches="tight", dpi=200)
    print(f"  Figure saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
