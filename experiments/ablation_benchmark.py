"""
Ablation + Retention Curve Benchmark
=====================================
Answers two questions that determine whether Calibra's gains are real:

  1. ABLATION — which component drives the improvement, and does Calibra beat
     established coreset methods (not just random)?
     Conditions (all at the same episode budget k, same features, same pipeline):
       - Full dataset         (all data, upper bound)
       - Random k             (no selection, baseline)
       - K-Center greedy      (published: Gonzalez farthest-point sampling)
       - Herding              (published: mean-matching greedy selection)
       - Facility Location    (published: greedy submodular coverage)
       - Quality-filter only  (remove bad episodes, keep top-k by quality score)
       - Diversity-only       (no quality filter, greedy max-coverage of k)
       - Calibra full         (quality filter + diversity selection)

     The three published baselines run on the SAME behavioral features as Calibra's
     diversity stage, so the only variable is the selection algorithm. They see the
     full set with no quality filtering — isolating whether Calibra's quality-aware
     two-stage pipeline beats off-the-shelf coreset geometry.

     If quality-only beats random but diversity-only doesn't -> quality filter is the mechanism.
     If both beat random and full > both -> they are complementary and the interaction matters.
     On DROID-style noisy data we expect: quality-only >= full > diversity-only > random,
     and Calibra full > the best published baseline (which has no notion of data quality).

  2. RETENTION CURVE — is the Pareto advantage stable across data fractions?
     Sweeps keep_fraction in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00].
     Plots Calibra vs. random at each point.
     A stable upward gap across the whole curve is the strongest possible claim.

Usage
-----
    # Ablation on ALOHA
    python experiments/ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet

    # Ablation + retention curve on DROID
    python experiments/ablation_benchmark.py --dataset lerobot/droid_100 --curve

    # All datasets, save figures + JSON
    python experiments/ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet \\
        --curve --n-epochs 300 --seeds 5 --save-fig --json results/ablation_aloha.json
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

_W = 72


# ── reuse helpers from real_il_benchmark ─────────────────────────────────────


def _load(path: str):
    from calibra.ingestion.registry import load

    return load(path)


def _obs_key(ep):
    for k in ("state", "proprio", "joint_position", "eef_position"):
        if k in ep.observations and ep.observations[k].ndim == 2:
            return k
    return None


def _split(batch, test_fraction=0.20, seed=0):
    from calibra.schema.episode import EpisodeBatch

    eps = list(batch.episodes)
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_test = max(1, round(len(eps) * test_fraction))
    test_eps, train_eps = eps[:n_test], eps[n_test:]
    return (
        EpisodeBatch(train_eps, batch.dataset_name + "_train", batch.format, batch.source_path),
        EpisodeBatch(test_eps, batch.dataset_name + "_test", batch.format, batch.source_path),
    )


def _collect(batch):
    states_all, actions_all = [], []
    for ep in batch.episodes:
        key = _obs_key(ep)
        if key is None:
            continue
        s, a = ep.observations[key], ep.actions
        ml = min(len(s), len(a))
        if ml < 2:
            continue
        states_all.append(s[:ml])
        actions_all.append(a[:ml])
    if not states_all:
        raise ValueError("No usable observation arrays found.")
    return np.concatenate(states_all), np.concatenate(actions_all)


def _train_bc(batch, n_epochs=200, lr=1e-3, batch_size=256, hidden=256, seed=None):
    import torch
    import torch.nn as nn

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    S_np, A_np = _collect(batch)
    state_dim, action_dim = S_np.shape[1], A_np.shape[1]
    S = torch.from_numpy(S_np).float().to(device)
    A = torch.from_numpy(A_np).float().to(device)
    s_mean, s_std = S.mean(0), S.std(0).clamp(min=1e-6)
    a_mean, a_std = A.mean(0), A.std(0).clamp(min=1e-6)
    S_n, A_n = (S - s_mean) / s_std, (A - a_mean) / a_std

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
    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            loss = ((net(S_n[idx]) - A_n[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return dict(net=net, s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std, device=device)


def _eval_mse(artifacts, test_batch):
    import torch

    net, s_mean, s_std, a_mean, a_std, device = (
        artifacts[k] for k in ("net", "s_mean", "s_std", "a_mean", "a_std", "device")
    )
    S_np, A_np = _collect(test_batch)
    S = torch.from_numpy(S_np).float().to(device)
    A = torch.from_numpy(A_np).float().to(device)
    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std
    with torch.no_grad():
        mse = ((net(S_n) - A_n) ** 2).mean().item()
    return mse


def _random_subset(batch, k, seed=42):
    from calibra.schema.episode import EpisodeBatch

    eps = list(batch.episodes)
    rng = random.Random(seed)
    chosen = rng.sample(eps, min(k, len(eps)))
    return EpisodeBatch(chosen, batch.dataset_name + "_rnd", batch.format, batch.source_path)


def _run_calibra_pipeline(batch):
    from calibra.pipeline import Pipeline

    return Pipeline().run(batch)


def _subset_from_ids(batch, keep_ids):
    from calibra.schema.episode import EpisodeBatch

    keep_set = set(keep_ids)
    chosen = [ep for ep in batch.episodes if ep.metadata.episode_id in keep_set]
    return EpisodeBatch(chosen, batch.dataset_name + "_sub", batch.format, batch.source_path)


def _subset_from_indices(batch, indices, tag):
    from calibra.schema.episode import EpisodeBatch

    eps = list(batch.episodes)
    chosen = [eps[i] for i in indices]
    return EpisodeBatch(chosen, batch.dataset_name + tag, batch.format, batch.source_path)


# ── published coreset baselines ───────────────────────────────────────────────
#
# K-Center greedy, Herding, and Facility Location are classic data-selection
# methods from the coreset / active-learning literature. To make the comparison
# fair, all three operate on the SAME per-episode feature representation Calibra's
# diversity stage uses (normalized [action_mean, action_std] per dim — the
# behavioral branch of calibra.pruning._build_feature_matrix). The only variable
# across conditions is therefore the SELECTION ALGORITHM, not the features or the
# training/eval pipeline. Unlike Calibra, these baselines see the full set with no
# quality filtering — they select purely on behavioral geometry.


def _episode_feature_matrix(batch) -> np.ndarray:
    """(n_episodes, 2*action_dim) behavioral features, min-max normalized to [0, 1].

    Matches the behavioral representation used by Calibra's diversity selector so
    the baselines are compared on identical features.
    """
    # Determine action_dim from the first non-empty episode so empty episodes get
    # a correctly-sized zero row (a single dataset has one consistent action_dim).
    action_dim = 1
    for ep in batch.episodes:
        a = ep.actions
        if a.size:
            action_dim = a.shape[1] if a.ndim > 1 else 1
            break

    rows = []
    for ep in batch.episodes:
        acts = ep.actions
        if acts.ndim == 1:
            acts = acts[:, np.newaxis]
        if len(acts) == 0:
            rows.append(np.zeros(2 * action_dim, dtype=np.float64))
            continue
        rows.append(np.concatenate([acts.mean(axis=0), acts.std(axis=0)]))
    mat = np.stack(rows, axis=0).astype(np.float64)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = mat.min(axis=0), mat.max(axis=0)
    scale = hi - lo
    scale[scale == 0] = 1.0  # constant features contribute nothing
    return (mat - lo) / scale


def _kcenter_indices(feats: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """Gonzalez greedy k-center (farthest-point sampling). O(N*K*D)."""
    n = len(feats)
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    selected = [start]
    min_d = np.linalg.norm(feats - feats[start], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(min_d))
        selected.append(i)
        min_d = np.minimum(min_d, np.linalg.norm(feats - feats[i], axis=1))
    return selected


def _herding_indices(feats: np.ndarray, k: int) -> list[int]:
    """Greedy herding: pick points so the running mean matches the full-set mean.

    At step t, choose the unselected episode that minimizes
    || target_mean - mean(selected + {i}) ||. O(N*K*D).
    """
    n = len(feats)
    if k >= n:
        return list(range(n))
    target = feats.mean(axis=0)
    chosen = np.zeros(n, dtype=bool)
    running = np.zeros_like(target)
    selected: list[int] = []
    for t in range(k):
        cand_mean = (running + feats) / (t + 1)  # (N, D)
        errs = np.linalg.norm(cand_mean - target, axis=1)
        errs[chosen] = np.inf
        i = int(np.argmin(errs))
        selected.append(i)
        chosen[i] = True
        running = running + feats[i]
    return selected


def _facility_location_indices(feats: np.ndarray, k: int) -> list[int]:
    """Greedy submodular facility-location: maximize sum_i max_{j in S} sim(i, j).

    Similarity is a Gaussian kernel over feature distance with sigma = median
    pairwise distance. O(N^2) memory — intended for benchmark-scale datasets
    (up to a few thousand episodes).
    """
    n = len(feats)
    if k >= n:
        return list(range(n))
    diff = feats[:, None, :] - feats[None, :, :]
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    pos = dist[dist > 0]
    sigma = float(np.median(pos)) if pos.size else 1.0
    if sigma == 0:
        sigma = 1.0
    sim = np.exp(-(dist**2) / (2 * sigma**2))
    chosen = np.zeros(n, dtype=bool)
    cur_max = np.zeros(n)  # max sim of each i to selected set
    selected: list[int] = []
    for _ in range(k):
        gains = np.maximum(sim - cur_max[:, None], 0.0).sum(axis=0)
        gains[chosen] = -np.inf
        j = int(np.argmax(gains))
        selected.append(j)
        chosen[j] = True
        cur_max = np.maximum(cur_max, sim[:, j])
    return selected


def select_kcenter(batch, k, seed=0):
    return _subset_from_indices(
        batch, _kcenter_indices(_episode_feature_matrix(batch), k, seed), "_kcenter"
    )


def select_herding(batch, k):
    return _subset_from_indices(
        batch, _herding_indices(_episode_feature_matrix(batch), k), "_herding"
    )


def select_facility(batch, k):
    return _subset_from_indices(
        batch, _facility_location_indices(_episode_feature_matrix(batch), k), "_facility"
    )


# ── ablation selectors ────────────────────────────────────────────────────────


def select_quality_only(batch, report, k):
    """Top-k by composite quality score (lower = cleaner). No diversity."""
    from calibra.pruning import CoresetSelector

    sel = CoresetSelector(
        keep_fraction=k / max(len(batch.episodes), 1),
        quality_only=False,  # run both stages but...
        diversity_weight=0.0,  # ...zero diversity weight -> quality drives everything
        max_spike_rate=0.10,
        max_vel_disc_rate=0.25,
        max_dropout_fraction=0.10,
        min_ldlj=-30.0,
    )
    result = sel.select(batch, report)
    return _subset_from_ids(batch, result.keep_episode_ids)


def select_diversity_only(batch, report, k):
    """Diversity selection with quality filter disabled (max thresholds)."""
    from calibra.pruning import CoresetSelector

    sel = CoresetSelector(
        keep_fraction=k / max(len(batch.episodes), 1),
        quality_only=False,
        diversity_weight=1.0,
        max_spike_rate=1.0,  # no quality gating
        max_vel_disc_rate=1.0,
        max_dropout_fraction=1.0,
        min_ldlj=-1e6,
    )
    result = sel.select(batch, report)
    return _subset_from_ids(batch, result.keep_episode_ids)


def select_calibra_full(batch, report, k):
    """Full Calibra pipeline: quality filter + diversity selection."""
    from calibra.pruning import CoresetSelector

    sel = CoresetSelector(
        keep_fraction=k / max(len(batch.episodes), 1),
        strategy="diversity",
    )
    result = sel.select(batch, report)
    return _subset_from_ids(batch, result.keep_episode_ids)


# ── ablation run ──────────────────────────────────────────────────────────────


def run_ablation(
    train_batch,
    test_batch,
    keep_fraction: float = 0.30,
    n_epochs: int = 200,
    n_random_seeds: int = 5,
) -> list[dict]:
    """Run all 5 ablation conditions and return list of result dicts."""
    k = max(1, round(len(train_batch.episodes) * keep_fraction))

    print("  Running Calibra pipeline ...", flush=True)
    t0 = time.perf_counter()
    report = _run_calibra_pipeline(train_batch)
    print(f"  Pipeline done in {time.perf_counter() - t0:.1f}s", flush=True)

    conditions = [
        ("Full dataset", lambda: train_batch),
        # published coreset baselines (same features, no quality filter)
        ("K-Center greedy", lambda: select_kcenter(train_batch, k, seed=0)),
        ("Herding", lambda: select_herding(train_batch, k)),
        ("Facility Location", lambda: select_facility(train_batch, k)),
        # Calibra ablation ladder
        ("Quality-filter only", lambda: select_quality_only(train_batch, report, k)),
        ("Diversity-only", lambda: select_diversity_only(train_batch, report, k)),
        ("Calibra full", lambda: select_calibra_full(train_batch, report, k)),
    ]

    # Every condition is trained over the SAME set of training seeds so per-seed
    # MSEs are paired across conditions (enables a paired significance test).
    # Selection is deterministic per condition, so only BC training varies by seed;
    # the random baseline additionally varies its subset per seed.
    seeds = list(range(n_random_seeds))

    random_mses = [
        _eval_mse(
            _train_bc(_random_subset(train_batch, k, seed=s * 17 + 42), n_epochs=n_epochs, seed=s),
            test_batch,
        )
        for s in seeds
    ]
    random_mean = float(np.mean(random_mses))
    random_std = float(np.std(random_mses))

    rows = [
        {
            "condition": "Random",
            "n_episodes": k,
            "test_mse": random_mean,
            "test_mse_std": random_std,
            "per_seed_mse": random_mses,
            "vs_random": 0.0,
        }
    ]
    print(
        f"  Random (k={k}, {n_random_seeds} seeds): mse={random_mean:.5f} +/- {random_std:.5f}",
        flush=True,
    )

    for label, make_subset in conditions:
        sub = make_subset()  # deterministic subset
        per_seed = [_eval_mse(_train_bc(sub, n_epochs=n_epochs, seed=s), test_batch) for s in seeds]
        mse = float(np.mean(per_seed))
        sd = float(np.std(per_seed))
        delta = (
            (random_mean - mse) / random_mean * 100
        )  # % improvement vs random (positive = better)
        rows.append(
            {
                "condition": label,
                "n_episodes": len(sub.episodes),
                "test_mse": mse,
                "test_mse_std": sd,
                "per_seed_mse": per_seed,
                "vs_random": delta,
            }
        )
        marker = "+++ " if label == "Calibra full" else "    "
        print(
            f"  {marker}{label:<26} k={len(sub.episodes):>3}  "
            f"mse={mse:.5f}+/-{sd:.5f}  vs_random={delta:+.1f}%",
            flush=True,
        )

    return rows


# ── retention curve ───────────────────────────────────────────────────────────


def run_retention_curve(
    train_batch,
    test_batch,
    keep_fractions: list[float],
    n_epochs: int = 200,
    n_random_seeds: int = 3,
) -> list[dict]:
    """Sweep keep_fraction. At each point: Calibra full vs. random mean."""
    print("  Running Calibra pipeline ...", flush=True)
    t0 = time.perf_counter()
    report = _run_calibra_pipeline(train_batch)
    print(f"  Pipeline done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Full dataset baseline
    art_full = _train_bc(train_batch, n_epochs=n_epochs)
    full_mse = _eval_mse(art_full, test_batch)
    print(f"  Full dataset (k={len(train_batch.episodes)}): mse={full_mse:.5f}", flush=True)

    curve = []
    for frac in keep_fractions:
        k = max(1, round(len(train_batch.episodes) * frac))

        rnd_mses = []
        for seed in range(n_random_seeds):
            sub = _random_subset(train_batch, k, seed=seed * 17 + 42)
            rnd_mses.append(_eval_mse(_train_bc(sub, n_epochs=n_epochs), test_batch))
        rnd_mean = float(np.mean(rnd_mses))

        cal_sub = select_calibra_full(train_batch, report, k)
        cal_mse = _eval_mse(_train_bc(cal_sub, n_epochs=n_epochs), test_batch)

        gap_pct = (rnd_mean - cal_mse) / rnd_mean * 100
        curve.append(
            {
                "keep_fraction": frac,
                "k": k,
                "full_mse": full_mse,
                "random_mse": rnd_mean,
                "calibra_mse": cal_mse,
                "calibra_vs_random_pct": gap_pct,
            }
        )
        print(
            f"  keep={frac:.0%}  k={k:>3}  rand={rnd_mean:.5f}  calibra={cal_mse:.5f}  gap={gap_pct:+.1f}%",
            flush=True,
        )

    return curve, full_mse


# ── printing ──────────────────────────────────────────────────────────────────


def print_ablation(dataset_name: str, keep_fraction: float, rows: list[dict]) -> None:
    print(f"\n{'=' * _W}")
    print(f"  CALIBRA ABLATION - {dataset_name.upper()} (keep={keep_fraction:.0%})")
    print(f"{'=' * _W}")
    print(f"  {'Condition':<30} {'Episodes':>8}  {'Test MSE':>10}  {'vs. Random':>11}")
    print(f"  {'-' * 66}")
    random_row = next(r for r in rows if "Random" in r["condition"])
    for r in rows:
        marker = ">>>" if r["condition"] == "Calibra full" else "   "
        std_str = f" +/-{r['test_mse_std']:.5f}" if r.get("test_mse_std") else ""
        vs_str = (
            f"{r['vs_random']:+.1f}%" if r["condition"] != random_row["condition"] else "baseline"
        )
        print(
            f"  {marker} {r['condition']:<28} {r['n_episodes']:>8}  {r['test_mse']:>10.5f}{std_str}  {vs_str:>11}"
        )
    print(f"{'=' * _W}")

    # Interpret mechanism
    rows_by_name = {r["condition"]: r for r in rows}
    q_gain = rows_by_name.get("Quality-filter only", {}).get("vs_random", 0)
    d_gain = rows_by_name.get("Diversity-only", {}).get("vs_random", 0)
    f_gain = rows_by_name.get("Calibra full", {}).get("vs_random", 0)

    print("\n  Mechanism interpretation:")
    if q_gain > 1 and d_gain > 1:
        print(f"  Both quality filter (+{q_gain:.1f}%) and diversity (+{d_gain:.1f}%) contribute.")
        print(f"  Full pipeline ({f_gain:+.1f}%) confirms they are complementary.")
    elif q_gain > d_gain:
        print(
            f"  Quality filter is the primary driver ({q_gain:+.1f}% vs diversity {d_gain:+.1f}%)."
        )
        print("  Dataset has real corruption that naive sampling retains.")
    else:
        print(
            f"  Diversity selection is the primary driver ({d_gain:+.1f}% vs quality {q_gain:+.1f}%)."
        )
        print("  Dataset is clean but redundant.")

    # Headline vs. published coreset baselines
    baseline_gains = {
        name: rows_by_name[name]["vs_random"]
        for name in ("K-Center greedy", "Herding", "Facility Location")
        if name in rows_by_name
    }
    if baseline_gains:
        best_bn = max(baseline_gains, key=baseline_gains.get)
        best_bg = baseline_gains[best_bn]
        print("\n  vs. published baselines:")
        for name, g in baseline_gains.items():
            print(f"    {name:<20} {g:+.1f}% vs random")
        print(
            f"  Best baseline: {best_bn} ({best_bg:+.1f}%). "
            f"Calibra full ({f_gain:+.1f}%) -> {f_gain - best_bg:+.1f}% vs {best_bn}."
        )
    print()


def print_curve(dataset_name: str, curve: list[dict], full_mse: float) -> None:
    print(f"\n{'=' * _W}")
    print(f"  CALIBRA RETENTION CURVE - {dataset_name.upper()}")
    print(f"{'=' * _W}")
    print(
        f"  {'Keep':>6}  {'k':>4}  {'Random MSE':>11}  {'Calibra MSE':>12}  {'Gap':>8}  {'vs. Full':>9}"
    )
    print(f"  {'-' * 60}")
    for r in curve:
        vs_full = (full_mse - r["calibra_mse"]) / full_mse * 100
        print(
            f"  {r['keep_fraction']:>5.0%}  {r['k']:>4}  "
            f"{r['random_mse']:>11.5f}  {r['calibra_mse']:>12.5f}  "
            f"{r['calibra_vs_random_pct']:>+7.1f}%  {vs_full:>+8.1f}%"
        )
    print(
        f"  {'100%':>5}  {curve[0]['k'] // 1:>4}  {'(full)':>11}  {full_mse:>12.5f}  {'—':>8}  {'0.0%':>9}"
    )
    print(f"{'=' * _W}\n")


# ── figure ────────────────────────────────────────────────────────────────────


def save_ablation_figure(dataset_name: str, keep_fraction: float, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipping figure)")
        return

    labels = [r["condition"] for r in rows]
    mses = [r["test_mse"] for r in rows]
    colors = []
    for r in rows:
        c = r["condition"]
        if "Full" in c:
            colors.append("#6B7280")
        elif "Random" in c:
            colors.append("#EF4444")
        elif "K-Center" in c:
            colors.append("#14B8A6")
        elif "Herding" in c:
            colors.append("#EC4899")
        elif "Facility" in c:
            colors.append("#A16207")
        elif "Quality" in c:
            colors.append("#F59E0B")
        elif "Diversity" in c:
            colors.append("#8B5CF6")
        else:
            colors.append("#2563EB")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labels)), mses, color=colors, width=0.55, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Test MSE (action prediction, lower=better)", fontsize=10)
    ax.set_title(
        f"Ablation: which component drives Calibra's gains?\n{dataset_name}  keep={keep_fraction:.0%}",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mse in zip(bars, mses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{mse:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    import matplotlib.patches as mpatches

    legend = [
        mpatches.Patch(color="#6B7280", label="Full dataset"),
        mpatches.Patch(color="#EF4444", label="Random k"),
        mpatches.Patch(color="#14B8A6", label="K-Center greedy"),
        mpatches.Patch(color="#EC4899", label="Herding"),
        mpatches.Patch(color="#A16207", label="Facility Location"),
        mpatches.Patch(color="#F59E0B", label="Quality-filter only"),
        mpatches.Patch(color="#8B5CF6", label="Diversity-only"),
        mpatches.Patch(color="#2563EB", label="Calibra full pipeline"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")
    fig.tight_layout()

    slug = dataset_name.lower().replace("/", "_").replace("-", "_")
    out = FIG_DIR / f"fig_ablation_{slug}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"fig_ablation_{slug}.png", bbox_inches="tight", dpi=150)
    print(f"  Ablation figure saved: {out}")
    plt.close()


def save_curve_figure(dataset_name: str, curve: list[dict], full_mse: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fracs = [r["keep_fraction"] * 100 for r in curve]
    rnd = [r["random_mse"] for r in curve]
    cal = [r["calibra_mse"] for r in curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, rnd, "s--", color="#EF4444", linewidth=2, label="Random pruned", zorder=3)
    ax.plot(fracs, cal, "o-", color="#2563EB", linewidth=2, label="Calibra coreset", zorder=4)
    ax.axhline(full_mse, color="#6B7280", linestyle=":", linewidth=1.5, label="Full dataset")
    ax.fill_between(fracs, rnd, cal, alpha=0.12, color="#2563EB", label="Calibra advantage")

    ax.set_xlabel("Data fraction kept (%)", fontsize=11)
    ax.set_ylabel("Test MSE (lower = better)", fontsize=11)
    ax.set_title(
        f"Retention curve: Calibra Pareto frontier\n{dataset_name}", fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    slug = dataset_name.lower().replace("/", "_").replace("-", "_")
    out = FIG_DIR / f"fig_retention_{slug}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"fig_retention_{slug}.png", bbox_inches="tight", dpi=150)
    print(f"  Retention curve figure saved: {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    p = argparse.ArgumentParser(prog="ablation_benchmark")
    p.add_argument(
        "--dataset", default="lerobot/aloha_mobile_cabinet", help="LeRobot Hub ID or local path"
    )
    p.add_argument(
        "--keep", "-k", type=float, default=0.30, help="Keep fraction for ablation (default: 0.30)"
    )
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument(
        "--seeds", type=int, default=5, help="Random seeds to average for baseline (default: 5)"
    )
    p.add_argument("--curve", action="store_true", help="Also run retention curve sweep")
    p.add_argument(
        "--curve-fractions",
        default="0.10,0.20,0.30,0.50,0.70",
        help="Comma-separated keep fractions for retention curve",
    )
    p.add_argument("--save-fig", action="store_true")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args(argv)

    print("=" * _W)
    print("  CALIBRA ABLATION BENCHMARK")
    print("=" * _W)
    print(f"  Dataset : {args.dataset}")
    print(f"  Keep    : {args.keep:.0%}")
    print(f"  Epochs  : {args.n_epochs}")
    print(f"  Seeds   : {args.seeds}")
    print()

    print("[1/3] Loading dataset ...")
    batch = _load(args.dataset)
    ep0 = batch.episodes[0]
    state_dim = (
        ep0.observations.get(_obs_key(ep0), np.zeros((1, 1))).shape[1] if _obs_key(ep0) else 0
    )
    print(
        f"  {batch.n_episodes} episodes  state_dim={state_dim}  action_dim={ep0.actions.shape[1]}"
    )

    print("[2/3] Train/test split (80/20) ...")
    train_batch, test_batch = _split(batch)
    print(f"  train={train_batch.n_episodes}  test={test_batch.n_episodes}")

    output = {
        "dataset": args.dataset,
        "keep_fraction": args.keep,
        "n_epochs": args.n_epochs,
        "n_seeds": args.seeds,
        "train_episodes": train_batch.n_episodes,
        "test_episodes": test_batch.n_episodes,
    }

    print(f"\n[3/3] Running ablation (keep={args.keep:.0%}, {args.seeds} random seeds) ...")
    ablation_rows = run_ablation(
        train_batch,
        test_batch,
        keep_fraction=args.keep,
        n_epochs=args.n_epochs,
        n_random_seeds=args.seeds,
    )
    print_ablation(batch.dataset_name, args.keep, ablation_rows)
    output["ablation"] = ablation_rows

    if args.save_fig:
        save_ablation_figure(batch.dataset_name, args.keep, ablation_rows)

    if args.curve:
        fracs = [float(x) for x in args.curve_fractions.split(",")]
        print(f"\n[4/4] Retention curve: {fracs} ...")
        curve, full_mse = run_retention_curve(
            train_batch, test_batch, keep_fractions=fracs, n_epochs=args.n_epochs, n_random_seeds=3
        )
        print_curve(batch.dataset_name, curve, full_mse)
        output["retention_curve"] = curve
        output["full_dataset_mse"] = full_mse
        if args.save_fig:
            save_curve_figure(batch.dataset_name, curve, full_mse)

    if args.json:
        out_path = pathlib.Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
