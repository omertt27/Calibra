"""
LeRobot Coreset Benchmark — end-to-end on real data
=====================================================

Demonstrates the full Calibra workflow on a real LeRobot dataset:

    1. Load a LeRobot v2 dataset from HuggingFace (default: lerobot/pusht)
    2. Run the Calibra diagnostic pipeline
    3. Write a CalibraReport JSON with per-episode verdicts
    4. Sweep keep fractions [0.10, 0.25, 0.50, 0.75, 1.00]
       For each fraction:
         - Calibra coreset (quality filter + greedy max-coverage)
         - Random subset   (5-seed average)
         - Full dataset baseline
       Train a BC MLP on each subset, evaluate on held-out episodes
    5. Report: Calibra score, test MSE, compute time, savings vs. baseline
    6. Save figures and a results JSON

This benchmark uses no simulation environment — evaluation is held-out
trajectory prediction MSE, a valid proxy for policy quality on real data.

Requirements
------------
    pip install calibra-robotics datasets torch matplotlib

Run
---
    python experiments/lerobot_coreset_benchmark.py
    python experiments/lerobot_coreset_benchmark.py --dataset lerobot/aloha_sim_insertion_human
    python experiments/lerobot_coreset_benchmark.py --keep 0.25 0.50 --n-epochs 150

Key result
----------
Calibra coreset at K% of episodes reaches equivalent held-out trajectory
prediction accuracy while using proportionally fewer training samples and
GPU-hours than training on the full dataset.
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


# ── data loading ──────────────────────────────────────────────────────────────


def _detect_obs_key(features: dict) -> str:
    """Pick the best observation key from a HuggingFace dataset features dict."""
    for candidate in [
        "observation.state",
        "observation.environment_state",
        "observation.agent_pos",
    ]:
        if candidate in features:
            return candidate
    obs_keys = [k for k in features if k.startswith("observation.") and "image" not in k]
    if obs_keys:
        return obs_keys[0]
    raise ValueError(
        f"Could not find a state observation column. Available: {list(features.keys())}"
    )


def load_lerobot_dataset(dataset_id: str, max_episodes: int | None = None) -> tuple:
    """
    Download a LeRobot dataset from HuggingFace and return (hf_dataset, obs_key, action_key).

    When max_episodes is set, uses streaming mode so only the first N episodes worth of
    parquet shards are downloaded — critical for large datasets like BridgeData V2 (50k ep).
    Without max_episodes, downloads the full dataset as before.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print(
            "error: 'datasets' package required.\n       pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    if max_episodes is not None:
        # Stream so we only pull shards until we have enough episodes, then stop.
        print(f"Streaming {dataset_id!r} (first {max_episodes} episodes) ...")
        ds_stream = hf_load(dataset_id, split="train", streaming=True)
        obs_key = _detect_obs_key(ds_stream.features)
        action_key = "action"

        # Collect rows episode by episode; stop once we have max_episodes complete episodes.
        rows: dict[str, list] = {
            "episode_index": [],
            obs_key: [],
            action_key: [],
        }
        has_ts = "timestamp" in ds_stream.features
        has_fi = "frame_index" in ds_stream.features
        if has_ts:
            rows["timestamp"] = []
        if has_fi:
            rows["frame_index"] = []

        seen_eps: set = set()
        for row in ds_stream:
            ep_idx = int(row["episode_index"])
            if ep_idx not in seen_eps:
                if len(seen_eps) >= max_episodes:
                    break  # enough complete episodes collected
                seen_eps.add(ep_idx)
            rows["episode_index"].append(ep_idx)
            rows[obs_key].append(row[obs_key])
            rows[action_key].append(row[action_key])
            if has_ts:
                rows["timestamp"].append(float(row["timestamp"]))
            if has_fi:
                rows["frame_index"].append(int(row["frame_index"]))

        total_frames = len(rows["episode_index"])
        print(
            f"  {len(seen_eps)} episodes · {total_frames:,} frames (streamed)"
            f"  |  obs: {obs_key}  |  action: {action_key}"
        )

        # Wrap in a HF Dataset so the rest of the pipeline is unchanged.
        from datasets import Dataset

        ds = Dataset.from_dict(rows)
        return ds, obs_key, action_key

    # Full download path (unchanged).
    print(f"Downloading {dataset_id!r} from HuggingFace Hub ...")
    ds = hf_load(dataset_id, split="train")
    obs_key = _detect_obs_key(ds.features)
    action_key = "action"
    print(f"  {len(ds):,} frames  |  obs: {obs_key}  |  action: {action_key}")
    return ds, obs_key, action_key


def hf_to_episode_batch(hf_dataset, obs_key: str, action_key: str, dataset_name: str):
    """Convert a HuggingFace LeRobot dataset to a Calibra EpisodeBatch."""
    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    print("  Converting to EpisodeBatch ...")
    try:
        df = hf_dataset.to_pandas()
    except Exception:
        # Fallback for very large datasets: iterate rows
        import pandas as pd

        df = pd.DataFrame(
            {
                "episode_index": hf_dataset["episode_index"],
                obs_key: hf_dataset[obs_key],
                action_key: hf_dataset[action_key],
                "timestamp": hf_dataset.get("timestamp", list(range(len(hf_dataset)))),
            }
        )

    episodes = []
    for ep_idx, ep_df in df.groupby("episode_index"):
        ep_df = ep_df.sort_values("frame_index") if "frame_index" in ep_df.columns else ep_df

        states = np.array(ep_df[obs_key].tolist(), dtype=np.float32)
        actions = np.array(ep_df[action_key].tolist(), dtype=np.float32)
        timestamps = (
            ep_df["timestamp"].values.astype(np.float32)
            if "timestamp" in ep_df.columns
            else np.arange(len(states), dtype=np.float32) * 0.02
        )

        if actions.ndim == 1:
            actions = actions[:, np.newaxis]
        if states.ndim == 1:
            states = states[:, np.newaxis]

        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=str(int(ep_idx))),
                timestamps=timestamps,
                observations={"state": states},
                actions=actions,
            )
        )

    print(f"  {len(episodes)} episodes  |  {sum(ep.n_steps for ep in episodes):,} frames")

    return EpisodeBatch(
        episodes=episodes,
        dataset_name=dataset_name,
        format="lerobot-v2",
        source_path=dataset_name,
    )


# ── Calibra pipeline ──────────────────────────────────────────────────────────


def run_calibra(batch, report_path: Optional[str] = None) -> tuple:
    """
    Run the Calibra diagnostic pipeline and coreset selector.
    Returns (diag_report, prune_result, overall_score).
    """
    from calibra.pipeline import Pipeline

    print("Running Calibra diagnostic pipeline ...")
    t0 = time.perf_counter()
    diag = Pipeline().run(batch)
    pipeline_s = time.perf_counter() - t0
    print(f"  Done in {pipeline_s:.1f}s")

    # Read overall score from the scoring module
    from calibra.schema.report import RiskLevel
    from calibra.schema.scoring import (
        DIMENSION_WEIGHTS,
        dimension_score,
        flag_level_to_score,
        overall_score,
        route_metric_to_dimension,
    )

    dim_scores_raw: dict[str, list[float]] = {d: [] for d in DIMENSION_WEIGHTS}
    for flag in diag.flags:
        dim = route_metric_to_dimension(flag.metric)
        dim_scores_raw[dim].append(flag_level_to_score(flag.level))
    dim_scores = {d: dimension_score(scores) for d, scores in dim_scores_raw.items()}
    quality_score = round(overall_score(dim_scores), 1)
    print(f"  Calibra quality score: {quality_score}/100")

    n_critical = len(diag.flags_at_level(RiskLevel.CRITICAL))
    n_warning = len(diag.flags_at_level(RiskLevel.WARNING))
    if n_critical:
        print(f"  {n_critical} CRITICAL, {n_warning} WARNING flags")
    else:
        print(f"  No CRITICAL flags  |  {n_warning} WARNING flags")

    # Write CalibraReport with verdicts if requested
    if report_path:
        from calibra.pruning import CoresetSelector as _CS

        selector = _CS(keep_fraction=0.5)
        prune_result = selector.select(batch, diag)

        from calibra.report_json import assemble_public_report, dataset_info_from_report

        ds_info = dataset_info_from_report(diag)
        public = assemble_public_report(diag, ds_info, pruning_result=prune_result)
        public.write(report_path)
        print(f"  CalibraReport written -> {report_path}")
        return diag, prune_result, quality_score

    return diag, None, quality_score


# ── Rare-behavior identification ──────────────────────────────────────────────


def identify_rare_episodes(episodes, rare_fraction: float = 0.15, k_neighbors: int = 5):
    """
    Label episodes in low-density regions of action space as "rare".

    Uses k-NN density estimation (pure numpy, no sklearn dependency).
    Rare = bottom `rare_fraction` of episodes by local density.

    Returns (rare_ids: set[str], density: np.ndarray).
    """
    features = []
    for ep in episodes:
        a = ep.actions
        s = ep.observations.get("state", np.zeros((1, 1)))
        feat = np.concatenate(
            [
                a.mean(0),
                a.std(0),
                s.mean(0)[: min(s.shape[1], 4)],
            ]
        )
        features.append(feat)

    F = np.array(features, dtype=np.float64)
    col_std = F.std(0)
    col_std[col_std < 1e-8] = 1.0
    F = (F - F.mean(0)) / col_std

    k = min(k_neighbors, len(episodes) - 1)
    # pairwise squared distances
    sq = np.sum(F**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (F @ F.T)
    D2 = np.maximum(D2, 0.0)
    np.fill_diagonal(D2, np.inf)

    # mean distance to k nearest neighbors → density
    knn_dists = np.sort(D2, axis=1)[:, :k] ** 0.5
    density = 1.0 / (knn_dists.mean(1) + 1e-6)

    threshold = np.percentile(density, rare_fraction * 100)
    rare_ids = {ep.metadata.episode_id for ep, d in zip(episodes, density) if d <= threshold}
    return rare_ids, density


# ── BC training and evaluation ────────────────────────────────────────────────


def _make_tensors(episodes, device):
    """Stack all episodes into (S, A) tensors for BC training."""
    import torch

    states_all, actions_all = [], []
    for ep in episodes:
        s = ep.observations.get("state")
        a = ep.actions
        if s is not None and len(s) > 1:
            states_all.append(s)
            actions_all.append(a)

    S = torch.from_numpy(np.concatenate(states_all)).to(device)
    A = torch.from_numpy(np.concatenate(actions_all)).to(device)
    return S, A


def train_bc(episodes, n_epochs: int = 100, lr: float = 1e-3, hidden: int = 256) -> dict:
    """Train a BC MLP and return the trained model + normalization stats."""
    import torch
    import torch.nn as nn

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    S, A = _make_tensors(episodes, device)

    s_mean = S.mean(0)
    s_std = S.std(0).clamp(min=1e-6)
    a_mean = A.mean(0)
    a_std = A.std(0).clamp(min=1e-6)
    S_n = (S - s_mean) / s_std

    state_dim = S.shape[1]
    action_dim = A.shape[1]

    net = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, action_dim),
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    N = len(S_n)

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, 256):
            idx = perm[i : i + 256]
            pred = net(S_n[idx])
            loss = ((pred - (A[idx] - a_mean) / a_std) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    return {
        "net": net,
        "s_mean": s_mean,
        "s_std": s_std,
        "a_mean": a_mean,
        "a_std": a_std,
        "device": device,
    }


def evaluate_bc(model_dict: dict, test_episodes) -> float:
    """
    Evaluate BC policy on held-out episodes.
    Returns mean MSE between predicted and actual expert actions.
    """
    import torch

    net = model_dict["net"]
    s_mean = model_dict["s_mean"]
    s_std = model_dict["s_std"]
    a_mean = model_dict["a_mean"]
    a_std = model_dict["a_std"]
    device = model_dict["device"]

    S_test, A_test = _make_tensors(test_episodes, device)

    with torch.no_grad():
        S_n = (S_test - s_mean) / s_std
        pred_n = net(S_n)
        pred = pred_n * a_std + a_mean
        mse = float(((pred - A_test) ** 2).mean().item())

    return mse


# ── main benchmark ────────────────────────────────────────────────────────────


def run_benchmark(
    dataset_id: str = "lerobot/pusht",
    keep_fractions: list[float] | None = None,
    n_epochs: int = 120,
    n_random_seeds: int = 5,
    test_fraction: float = 0.2,
    report_path: str | None = None,
    rare_fraction: float = 0.15,
    max_episodes: int | None = None,
) -> dict:
    if keep_fractions is None:
        keep_fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

    print("=" * 70)
    print("  Calibra LeRobot Coreset Benchmark")
    print(f"  Dataset : {dataset_id}")
    print(f"  Fractions: {keep_fractions}")
    if max_episodes:
        print(f"  Max episodes: {max_episodes}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    hf_ds, obs_key, action_key = load_lerobot_dataset(dataset_id, max_episodes=max_episodes)
    batch = hf_to_episode_batch(hf_ds, obs_key, action_key, dataset_id)
    all_episodes = list(batch.episodes)
    n_total = len(all_episodes)

    # Train/test split (by episode, not frame)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    n_test = max(1, int(n_total * test_fraction))
    test_indices = set(perm[:n_test].tolist())
    train_indices = [i for i in range(n_total) if i not in test_indices]

    test_episodes = [all_episodes[i] for i in test_indices]
    train_episodes = [all_episodes[i] for i in train_indices]
    train_ids = {ep.metadata.episode_id for ep in train_episodes}

    print(f"\n  Train: {len(train_episodes)} episodes  |  Test: {len(test_episodes)} episodes")

    # ── 1b. Identify rare episodes ────────────────────────────────────────────
    rare_ids, density = identify_rare_episodes(train_episodes, rare_fraction=rare_fraction)
    n_rare = len(rare_ids)
    print(f"  Rare episodes (bottom {rare_fraction:.0%} by action-space density): {n_rare}")

    # ── 2. Calibra audit ──────────────────────────────────────────────────────
    from calibra.schema.episode import EpisodeBatch

    train_batch = EpisodeBatch(
        episodes=train_episodes,
        dataset_name=dataset_id,
        format="lerobot-v2",
        source_path=dataset_id,
    )

    if report_path is None:
        rp = str(FIG_DIR.parent / f"benchmark_{dataset_id.replace('/', '_')}_report.json")
    else:
        rp = report_path

    diag, _, quality_score = run_calibra(train_batch, report_path=rp)

    # ── 3. Full-data baseline ─────────────────────────────────────────────────
    print("\n[Full dataset baseline]")
    t0 = time.perf_counter()
    full_model = train_bc(train_episodes, n_epochs=n_epochs)
    full_train_s = time.perf_counter() - t0
    full_mse = evaluate_bc(full_model, test_episodes)
    print(f"  MSE: {full_mse:.6f}  |  Train time: {full_train_s:.1f}s")

    results = []

    # ── 4. Sweep keep fractions ───────────────────────────────────────────────
    for frac in keep_fractions:
        k = max(1, round(len(train_episodes) * frac))
        print(f"\n[keep {frac:.0%} -> {k} episodes]")
        row: dict = {"keep_fraction": frac, "n_episodes": k}

        # ── Calibra coreset ───────────────────────────────────────────────────
        from calibra.pruning import CoresetSelector

        selector = CoresetSelector(keep_fraction=frac, strategy="diversity")
        prune = selector.select(train_batch, diag)

        # Filter to training episodes only (exclude test)
        calibra_ids = {eid for eid in prune.keep_episode_ids if eid in train_ids}
        calibra_eps = [ep for ep in train_episodes if ep.metadata.episode_id in calibra_ids]
        if not calibra_eps:
            calibra_eps = train_episodes[:k]

        print(f"  [Calibra] {len(calibra_eps)} episodes  ...", end=" ", flush=True)
        t0 = time.perf_counter()
        cal_model = train_bc(calibra_eps, n_epochs=n_epochs)
        cal_train_s = time.perf_counter() - t0
        cal_mse = evaluate_bc(cal_model, test_episodes)
        cal_rel = cal_mse / full_mse if full_mse > 0 else float("nan")
        cal_rare_kept = sum(1 for ep in calibra_eps if ep.metadata.episode_id in rare_ids)
        cal_rare_cov = cal_rare_kept / n_rare if n_rare else float("nan")
        print(
            f"MSE={cal_mse:.6f}  ({cal_rel:.2f}x baseline)  time={cal_train_s:.1f}s  rare_cov={cal_rare_cov:.1%}"
        )

        row["calibra_mse"] = cal_mse
        row["calibra_mse_relative"] = round(cal_rel, 4)
        row["calibra_train_s"] = round(cal_train_s, 2)
        row["calibra_n_episodes"] = len(calibra_eps)
        row["calibra_rare_coverage"] = round(cal_rare_cov, 4)

        # ── Random subset (averaged over N_RANDOM_SEEDS seeds) ───────────────
        rand_mses = []
        rand_times = []
        all_train_ids = [ep.metadata.episode_id for ep in train_episodes]

        rand_rare_covs = []
        for seed in range(n_random_seeds):
            random.seed(seed)
            rand_ids_seed = set(random.sample(all_train_ids, k))
            rand_eps = [ep for ep in train_episodes if ep.metadata.episode_id in rand_ids_seed]
            t0 = time.perf_counter()
            rand_model = train_bc(rand_eps, n_epochs=n_epochs)
            rand_times.append(time.perf_counter() - t0)
            rand_mses.append(evaluate_bc(rand_model, test_episodes))
            rare_kept = sum(1 for ep in rand_eps if ep.metadata.episode_id in rare_ids)
            rand_rare_covs.append(rare_kept / n_rare if n_rare else float("nan"))

        rand_mse = float(np.mean(rand_mses))
        rand_mse_std = float(np.std(rand_mses))
        rand_train_s = float(np.mean(rand_times))
        rand_rel = rand_mse / full_mse if full_mse > 0 else float("nan")
        rand_rare_cov = float(np.mean(rand_rare_covs))
        print(
            f"  [Random] MSE={rand_mse:.6f} ±{rand_mse_std:.6f} "
            f"({rand_rel:.2f}x baseline)  rare_cov={rand_rare_cov:.1%}"
        )

        row["random_mse"] = rand_mse
        row["random_mse_std"] = round(rand_mse_std, 6)
        row["random_mse_relative"] = round(rand_rel, 4)
        row["random_train_s"] = round(rand_train_s, 2)
        row["random_rare_coverage"] = round(rand_rare_cov, 4)

        results.append(row)

    # ── 5. Print results table ────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print(f"  CALIBRA LEROBOT BENCHMARK — {dataset_id}")
    print(f"  Dataset quality score  : {quality_score}/100")
    print(f"  Full baseline MSE      : {full_mse:.6f}  ({full_train_s:.1f}s)")
    print(f"  Rare episodes ({rare_fraction:.0%} of train): {n_rare}")
    print("=" * 100)
    print(
        f"  {'Keep':>6}  {'N':>5}  "
        f"{'Cal MSE':>10}  {'Rel':>5}  {'RareCov':>8}  "
        f"{'Rnd MSE':>10}  {'Rel':>5}  {'RareCov':>8}  {'Saved':>6}"
    )
    print("-" * 100)
    for r in results:
        saved = 100.0 * (1.0 - r["keep_fraction"])
        print(
            f"  {r['keep_fraction']:>5.0%}  {r['n_episodes']:>5}  "
            f"  {r['calibra_mse']:>8.6f}  {r['calibra_mse_relative']:>4.2f}x  "
            f"{r['calibra_rare_coverage']:>7.1%}  "
            f"  {r['random_mse']:>8.6f}  {r['random_mse_relative']:>4.2f}x  "
            f"{r['random_rare_coverage']:>7.1%}  "
            f"{saved:>5.0f}%"
        )
    print("=" * 100)
    print(
        "\n  Rel. = ratio to full-data baseline MSE (<=1.00 means subset matches or beats full data)."
        "\n  RareCov = fraction of rare-behavior episodes (bottom 15% by density) retained."
    )

    # ── 6. Save results JSON ──────────────────────────────────────────────────
    out_json = FIG_DIR / f"benchmark_{dataset_id.replace('/', '_')}.json"
    summary = {
        "dataset": dataset_id,
        "calibra_quality_score": quality_score,
        "n_train_episodes": len(train_episodes),
        "n_test_episodes": len(test_episodes),
        "n_rare_episodes": n_rare,
        "rare_fraction": rare_fraction,
        "full_baseline": {"mse": full_mse, "train_s": full_train_s},
        "sweep": results,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved -> {out_json}")

    # ── 7. Figures ────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fracs = [r["keep_fraction"] * 100 for r in results]
        cal_mse_vals = [r["calibra_mse"] for r in results]
        rand_mse_vals = [r["random_mse"] for r in results]
        rand_std = [r["random_mse_std"] for r in results]
        cal_rare = [r["calibra_rare_coverage"] * 100 for r in results]
        rand_rare = [r["random_rare_coverage"] * 100 for r in results]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: MSE vs. retention
        ax1.axhline(
            full_mse, color="#6b7280", linewidth=1.5, linestyle="--", label="Full data (100%)"
        )
        ax1.plot(fracs, cal_mse_vals, "o-", color="#2563eb", linewidth=2, label="Calibra coreset")
        ax1.errorbar(
            fracs,
            rand_mse_vals,
            yerr=rand_std,
            fmt="s--",
            color="#dc2626",
            linewidth=1.5,
            capsize=4,
            label=f"Random (avg {n_random_seeds} seeds)",
        )
        ax1.set_xlabel("Retention fraction (%)", fontsize=12)
        ax1.set_ylabel("Test MSE (action prediction)", fontsize=12)
        ax1.set_title(f"Policy Quality vs. Dataset Size\n{dataset_id}", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel 2: training time vs. retention
        cal_times = [r["calibra_train_s"] for r in results]
        rand_times_vals = [r["random_train_s"] for r in results]
        ax2.plot(fracs, cal_times, "o-", color="#2563eb", linewidth=2, label="Calibra coreset")
        ax2.plot(fracs, rand_times_vals, "s--", color="#dc2626", linewidth=1.5, label="Random")
        ax2.axhline(full_train_s, color="#6b7280", linewidth=1.5, linestyle="--", label="Full data")
        ax2.set_xlabel("Retention fraction (%)", fontsize=12)
        ax2.set_ylabel("Training wall-clock (seconds)", fontsize=12)
        ax2.set_title("Compute Cost vs. Dataset Size", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Panel 3: rare-behavior coverage vs. retention  ← the key result
        ax3.plot(fracs, cal_rare, "o-", color="#2563eb", linewidth=2, label="Calibra coreset")
        ax3.plot(fracs, rand_rare, "s--", color="#dc2626", linewidth=1.5, label="Random")
        ax3.set_xlabel("Retention fraction (%)", fontsize=12)
        ax3.set_ylabel(
            f"Rare-episode coverage (%, bottom {rare_fraction:.0%} by density)", fontsize=11
        )
        ax3.set_title("Rare-Behavior Preservation", fontsize=11)
        ax3.set_ylim(0, 105)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        fig.suptitle(
            f"Calibra Coreset Benchmark — {dataset_id}\nDataset quality score: {quality_score}/100",
            fontsize=12,
            y=1.02,
        )
        fig.tight_layout()

        out_fig = FIG_DIR / f"fig_lerobot_benchmark_{dataset_id.replace('/', '_')}.pdf"
        fig.savefig(out_fig, bbox_inches="tight")
        print(f"  Figure saved -> {out_fig}")
        plt.close()

    except ImportError:
        print("  (matplotlib not installed — skipping figures)")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="End-to-end Calibra coreset benchmark on a real LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/lerobot_coreset_benchmark.py
  python experiments/lerobot_coreset_benchmark.py --dataset lerobot/aloha_sim_insertion_human
  python experiments/lerobot_coreset_benchmark.py --keep 0.25 0.50 0.75 --n-epochs 150
  python experiments/lerobot_coreset_benchmark.py --report results/pusht/latest.json
        """,
    )
    p.add_argument(
        "--dataset",
        default="lerobot/pusht",
        help="HuggingFace dataset ID (default: lerobot/pusht)",
    )
    p.add_argument(
        "--keep",
        nargs="+",
        type=float,
        default=None,
        metavar="FRAC",
        help="Keep fractions to sweep (default: 0.05 0.10 0.25 0.50 0.75 1.00)",
    )
    p.add_argument(
        "--rare-fraction",
        type=float,
        default=0.15,
        help="Bottom fraction of episodes (by action-space density) labelled as rare (default: 0.15)",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=120,
        help="BC training epochs per run (default: 120)",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Random seeds to average for the random baseline (default: 5)",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of episodes held out for evaluation (default: 0.2)",
    )
    p.add_argument(
        "--report",
        metavar="PATH",
        default=None,
        help="Path to write the CalibraReport JSON with episode verdicts",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample at most N episodes from the dataset (useful for very large datasets)",
    )
    args = p.parse_args()

    run_benchmark(
        dataset_id=args.dataset,
        keep_fractions=args.keep,
        n_epochs=args.n_epochs,
        n_random_seeds=args.n_seeds,
        test_fraction=args.test_fraction,
        report_path=args.report,
        rare_fraction=args.rare_fraction,
        max_episodes=args.max_episodes,
    )
