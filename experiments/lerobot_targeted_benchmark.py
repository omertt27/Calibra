"""
Calibra Targeted Benchmark -- Statistical validation with tail-conditioned evaluation.
=======================================================================================

Answers three specific questions on real LeRobot data:

  5%  -- Does Calibra help under extreme compression?
  10% -- Is the apparent Random advantage repeatable or noise?
  25% -- Can Calibra reliably match full-data MSE with 75% fewer episodes?

Methods compared (per fraction):
  full_unfiltered    -- all training episodes, no filtering
  random_full        -- random from full pool (random selection + training seed)
  random_quality     -- random from quality-approved pool (same k episodes)
  quality_only       -- quality filter, then random from approved pool
  diversity_only     -- no quality filter, k-center from full pool
  calibra            -- quality filter + k-center diversity selection

Evaluation (per trained model):
  mse_overall        -- mean squared action prediction error on all test episodes
  mse_common         -- same, restricted to top-85% density test episodes
  mse_tail           -- same, restricted to bottom-15% density test episodes
  rare_coverage      -- fraction of tail train episodes selected

Statistics (per method per fraction):
  mean, std, 95% CI, paired difference vs. random_full,
  Cohen's d vs. random_full, paired t-test p-value

Design note on seeds
--------------------
For methods with deterministic episode selection (calibra, quality_only,
diversity_only), each seed controls only training randomness (PyTorch seed).
For random methods (random_full, random_quality) each seed controls both
episode selection and training.  This separates selection variance from
training variance and enables paired statistical comparison.

Run
---
    python experiments/lerobot_targeted_benchmark.py
    python experiments/lerobot_targeted_benchmark.py --dataset lerobot/pusht --n-seeds 10
    python experiments/lerobot_targeted_benchmark.py --n-seeds 3 --n-epochs 60  # smoke test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random as pyrandom
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "experiments" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# reuse data-loading and training helpers from the sweep benchmark
from experiments.lerobot_coreset_benchmark import (  # noqa: E402
    evaluate_bc,
    hf_to_episode_batch,
    identify_rare_episodes,
    load_lerobot_dataset,
    train_bc,
)

# ── statistical helpers ────────────────────────────────────────────────────────


def _ci95(values: list[float]) -> float:
    """Half-width of the 95% t-interval (returns 0 if n < 2)."""
    from math import sqrt

    n = len(values)
    if n < 2:
        return float("nan")
    s = float(np.std(values, ddof=1))
    # t_{0.975, n-1} approximated; exact for small n via table
    t_table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        12: 2.179,
        15: 2.131,
        20: 2.086,
    }
    t = t_table.get(n - 1, 1.96)  # fallback to z for large n
    return t * s / sqrt(n)


def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d: (mean_b - mean_a) / pooled_std. Positive = b is larger."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    pooled = ((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2)
    return (float(np.mean(b)) - float(np.mean(a))) / (pooled**0.5 + 1e-12)


def paired_ttest_p(a: list[float], b: list[float]) -> float:
    """Two-sided paired t-test p-value.  a and b must have the same length."""
    diffs = [bi - ai for ai, bi in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    if std_d < 1e-12:
        return 0.0
    t = mean_d / (std_d / n**0.5)
    # two-sided p using t-distribution survival approximation (valid for n>=2)
    from math import exp, lgamma

    df = n - 1
    x = df / (df + t * t)

    # regularised incomplete beta (scipy-free) via continued-fraction or series
    # use a simple numerical integration for robustness
    def beta_inc(a, b, x, steps=200):
        # numerical integration of Beta CDF
        dt = x / steps
        s = 0.0
        for i in range(steps):
            xi = (i + 0.5) * dt
            s += xi ** (a - 1) * (1 - xi) ** (b - 1) * dt

        return s * exp(lgamma(a + b) - lgamma(a) - lgamma(b))

    try:
        p_one = 0.5 * beta_inc(df / 2, 0.5, x)
    except Exception:
        return float("nan")
    return min(1.0, 2 * p_one)


def stats_summary(values: list[float], reference: list[float] | None = None) -> dict:
    """Return mean, std, 95% CI, and optionally paired stats vs. reference."""
    out: dict[str, Any] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "ci95": _ci95(values),
        "n": len(values),
    }
    if reference is not None and len(reference) == len(values):
        diffs = [float(r) - float(v) for v, r in zip(values, reference)]
        out["paired_diff_from_ref"] = float(np.mean(diffs))  # positive = ref is worse
        out["paired_diff_ci95"] = _ci95(diffs)
        out["cohens_d"] = cohen_d(values, reference)
        out["p_paired"] = paired_ttest_p(values, reference)
    return out


# ── episode density in action space ──────────────────────────────────────────


def density_split(episodes, tail_fraction: float = 0.15):
    """
    Split episodes into tail (low-density) and common (high-density) groups.
    Returns (tail_ids, common_ids, density_array).
    """
    tail_ids, density = identify_rare_episodes(episodes, rare_fraction=tail_fraction)
    common_ids = {
        ep.metadata.episode_id for ep in episodes if ep.metadata.episode_id not in tail_ids
    }
    return tail_ids, common_ids, density


# ── BC training with explicit torch seed ──────────────────────────────────────


def train_bc_seeded(episodes, n_epochs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return train_bc(episodes, n_epochs=n_epochs)


# ── coreset selectors ─────────────────────────────────────────────────────────


def select_random(pool, k: int, seed: int):
    pyrandom.seed(seed)
    ids = [ep.metadata.episode_id for ep in pool]
    chosen = set(pyrandom.sample(ids, min(k, len(ids))))
    return [ep for ep in pool if ep.metadata.episode_id in chosen]


def select_diversity(pool_batch, diag, frac: float, quality: bool):
    """k-center diversity selection, optionally with quality pre-filter."""
    from calibra.pruning import CoresetSelector

    if quality:
        selector = CoresetSelector(keep_fraction=frac, strategy="diversity")
    else:
        selector = CoresetSelector(
            keep_fraction=frac,
            strategy="diversity",
            max_spike_rate=1.0,
            max_vel_disc_rate=1.0,
            max_dropout_fraction=1.0,
            min_ldlj=-1000.0,
        )
    result = selector.select(pool_batch, diag)
    chosen = result.keep_episode_ids
    return [ep for ep in pool_batch.episodes if ep.metadata.episode_id in chosen]


# ── main benchmark ─────────────────────────────────────────────────────────────


def run_targeted_benchmark(
    dataset_id: str = "lerobot/pusht",
    fractions: list[float] | None = None,
    n_epochs: int = 120,
    n_seeds: int = 10,
    test_fraction: float = 0.2,
    tail_fraction: float = 0.15,
) -> dict:
    if fractions is None:
        fractions = [0.05, 0.10, 0.25]

    print("=" * 72)
    print("  Calibra Targeted Benchmark -- Tail-Conditioned Statistical Evaluation")
    print(f"  Dataset  : {dataset_id}")
    print(f"  Fractions: {fractions}  |  Seeds: {n_seeds}  |  Epochs: {n_epochs}")
    print("=" * 72)

    # ── 1. Load and split ─────────────────────────────────────────────────────
    hf_ds, obs_key, action_key = load_lerobot_dataset(dataset_id)
    batch = hf_to_episode_batch(hf_ds, obs_key, action_key, dataset_id)
    all_eps = list(batch.episodes)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_eps))
    n_test = max(1, int(len(all_eps) * test_fraction))
    test_eps = [all_eps[i] for i in perm[:n_test]]
    train_eps = [all_eps[i] for i in perm[n_test:]]

    print(f"\n  Train: {len(train_eps)} ep  |  Test: {len(test_eps)} ep")

    # ── 2. Tail identification (train and test independently) ─────────────────
    train_tail_ids, train_common_ids, _ = density_split(train_eps, tail_fraction)
    test_tail_ids, test_common_ids, _ = density_split(test_eps, tail_fraction)
    n_train_tail = len(train_tail_ids)

    test_tail_eps = [ep for ep in test_eps if ep.metadata.episode_id in test_tail_ids]
    test_common_eps = [ep for ep in test_eps if ep.metadata.episode_id in test_common_ids]

    print(f"  Train tail (action-space low-density, bottom {tail_fraction:.0%}): {n_train_tail} ep")
    print(f"  Test  tail: {len(test_tail_eps)} ep  |  Test common: {len(test_common_eps)} ep")

    # ── 3. Calibra diagnostic pipeline ────────────────────────────────────────
    from calibra.pipeline import Pipeline
    from calibra.schema.episode import EpisodeBatch

    train_batch = EpisodeBatch(
        episodes=train_eps,
        dataset_name=dataset_id,
        format="lerobot-v2",
        source_path=dataset_id,
    )

    print("\nRunning Calibra diagnostic pipeline ...")
    t0 = time.perf_counter()
    diag = Pipeline().run(train_batch)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Quality-approved pool
    from calibra.pruning import CoresetSelector as _CS

    _qselector = _CS(
        keep_fraction=1.0,
        strategy="diversity",
        max_spike_rate=0.10,
        max_vel_disc_rate=0.25,
        max_dropout_fraction=0.10,
        min_ldlj=-30.0,
    )
    _qres = _qselector.select(train_batch, diag)
    quality_ids = _qres.keep_episode_ids
    quality_eps = [ep for ep in train_eps if ep.metadata.episode_id in quality_ids]
    EpisodeBatch(
        episodes=quality_eps,
        dataset_name=dataset_id,
        format="lerobot-v2",
        source_path=dataset_id,
    )

    print(
        f"  Quality-approved pool: {len(quality_eps)}/{len(train_eps)} ep "
        f"({len(quality_eps) / len(train_eps):.0%})"
    )

    # ── 4. Full baselines (single run each, train seed=0) ─────────────────────
    print("\n[Full unfiltered baseline]")
    t0 = time.perf_counter()
    full_model = train_bc_seeded(train_eps, n_epochs, seed=0)
    full_train_s = time.perf_counter() - t0
    full_mse_overall = evaluate_bc(full_model, test_eps)
    full_mse_common = evaluate_bc(full_model, test_common_eps) if test_common_eps else float("nan")
    full_mse_tail = evaluate_bc(full_model, test_tail_eps) if test_tail_eps else float("nan")
    print(
        f"  MSE overall={full_mse_overall:.4f}  common={full_mse_common:.4f}  "
        f"tail={full_mse_tail:.4f}  ({full_train_s:.1f}s)"
    )

    print("\n[Quality-approved full baseline]")
    t0 = time.perf_counter()
    qual_model = train_bc_seeded(quality_eps, n_epochs, seed=0)
    qual_train_s = time.perf_counter() - t0
    qual_mse_overall = evaluate_bc(qual_model, test_eps)
    qual_mse_common = evaluate_bc(qual_model, test_common_eps) if test_common_eps else float("nan")
    qual_mse_tail = evaluate_bc(qual_model, test_tail_eps) if test_tail_eps else float("nan")
    print(
        f"  MSE overall={qual_mse_overall:.4f}  common={qual_mse_common:.4f}  "
        f"tail={qual_mse_tail:.4f}  ({qual_train_s:.1f}s)"
    )

    # ── 5. Seeded sweep ───────────────────────────────────────────────────────
    METHODS = [
        "random_full",
        "random_quality",
        "quality_only",
        "diversity_only",
        "calibra",
    ]

    # Pre-compute deterministic selections (same across all seeds for deterministic methods)
    {m: {} for m in ("quality_only", "diversity_only", "calibra")}

    results: dict[str, dict] = {}

    for frac in fractions:
        k = max(1, round(len(train_eps) * frac))
        k_q = max(1, round(len(quality_eps) * frac))
        print(f"\n{'=' * 72}")
        print(f"  Fraction {frac:.0%} -- k={k} from full pool, k_q={k_q} from quality pool")
        print("=" * 72)

        # deterministic selections for this fraction
        calibra_eps = select_diversity(train_batch, diag, frac=frac, quality=True)
        divonly_eps = select_diversity(train_batch, diag, frac=frac, quality=False)
        # quality_only: quality filter -> random sample (use seed=999 for consistent baseline)
        pyrandom.seed(999)
        qualonly_eps = select_random(quality_eps, k_q, seed=999)

        tail_cov = {
            "quality_only": sum(
                1 for ep in qualonly_eps if ep.metadata.episode_id in train_tail_ids
            )
            / n_train_tail,
            "diversity_only": sum(
                1 for ep in divonly_eps if ep.metadata.episode_id in train_tail_ids
            )
            / n_train_tail,
            "calibra": sum(1 for ep in calibra_eps if ep.metadata.episode_id in train_tail_ids)
            / n_train_tail,
        }

        print(
            f"  Calibra selected {len(calibra_eps)} ep  |  "
            f"diversity_only {len(divonly_eps)} ep  |  quality_only {len(qualonly_eps)} ep"
        )
        print(
            f"  Tail coverage: calibra={tail_cov['calibra']:.1%}  "
            f"div_only={tail_cov['diversity_only']:.1%}  "
            f"qual_only={tail_cov['quality_only']:.1%}"
        )

        method_runs: dict[str, dict[str, list[float]]] = {
            m: {"overall": [], "common": [], "tail": [], "train_s": []} for m in METHODS
        }

        for seed in range(n_seeds):
            print(f"  seed {seed:2d}", end="")

            # random_full: selection varies per seed
            rnd_full_eps = select_random(train_eps, k, seed=seed)
            t0 = time.perf_counter()
            m = train_bc_seeded(rnd_full_eps, n_epochs, seed=seed)
            method_runs["random_full"]["train_s"].append(time.perf_counter() - t0)
            method_runs["random_full"]["overall"].append(evaluate_bc(m, test_eps))
            method_runs["random_full"]["common"].append(
                evaluate_bc(m, test_common_eps) if test_common_eps else float("nan")
            )
            method_runs["random_full"]["tail"].append(
                evaluate_bc(m, test_tail_eps) if test_tail_eps else float("nan")
            )
            print("  rnd_full=done", end="")

            # random_quality: selection from quality pool varies per seed
            rnd_qual_eps = select_random(quality_eps, k_q, seed=seed)
            t0 = time.perf_counter()
            m = train_bc_seeded(rnd_qual_eps, n_epochs, seed=seed)
            method_runs["random_quality"]["train_s"].append(time.perf_counter() - t0)
            method_runs["random_quality"]["overall"].append(evaluate_bc(m, test_eps))
            method_runs["random_quality"]["common"].append(
                evaluate_bc(m, test_common_eps) if test_common_eps else float("nan")
            )
            method_runs["random_quality"]["tail"].append(
                evaluate_bc(m, test_tail_eps) if test_tail_eps else float("nan")
            )
            print("  rnd_qual=done", end="")

            # deterministic methods: selection fixed, only training seed varies
            for label, eps in [
                ("quality_only", qualonly_eps),
                ("diversity_only", divonly_eps),
                ("calibra", calibra_eps),
            ]:
                t0 = time.perf_counter()
                m = train_bc_seeded(eps, n_epochs, seed=seed)
                method_runs[label]["train_s"].append(time.perf_counter() - t0)
                method_runs[label]["overall"].append(evaluate_bc(m, test_eps))
                method_runs[label]["common"].append(
                    evaluate_bc(m, test_common_eps) if test_common_eps else float("nan")
                )
                method_runs[label]["tail"].append(
                    evaluate_bc(m, test_tail_eps) if test_tail_eps else float("nan")
                )
            print("  det=done")

        # Build statistics for this fraction
        ref_overall = method_runs["random_full"]["overall"]
        ref_tail = method_runs["random_full"]["tail"]

        frac_result: dict[str, Any] = {
            "fraction": frac,
            "k_full": k,
            "k_quality": k_q,
            "n_quality_pool": len(quality_eps),
            "methods": {},
        }

        for method in METHODS:
            runs = method_runs[method]
            tc = tail_cov.get(method)
            if method == "random_full":
                # tail coverage is expected value: k/n_total * n_train_tail / n_train_tail = k/N
                rnd_tail_covs = []
                for seed in range(n_seeds):
                    rnd_eps_s = select_random(train_eps, k, seed=seed)
                    rnd_tail_covs.append(
                        sum(1 for ep in rnd_eps_s if ep.metadata.episode_id in train_tail_ids)
                        / n_train_tail
                    )
                tc = float(np.mean(rnd_tail_covs))
            elif method == "random_quality":
                rnd_q_tail_covs = []
                for seed in range(n_seeds):
                    rnd_eps_s = select_random(quality_eps, k_q, seed=seed)
                    rnd_q_tail_covs.append(
                        sum(1 for ep in rnd_eps_s if ep.metadata.episode_id in train_tail_ids)
                        / n_train_tail
                    )
                tc = float(np.mean(rnd_q_tail_covs))

            frac_result["methods"][method] = {
                "tail_coverage": round(tc, 4) if tc is not None else None,
                "overall": stats_summary(runs["overall"], ref_overall),
                "common": stats_summary(runs["common"], None),
                "tail": stats_summary(runs["tail"], ref_tail),
                "train_s": float(np.mean(runs["train_s"])),
            }

        results[str(frac)] = frac_result

    # ── 6. Print table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print(f"  CALIBRA TARGETED BENCHMARK -- {dataset_id}")
    print(
        f"  Full unfiltered: overall={full_mse_overall:.2f}  common={full_mse_common:.2f}  tail={full_mse_tail:.2f}"
    )
    print(
        f"  Quality-approved full: overall={qual_mse_overall:.2f}  common={qual_mse_common:.2f}  tail={qual_mse_tail:.2f}"
    )
    print(
        f"  Quality pool: {len(quality_eps)}/{len(train_eps)} ep  |  "
        f"Train tail episodes: {n_train_tail}  |  Test tail episodes: {len(test_tail_eps)}"
    )
    print("=" * 100)

    for frac in fractions:
        fr = results[str(frac)]
        k = fr["k_full"]
        print(
            f"\n--- {frac:.0%} retention ({k} ep from {len(train_eps)} full / "
            f"{fr['k_quality']} ep from {fr['n_quality_pool']} quality pool) ---"
        )
        print(
            f"  {'Method':<18}  {'TailCov':>8}  {'Overall':>9}  {'+-CI95':>7}  "
            f"{'DiffRef':>8}  {'p':>6}  {'d':>6}  {'Tail MSE':>9}  {'Common':>9}"
        )
        print(
            f"  {'-' * 18}  {'-' * 8}  {'-' * 9}  {'-' * 7}  {'-' * 8}  {'-' * 6}  {'-' * 6}  {'-' * 9}  {'-' * 9}"
        )
        for method in METHODS:
            m = fr["methods"][method]
            ov = m["overall"]
            tl = m["tail"]
            tc = m["tail_coverage"]
            diff = ov.get("paired_diff_from_ref", float("nan"))
            p = ov.get("p_paired", float("nan"))
            d = ov.get("cohens_d", float("nan"))
            print(
                f"  {method:<18}  {tc:>7.1%}  "
                f"{ov['mean']:>9.2f}  {ov['ci95']:>7.2f}  "
                f"{diff:>+8.2f}  {p:>6.3f}  {d:>+6.2f}  "
                f"{tl['mean']:>9.2f}  {m['common']['mean']:>9.2f}"
            )

    print("\n" + "=" * 100)
    print("  DiffRef = (random_full mean) - (method mean), positive = method is better")
    print("  d = Cohen's d vs. random_full, positive = method has lower MSE")
    print("  p = paired t-test vs. random_full")

    # ── 7. Save JSON ───────────────────────────────────────────────────────────
    summary = {
        "dataset": dataset_id,
        "n_train": len(train_eps),
        "n_test": len(test_eps),
        "n_quality_pool": len(quality_eps),
        "n_train_tail": n_train_tail,
        "n_test_tail": len(test_tail_eps),
        "tail_fraction": tail_fraction,
        "n_seeds": n_seeds,
        "n_epochs": n_epochs,
        "full_unfiltered": {
            "mse_overall": full_mse_overall,
            "mse_common": full_mse_common,
            "mse_tail": full_mse_tail,
            "train_s": full_train_s,
        },
        "quality_approved_full": {
            "mse_overall": qual_mse_overall,
            "mse_common": qual_mse_common,
            "mse_tail": qual_mse_tail,
            "train_s": qual_train_s,
        },
        "fractions": results,
    }

    tag = dataset_id.replace("/", "_")
    out_json = OUT_DIR / f"targeted_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved -> {out_json}")

    # ── 8. Figures ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        METHOD_STYLE = {
            "random_full": ("#dc2626", "s--", "Random (full)"),
            "random_quality": ("#f97316", "^--", "Random (quality pool)"),
            "quality_only": ("#8b5cf6", "D-.", "Quality-only"),
            "diversity_only": ("#10b981", "v:", "Diversity-only"),
            "calibra": ("#2563eb", "o-", "Calibra"),
        }

        xs = [f"{f:.0%}" for f in fractions]
        x = np.arange(len(fractions))
        width = 0.15

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        ax_ov = fig.add_subplot(gs[0, 0])
        ax_com = fig.add_subplot(gs[0, 1])
        ax_tl = fig.add_subplot(gs[0, 2])
        ax_tc = fig.add_subplot(gs[1, 0])
        ax_p = fig.add_subplot(gs[1, 1])
        ax_d = fig.add_subplot(gs[1, 2])

        for i, (method, (color, ls, label)) in enumerate(METHOD_STYLE.items()):
            means_ov, ci_ov, means_tl, ci_tl, means_co = [], [], [], [], []
            tc_vals, p_vals, d_vals = [], [], []
            for frac in fractions:
                m = results[str(frac)]["methods"][method]
                means_ov.append(m["overall"]["mean"])
                ci_ov.append(m["overall"]["ci95"])
                means_tl.append(m["tail"]["mean"])
                ci_tl.append(m["tail"]["ci95"])
                means_co.append(m["common"]["mean"])
                tc_vals.append(m["tail_coverage"] * 100)
                p_vals.append(m["overall"].get("p_paired", float("nan")))
                d_vals.append(m["overall"].get("cohens_d", 0.0))

            offset = (i - 2) * width
            ax_ov.bar(
                x + offset,
                means_ov,
                width,
                label=label,
                color=color,
                alpha=0.8,
                yerr=ci_ov,
                capsize=3,
            )
            ax_com.bar(x + offset, means_co, width, label=label, color=color, alpha=0.8)
            ax_tl.bar(
                x + offset,
                means_tl,
                width,
                label=label,
                color=color,
                alpha=0.8,
                yerr=ci_tl,
                capsize=3,
            )
            ax_tc.plot(xs, tc_vals, ls[1:], color=color, marker=ls[0], linewidth=2, label=label)
            if method != "random_full":
                ax_p.plot(xs, p_vals, ls[1:], color=color, marker=ls[0], linewidth=2, label=label)
                ax_d.plot(xs, d_vals, ls[1:], color=color, marker=ls[0], linewidth=2, label=label)

        for ax, title, ylabel in [
            (ax_ov, "Overall Test MSE", "MSE"),
            (ax_com, "Common-Test MSE", "MSE"),
            (ax_tl, "Tail-Test MSE (+/- 95% CI)", "MSE"),
        ]:
            ax.axhline(
                full_mse_overall,
                color="#6b7280",
                linewidth=1.2,
                linestyle=":",
                label="Full unfiltered",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(xs)
            ax.set_xlabel("Retention fraction")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        ax_tc.axhline(
            tail_fraction * 100,
            color="#6b7280",
            linewidth=1.2,
            linestyle=":",
            label=f"Expected random ({tail_fraction:.0%})",
        )
        ax_tc.set_xlabel("Retention fraction")
        ax_tc.set_ylabel("Tail coverage (%)")
        ax_tc.set_title("Action-Space Tail Coverage", fontsize=11)
        ax_tc.set_ylim(0, 105)
        ax_tc.legend(fontsize=8)
        ax_tc.grid(True, alpha=0.3)

        ax_p.axhline(0.05, color="#dc2626", linewidth=1.2, linestyle="--", label="p=0.05")
        ax_p.set_xlabel("Retention fraction")
        ax_p.set_ylabel("p-value (paired t-test vs. random_full)")
        ax_p.set_title("Statistical Significance", fontsize=11)
        ax_p.legend(fontsize=8)
        ax_p.grid(True, alpha=0.3)

        ax_d.axhline(0, color="#6b7280", linewidth=1.2, linestyle=":")
        ax_d.set_xlabel("Retention fraction")
        ax_d.set_ylabel("Cohen's d (positive = better than random_full)")
        ax_d.set_title("Effect Size vs. Random (full)", fontsize=11)
        ax_d.legend(fontsize=8)
        ax_d.grid(True, alpha=0.3)

        fig.suptitle(
            f"Calibra Targeted Benchmark -- {dataset_id}\n"
            f"(n_seeds={n_seeds}, n_epochs={n_epochs}, tail={tail_fraction:.0%} by action-space density)",
            fontsize=12,
        )

        out_fig = OUT_DIR / f"targeted_{tag}.pdf"
        fig.savefig(out_fig, bbox_inches="tight")
        print(f"  Figure saved -> {out_fig}")
        plt.close()

    except ImportError:
        print("  (matplotlib not installed -- skipping figures)")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Calibra targeted benchmark: tail-conditioned, statistically rigorous.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="lerobot/pusht", help="HuggingFace dataset ID (default: lerobot/pusht)"
    )
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=None,
        metavar="FRAC",
        help="Retention fractions (default: 0.05 0.10 0.25)",
    )
    p.add_argument("--n-epochs", type=int, default=120, help="BC training epochs (default: 120)")
    p.add_argument(
        "--n-seeds", type=int, default=10, help="Training/selection seeds per method (default: 10)"
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of episodes held out for evaluation (default: 0.2)",
    )
    p.add_argument(
        "--tail-fraction",
        type=float,
        default=0.15,
        help="Bottom fraction labeled tail by action-space density (default: 0.15)",
    )
    args = p.parse_args()

    run_targeted_benchmark(
        dataset_id=args.dataset,
        fractions=args.fractions,
        n_epochs=args.n_epochs,
        n_seeds=args.n_seeds,
        test_fraction=args.test_fraction,
        tail_fraction=args.tail_fraction,
    )
