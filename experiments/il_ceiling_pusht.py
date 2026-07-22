"""
Phase 2: IL Ceiling Experiment
==============================

Shows empirically that imitation learning performance has a data-quality
ceiling: as corruption level increases, BC success rate drops — and
Calibra's quality score tracks this ceiling BEFORE training.

Environment: 2D point-mass reach task (lightweight, no external deps)
  - State:  (x, y, vx, vy)  — end-effector position + velocity
  - Action: (ax, ay)         — acceleration command
  - Goal:   reach target within radius 0.15 from various start positions
  - Task phases: reach (approach), grasp (hover), transport, place

Conditions: corruption rates [0%, 10%, 20%, 30%, 40%, 60%, 80%]
  At each rate: Calibra quality score computed → BC MLP trained → success measured

Key result: Calibra quality score is a monotone proxy for the BC success
ceiling — measured BEFORE any training.

Run
---
    pip install calibra-robotics torch matplotlib
    python experiments/il_ceiling_pusht.py
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from calibra.schema.episode import EpisodeBatch

# ── environment paths ──────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── simulation environment ─────────────────────────────────────────────────────


class PointMassEnv:
    """
    2D point-mass environment for robot reach/manipulation tasks.
    Lightweight, no external dependencies.
    """

    DT = 0.05
    MAX_VEL = 2.0
    DAMPING = 0.85

    def __init__(self, goal: np.ndarray | None = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.goal = goal if goal is not None else np.array([0.8, 0.8])
        self.state = np.zeros(4, dtype=np.float32)
        self.reset()

    def reset(self, start: np.ndarray | None = None) -> np.ndarray:
        if start is not None:
            self.state = np.array([start[0], start[1], 0.0, 0.0], dtype=np.float32)
        else:
            pos = self.rng.uniform(-0.9, 0.9, size=2).astype(np.float32)
            self.state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float32)
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        ax, ay = float(action[0]), float(action[1])
        vx = self.state[2] * self.DAMPING + ax * self.DT
        vy = self.state[3] * self.DAMPING + ay * self.DT
        vx = np.clip(vx, -self.MAX_VEL, self.MAX_VEL)
        vy = np.clip(vy, -self.MAX_VEL, self.MAX_VEL)
        x = self.state[0] + vx * self.DT
        y = self.state[1] + vy * self.DT
        x = np.clip(x, -1.0, 1.0)
        y = np.clip(y, -1.0, 1.0)
        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        dist = float(np.linalg.norm(self.state[:2] - self.goal))
        reward = -dist
        done = dist < 0.15
        return self.state.copy(), reward, done

    @property
    def success(self) -> bool:
        return float(np.linalg.norm(self.state[:2] - self.goal)) < 0.15


def scripted_policy(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """PD controller: drives end-effector toward goal."""
    pos = state[:2]
    vel = state[2:]
    error = goal - pos
    kp, kd = 6.0, 2.0
    action = kp * error - kd * vel
    return np.clip(action, -3.0, 3.0).astype(np.float32)


# ── dataset generation ─────────────────────────────────────────────────────────


def generate_episode(
    seed: int,
    goal: np.ndarray,
    corruption_rate: float = 0.0,
    is_redundant: bool = False,
    ref_seed: int = 0,
    n_steps: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one demonstration episode with optional corruption."""
    rng_seed = ref_seed if is_redundant else seed
    rng = np.random.default_rng(rng_seed)
    env = PointMassEnv(goal=goal, seed=rng_seed)

    start = rng.uniform(-0.8, 0.0, size=2).astype(np.float32)
    if is_redundant:
        start += rng.normal(0, 0.03, size=2).astype(np.float32)
    env.reset(start=start)

    states, actions, timestamps = [], [], []
    dt = 0.05

    for t in range(n_steps):
        s = env.state.copy()
        a = scripted_policy(s, goal)

        if corruption_rate > 0 and rng.random() < corruption_rate:
            corrupt_type = rng.integers(3)
            if corrupt_type == 0:
                # Jerk spike
                a += rng.normal(0, 2.5, size=2).astype(np.float32)
            elif corrupt_type == 1:
                # Velocity discontinuity
                a = -a * rng.uniform(0.5, 1.5)
            else:
                # Action dropout (zero action)
                a = np.zeros(2, dtype=np.float32)

        states.append(s)
        actions.append(np.clip(a, -3.0, 3.0))
        timestamps.append(t * dt)
        env.step(a)

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(timestamps, dtype=np.float32),
    )


def build_dataset(
    n_episodes: int = 100,
    corruption_rate: float = 0.0,
    goal: np.ndarray | None = None,
    redundant_fraction: float = 0.20,
) -> "EpisodeBatch":
    """Build an EpisodeBatch with given corruption level."""
    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    if goal is None:
        goal = np.array([0.8, 0.8], dtype=np.float32)

    n_redundant = int(n_episodes * redundant_fraction)
    n_clean = n_episodes - n_redundant
    episodes = []

    for i in range(n_clean):
        states, actions, ts = generate_episode(seed=i, goal=goal, corruption_rate=corruption_rate)
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i:04d}"),
                timestamps=ts,
                observations={"proprio": states},
                actions=actions,
            )
        )

    for i in range(n_redundant):
        states, actions, ts = generate_episode(
            seed=n_clean + i,
            goal=goal,
            corruption_rate=corruption_rate,
            is_redundant=True,
            ref_seed=i % 5,
        )
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"redundant_{i:04d}"),
                timestamps=ts,
                observations={"proprio": states},
                actions=actions,
            )
        )

    return EpisodeBatch(
        episodes=episodes,
        dataset_name=f"pointmass_corrupt_{corruption_rate:.0%}",
        source_path="synthetic",
        format="synthetic",
    )


# ── BC policy ──────────────────────────────────────────────────────────────────


def train_bc(
    batch: "EpisodeBatch",
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> object:
    """Train a behavior cloning MLP on the given EpisodeBatch."""
    import torch
    import torch.nn as nn

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    states_all, actions_all = [], []
    for ep in batch.episodes:
        s = ep.observations.get("proprio")
        a = ep.actions
        if s is not None and len(s) > 1:
            states_all.append(s)
            actions_all.append(a)

    S = torch.from_numpy(np.concatenate(states_all)).to(device)
    A = torch.from_numpy(np.concatenate(actions_all)).to(device)

    s_mean, s_std = S.mean(0), S.std(0).clamp(min=1e-6)
    a_mean, a_std = A.mean(0), A.std(0).clamp(min=1e-6)
    S_n = (S - s_mean) / s_std

    net = nn.Sequential(
        nn.Linear(4, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 2),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = len(S_n)

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, 256):
            idx = perm[i : i + 256]
            pred = net(S_n[idx])
            loss = ((pred - (A[idx] - a_mean) / a_std) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return net, s_mean, s_std, a_mean, a_std, device


def evaluate_bc(
    net,
    s_mean,
    s_std,
    a_mean,
    a_std,
    device,
    goal: np.ndarray,
    n_trials: int = 200,
    ood: bool = False,
) -> float:
    """Evaluate BC policy success rate over random start positions."""
    import torch

    successes = 0
    rng = np.random.default_rng(999)

    for _ in range(n_trials):
        if ood:
            # Out-of-distribution: corners and edges not seen in training
            quadrant = rng.integers(4)
            if quadrant == 0:
                start = rng.uniform([-1.0, 0.5], [-0.5, 1.0])
            elif quadrant == 1:
                start = rng.uniform([0.5, 0.5], [1.0, 1.0])
            elif quadrant == 2:
                start = rng.uniform([-1.0, -1.0], [-0.5, -0.5])
            else:
                start = rng.uniform([0.5, -1.0], [1.0, -0.5])
        else:
            start = rng.uniform(-0.8, 0.0, size=2)

        env = PointMassEnv(goal=goal, seed=int(rng.integers(1_000_000)))
        s = env.reset(start=start.astype(np.float32))

        with torch.no_grad():
            for _ in range(120):
                s_t = torch.from_numpy(s).unsqueeze(0).to(device)
                s_n = (s_t - s_mean) / s_std
                a_n = net(s_n).squeeze(0)
                a = (a_n * a_std + a_mean).cpu().numpy()
                s, _, done = env.step(a)
                if done:
                    successes += 1
                    break

    return successes / n_trials


# ── Calibra quality score ──────────────────────────────────────────────────────


def calibra_quality_score(batch: "EpisodeBatch") -> float:
    """Run Calibra pipeline and return composite quality score (0-100)."""
    from calibra.pipeline import Pipeline
    from calibra.predict import predict_outcome

    report = Pipeline().run(batch)
    result = predict_outcome(report, policy_family="generic", use_outcome_db=False)
    return float(result["predicted_score"])


# ── main experiment ────────────────────────────────────────────────────────────


def run_il_ceiling_experiment():
    import torch  # noqa: F401 — guard early

    print("=" * 65)
    print("  Calibra — IL Ceiling Experiment (Phase 2)")
    print("=" * 65)

    goal = np.array([0.8, 0.8], dtype=np.float32)
    corruption_rates = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
    N_EPISODES = 100
    N_TRIALS_EVAL = 200

    results = []

    for rate in corruption_rates:
        t0 = time.perf_counter()
        print(f"\n[{rate:.0%} corruption]", flush=True)

        # 1. Build dataset
        batch = build_dataset(n_episodes=N_EPISODES, corruption_rate=rate, goal=goal)

        # 2. Calibra quality score (no training needed)
        print("  Computing Calibra quality score ...", end=" ", flush=True)
        quality = calibra_quality_score(batch)
        print(f"{quality:.1f}/100")

        # 3. Train BC on full dataset
        print("  Training BC ...", end=" ", flush=True)
        bc_artifacts = train_bc(batch, n_epochs=120)
        print("done")

        # 4. Evaluate in-distribution
        print("  Evaluating in-dist ...", end=" ", flush=True)
        success_id = evaluate_bc(*bc_artifacts, goal=goal, n_trials=N_TRIALS_EVAL, ood=False)
        print(f"{success_id:.1%}")

        # 5. Evaluate OOD
        print("  Evaluating OOD     ...", end=" ", flush=True)
        success_ood = evaluate_bc(*bc_artifacts, goal=goal, n_trials=N_TRIALS_EVAL, ood=True)
        print(f"{success_ood:.1%}")

        results.append(
            {
                "corruption_rate": rate,
                "calibra_score": quality,
                "bc_success_id": success_id,
                "bc_success_ood": success_ood,
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
        )

    # ── print results table ───────────────────────────────────────────────────
    print("\n")
    print("=" * 65)
    print("  IL CEILING RESULTS")
    print("=" * 65)
    print(f"{'Corruption':>12}  {'Calibra':>8}  {'BC (ID)':>8}  {'BC (OOD)':>10}")
    print("-" * 45)
    for r in results:
        print(
            f"  {r['corruption_rate']:>9.0%}  "
            f"{r['calibra_score']:>7.1f}  "
            f"{r['bc_success_id']:>7.1%}  "
            f"{r['bc_success_ood']:>9.1%}"
        )
    print("=" * 65)

    # ── save figure ───────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        rates = [r["corruption_rate"] * 100 for r in results]
        scores = [r["calibra_score"] for r in results]
        bc_id = [r["bc_success_id"] * 100 for r in results]
        bc_ood = [r["bc_success_ood"] * 100 for r in results]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        ax1.plot(rates, scores, "o--", color="#2563eb", linewidth=2, label="Calibra quality score")
        ax2.plot(rates, bc_id, "s-", color="#16a34a", linewidth=2, label="BC success (in-dist)")
        ax2.plot(rates, bc_ood, "^-", color="#dc2626", linewidth=2, label="BC success (OOD)")

        ax1.set_xlabel("Corruption rate (%)", fontsize=12)
        ax1.set_ylabel("Calibra quality score (0–100)", color="#2563eb", fontsize=11)
        ax2.set_ylabel("BC success rate (%)", fontsize=11)
        ax1.set_ylim(0, 110)
        ax2.set_ylim(0, 110)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

        ax1.set_title(
            "IL Ceiling Experiment — Calibra quality score tracks BC success ceiling",
            fontsize=12,
            pad=10,
        )
        ax1.grid(True, alpha=0.3)

        out = FIG_DIR / "fig_il_ceiling.pdf"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        print(f"\n  Figure saved to {out}")
        plt.close()
    except ImportError:
        print("\n  (matplotlib not installed — skipping figure)")

    return results


# ── coreset selection benchmark ────────────────────────────────────────────────


def run_coreset_benchmark():
    """
    Coreset Selection Benchmark (Phase 3)
    ======================================
    At 0% corruption, vary keep_fraction across [0.10, 0.25, 0.50, 0.75, 1.00].
    For each fraction, compare three strategies:
      - Full dataset (100% baseline)
      - Random subset (5-seed average)
      - Calibra coreset (greedy max-coverage)

    Measures:
      - BC success rate (policy quality)
      - Real training wall-clock seconds (compute cost)

    Key claim: Calibra coreset at K% episodes matches or exceeds the full-data
    baseline while using proportionally fewer compute seconds.
    """
    import random
    import torch  # noqa: F401 — guard early

    from calibra.pipeline import Pipeline
    from calibra.pruning import CoresetSelector
    from calibra.schema.episode import EpisodeBatch

    print("\n" + "=" * 65)
    print("  Calibra — Coreset Selection Benchmark (Phase 3)")
    print("=" * 65)

    goal = np.array([0.8, 0.8], dtype=np.float32)
    N_EPISODES = 200
    N_EPOCHS = 120
    N_TRIALS_EVAL = 200
    KEEP_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
    N_RANDOM_SEEDS = 5

    print(f"\nBuilding clean dataset ({N_EPISODES} episodes, 0% corruption) ...")
    full_batch = build_dataset(n_episodes=N_EPISODES, corruption_rate=0.0, goal=goal)

    print("Running Calibra diagnostics on full dataset ...", flush=True)
    report = Pipeline().run(full_batch)

    # Full-data baseline: train once, record time and success
    print("\n[100% baseline]", flush=True)
    t0 = time.perf_counter()
    full_artifacts = train_bc(full_batch, n_epochs=N_EPOCHS)
    full_train_s = time.perf_counter() - t0
    full_success = evaluate_bc(*full_artifacts, goal=goal, n_trials=N_TRIALS_EVAL)
    print(f"  Success: {full_success:.1%}  |  Train time: {full_train_s:.1f}s")

    results = []

    for frac in KEEP_FRACTIONS:
        k = max(1, round(N_EPISODES * frac))
        print(f"\n[keep {frac:.0%}  ->  {k} episodes]", flush=True)
        row: dict = {"keep_fraction": frac, "n_episodes": k}

        # ── Calibra coreset ──────────────────────────────────────────────────
        # Disable quality filtering: synthetic PD-controller data trips real-robot
        # thresholds (high acceleration at episode start). We want pure diversity
        # selection on already-clean data.
        selector = CoresetSelector(
            keep_fraction=frac,
            max_spike_rate=1.0,
            max_vel_disc_rate=1.0,
            max_dropout_fraction=1.0,
            min_ldlj=-1e6,
        )
        prune_res = selector.select(full_batch, report)
        calibra_eps = [
            ep for ep in full_batch.episodes
            if ep.metadata.episode_id in prune_res.keep_episode_ids
        ]
        calibra_batch = EpisodeBatch(
            episodes=calibra_eps,
            dataset_name=f"calibra_{frac:.0%}",
            format=full_batch.format,
            source_path=full_batch.source_path,
        )

        print("  [Calibra] training ...", end=" ", flush=True)
        t0 = time.perf_counter()
        cal_artifacts = train_bc(calibra_batch, n_epochs=N_EPOCHS)
        cal_train_s = time.perf_counter() - t0
        cal_success = evaluate_bc(*cal_artifacts, goal=goal, n_trials=N_TRIALS_EVAL)
        print(f"success={cal_success:.1%}  time={cal_train_s:.1f}s")
        row["calibra_success"] = cal_success
        row["calibra_train_s"] = round(cal_train_s, 2)

        # ── Random subset (averaged over N_RANDOM_SEEDS seeds) ──────────────
        rand_successes = []
        rand_times = []
        all_ids = [ep.metadata.episode_id for ep in full_batch.episodes]
        for seed in range(N_RANDOM_SEEDS):
            random.seed(seed)
            kept_ids = set(random.sample(all_ids, k))
            rand_eps = [ep for ep in full_batch.episodes if ep.metadata.episode_id in kept_ids]
            rand_batch = EpisodeBatch(
                episodes=rand_eps,
                dataset_name=f"random_{frac:.0%}_s{seed}",
                format=full_batch.format,
                source_path=full_batch.source_path,
            )
            t0 = time.perf_counter()
            rand_artifacts = train_bc(rand_batch, n_epochs=N_EPOCHS)
            rand_times.append(time.perf_counter() - t0)
            rand_successes.append(
                evaluate_bc(*rand_artifacts, goal=goal, n_trials=N_TRIALS_EVAL)
            )
        rand_success = float(np.mean(rand_successes))
        rand_train_s = float(np.mean(rand_times))
        print(
            f"  [Random] success={rand_success:.1%} (±{np.std(rand_successes):.1%})"
            f"  time={rand_train_s:.1f}s"
        )
        row["random_success"] = rand_success
        row["random_success_std"] = float(np.std(rand_successes))
        row["random_train_s"] = round(rand_train_s, 2)

        results.append(row)

    # ── print results table ───────────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print("  CORESET SELECTION BENCHMARK RESULTS")
    print(f"  Full-data baseline: success={full_success:.1%}  train_time={full_train_s:.1f}s")
    print("=" * 80)
    print(
        f"  {'Keep':>6}  {'N':>5}  "
        f"{'Calibra Succ':>13}  {'Calibra Time':>13}  "
        f"{'Random Succ':>12}  {'Random Time':>12}  {'Compute Saved':>14}"
    )
    print("-" * 80)
    for r in results:
        compute_saved = 100.0 * (1.0 - r["keep_fraction"])
        print(
            f"  {r['keep_fraction']:>5.0%}  {r['n_episodes']:>5}  "
            f"  {r['calibra_success']:>10.1%}  "
            f"  {r['calibra_train_s']:>9.1f}s  "
            f"  {r['random_success']:>9.1%}  "
            f"  {r['random_train_s']:>9.1f}s  "
            f"  {compute_saved:>11.0f}%"
        )
    print("=" * 80)

    # ── figure ────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fracs = [r["keep_fraction"] * 100 for r in results]
        cal_succ = [r["calibra_success"] * 100 for r in results]
        rand_succ = [r["random_success"] * 100 for r in results]
        rand_std = [r["random_success_std"] * 100 for r in results]
        full_line = [full_success * 100] * len(fracs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: success rate vs retention
        ax1.axhline(full_success * 100, color="#6b7280", linewidth=1.5,
                    linestyle="--", label="Full data (100%)")
        ax1.plot(fracs, cal_succ, "o-", color="#2563eb", linewidth=2, label="Calibra coreset")
        ax1.errorbar(fracs, rand_succ, yerr=rand_std, fmt="s--", color="#dc2626",
                     linewidth=1.5, capsize=4, label=f"Random (avg {N_RANDOM_SEEDS} seeds)")
        ax1.set_xlabel("Retention fraction (%)", fontsize=12)
        ax1.set_ylabel("BC success rate (%)", fontsize=12)
        ax1.set_title("Policy Quality vs. Dataset Size", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 110)

        # Right: training time vs retention
        cal_times = [r["calibra_train_s"] for r in results]
        rand_times_plot = [r["random_train_s"] for r in results]
        ax2.plot(fracs, cal_times, "o-", color="#2563eb", linewidth=2, label="Calibra coreset")
        ax2.plot(fracs, rand_times_plot, "s--", color="#dc2626", linewidth=1.5, label="Random")
        ax2.axhline(full_train_s, color="#6b7280", linewidth=1.5,
                    linestyle="--", label="Full data (100%)")
        ax2.set_xlabel("Retention fraction (%)", fontsize=12)
        ax2.set_ylabel("Training wall-clock (seconds)", fontsize=12)
        ax2.set_title("Compute Cost vs. Dataset Size", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            "Calibra Coreset Benchmark — Policy Quality & Compute vs. Retention",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        out = FIG_DIR / "fig_coreset_benchmark.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"\n  Figure saved to {out}")
        plt.close()
    except ImportError:
        print("\n  (matplotlib not installed — skipping figure)")

    return {"full_baseline": {"success": full_success, "train_s": full_train_s}, "sweep": results}


# ── correlation analysis ────────────────────────────────────────────────────────


def compute_score_success_correlation(ceiling_results: list[dict]) -> None:
    """
    Computes Pearson and Spearman ρ between Calibra quality score and BC success
    rate using the data already produced by run_il_ceiling_experiment().
    Prints the correlation coefficients — the key single-number paper claim.
    """
    scores = np.array([r["calibra_score"] for r in ceiling_results])
    success_id = np.array([r["bc_success_id"] for r in ceiling_results])
    success_ood = np.array([r["bc_success_ood"] for r in ceiling_results])

    def pearson(x: np.ndarray, y: np.ndarray) -> float:
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())
        return float((x_c * y_c).sum() / denom) if denom > 1e-12 else 0.0

    def spearman(x: np.ndarray, y: np.ndarray) -> float:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return pearson(rx, ry)

    r_id = pearson(scores, success_id)
    rho_id = spearman(scores, success_id)
    r_ood = pearson(scores, success_ood)
    rho_ood = spearman(scores, success_ood)

    print("\n")
    print("=" * 55)
    print("  CALIBRA SCORE <-> BC SUCCESS CORRELATION")
    print("=" * 55)
    print(f"  In-distribution:  Pearson r={r_id:+.3f}  Spearman rho={rho_id:+.3f}")
    print(f"  Out-of-dist:      Pearson r={r_ood:+.3f}  Spearman rho={rho_ood:+.3f}")
    print("=" * 55)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, y, label, r_val, rho_val in [
            (axes[0], success_id * 100, "In-distribution", r_id, rho_id),
            (axes[1], success_ood * 100, "Out-of-distribution", r_ood, rho_ood),
        ]:
            ax.scatter(scores, y, color="#2563eb", s=60, zorder=3)
            m, b = np.polyfit(scores, y, 1)
            x_line = np.linspace(scores.min(), scores.max(), 100)
            ax.plot(x_line, m * x_line + b, "--", color="#dc2626", linewidth=1.5)
            ax.set_xlabel("Calibra quality score", fontsize=11)
            ax.set_ylabel(f"BC success rate — {label} (%)", fontsize=11)
            ax.set_title(f"{label}\nPearson r={r_val:+.3f}  Spearman rho={rho_val:+.3f}", fontsize=11)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Calibra Score vs. Policy Success Correlation", fontsize=13)
        fig.tight_layout()
        out = FIG_DIR / "fig_score_correlation.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Correlation figure saved to {out}")
        plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    ceiling_results = run_il_ceiling_experiment()
    compute_score_success_correlation(ceiling_results)
    run_coreset_benchmark()
