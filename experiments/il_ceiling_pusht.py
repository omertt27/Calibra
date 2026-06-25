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


if __name__ == "__main__":
    run_il_ceiling_experiment()
