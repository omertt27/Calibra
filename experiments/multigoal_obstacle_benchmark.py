"""
Multigoal Obstacle Benchmark (Phase 4)
========================================

Environment
-----------
  2D point-mass in [-1, 1]^2.
  Vertical wall at x=0 blocking y in [-0.62, 0.62].
  Passages: top gap (y > 0.62) and bottom gap (y < -0.62).
  Start: random in left half (x in [-0.9, -0.2]).
  Goals:
    A: ( 0.8,  0.75)  -> upper-right  (natural route: top gap)
    B: ( 0.8, -0.75)  -> lower-right  (natural route: bottom gap)

  Two distinct trajectory modes arise naturally. Random subsets may miss one
  mode; diversity-aware coresets should preserve both.

Experiments
-----------
  1. Diagnostic validity sweep (separate corruption families):
       Each metric should respond monotonically to its target corruption.

  2. Coreset benchmark (multigoal):
       Vary keep_fraction; compare random vs Calibra by per-goal success.
       Key claim: Calibra preserves rare modes at low retention.

Run
---
    pip install calibra-robotics torch matplotlib
    python experiments/multigoal_obstacle_benchmark.py
"""

from __future__ import annotations

import pathlib
import random as _random
import sys
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from calibra.schema.episode import EpisodeBatch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── environment constants ──────────────────────────────────────────────────────

WALL_X = 0.0
WALL_GAP = 0.62  # |y| > WALL_GAP = open passage

GOALS = [
    np.array([0.8, 0.75], dtype=np.float32),  # A: upper-right
    np.array([0.8, -0.75], dtype=np.float32),  # B: lower-right
]
GOAL_NAMES = ["A:upper", "B:lower"]

# Waypoints are placed PAST the wall (x=0.10) inside the gap region.
# This guarantees the crossing happens at y > WALL_GAP and avoids
# the agent oscillating on the left side after overshooting.
_UPPER_WP = np.array([0.10, 0.88], dtype=np.float32)
_LOWER_WP = np.array([0.10, -0.88], dtype=np.float32)


# ── environment ────────────────────────────────────────────────────────────────


class MultiGoalObstacleEnv:
    DT = 0.05
    MAX_VEL = 2.0
    DAMPING = 0.85
    SUCCESS_RADIUS = 0.15

    def __init__(self, goal: np.ndarray, seed: int = 0):
        self.goal = goal.copy()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float32)

    def reset(self, start: Optional[np.ndarray] = None) -> np.ndarray:
        if start is not None:
            self.state = np.array([start[0], start[1], 0.0, 0.0], dtype=np.float32)
        else:
            x = self.rng.uniform(-0.9, -0.2)
            y = self.rng.uniform(-0.85, 0.85)
            self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        ax, ay = float(action[0]), float(action[1])
        vx = self.state[2] * self.DAMPING + ax * self.DT
        vy = self.state[3] * self.DAMPING + ay * self.DT
        vx = np.clip(vx, -self.MAX_VEL, self.MAX_VEL)
        vy = np.clip(vy, -self.MAX_VEL, self.MAX_VEL)

        ox, oy = self.state[0], self.state[1]
        nx = float(np.clip(ox + vx * self.DT, -1.0, 1.0))
        ny = float(np.clip(oy + vy * self.DT, -1.0, 1.0))

        if self._wall_collision(ox, oy, nx, ny):
            nx = ox  # block x-crossing
            vx = -vx * 0.3  # soft bounce

        self.state = np.array([nx, ny, vx, vy], dtype=np.float32)
        dist = float(np.linalg.norm(self.state[:2] - self.goal))
        done = dist < self.SUCCESS_RADIUS
        return self.state.copy(), -dist, done

    def _wall_collision(self, ox: float, oy: float, nx: float, ny: float) -> bool:
        if ox * nx >= 0:
            return False
        t = -ox / (nx - ox + 1e-12)
        y_cross = oy + t * (ny - oy)
        return abs(y_cross) <= WALL_GAP

    @property
    def success(self) -> bool:
        return float(np.linalg.norm(self.state[:2] - self.goal)) < self.SUCCESS_RADIUS


# ── scripted policy ────────────────────────────────────────────────────────────


def _route_waypoint(goal: np.ndarray) -> np.ndarray:
    return _UPPER_WP if goal[1] > 0 else _LOWER_WP


def _scripted(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = target - state[:2]
    action = 5.5 * error - 1.5 * state[2:]
    return np.clip(action, -3.0, 3.0).astype(np.float32)


# ── episode generation ─────────────────────────────────────────────────────────


def generate_episode(
    seed: int,
    goal: np.ndarray,
    *,
    n_steps: int = 140,
    dropout_rate: float = 0.0,
    spike_rate: float = 0.0,
    truncate_fraction: float = 1.0,
    obs_lag_steps: int = 0,
    perturbation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (states, actions, timestamps, goals_tiled)."""
    rng = np.random.default_rng(seed)
    env = MultiGoalObstacleEnv(goal=goal, seed=seed)
    x = rng.uniform(-0.9, -0.2)
    y = rng.uniform(-0.85, 0.85)
    env.reset(start=np.array([x, y], dtype=np.float32))

    wp = _route_waypoint(goal)
    perturb_step = int(n_steps * rng.uniform(0.25, 0.45)) if perturbation else -1
    wp_reached = False  # latch: commit to goal once waypoint is first passed

    raw_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for t in range(n_steps):
        s = env.state.copy()

        if t == perturb_step:
            env.state[0] = float(rng.uniform(0.2, 0.8))
            s = env.state.copy()
            wp_reached = False  # re-route after perturbation

        if not wp_reached and np.linalg.norm(s[:2] - wp) < 0.18:
            wp_reached = True

        target = goal if wp_reached else wp
        a = _scripted(s, target)

        if dropout_rate > 0 and rng.random() < dropout_rate:
            a = np.zeros(2, dtype=np.float32)
        elif spike_rate > 0 and rng.random() < spike_rate:
            a = np.clip(a + rng.normal(0, 2.5, 2).astype(np.float32), -3.0, 3.0)

        raw_states.append(s)
        actions.append(a)
        env.step(a)

        if env.success:
            for t2 in range(t + 1, n_steps):
                raw_states.append(env.state.copy())
                hover = _scripted(env.state, goal)
                actions.append(hover)
                env.step(hover)
            break

    actual = max(15, int(len(raw_states) * truncate_fraction))
    raw_states = raw_states[:actual]
    actions = actions[:actual]

    states_arr = np.array(raw_states, dtype=np.float32)
    actions_arr = np.array(actions, dtype=np.float32)
    ts = np.arange(len(states_arr), dtype=np.float32) * env.DT

    # Observation lag: shift observed state by lag_steps
    if obs_lag_steps > 0 and len(states_arr) > obs_lag_steps:
        observed = np.concatenate([states_arr[:obs_lag_steps], states_arr[:-obs_lag_steps]], axis=0)
    else:
        observed = states_arr

    goals_tiled = np.tile(goal, (len(states_arr), 1))
    return observed, actions_arr, ts, goals_tiled


def build_dataset(
    n_per_goal: int = 80,
    *,
    dropout_rate: float = 0.0,
    spike_rate: float = 0.0,
    truncate_fraction: float = 1.0,
    obs_lag_steps: int = 0,
    perturbation_rate: float = 0.0,
    duplicate_fraction: float = 0.0,
    mode_delete: Optional[int] = None,  # delete goal index
    seed_offset: int = 0,
    dataset_name: str = "multigoal",
) -> "EpisodeBatch":
    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    episodes = []
    ep_idx = 0

    goals_to_use = [
        (g, gn)
        for i, (g, gn) in enumerate(zip(GOALS, GOAL_NAMES))
        if mode_delete is None or i != mode_delete
    ]

    for goal, gname in goals_to_use:
        n_regular = max(1, int(n_per_goal * (1.0 - perturbation_rate)))
        n_recovery = n_per_goal - n_regular

        for i in range(n_regular):
            s, a, ts, g = generate_episode(
                seed=seed_offset + ep_idx,
                goal=goal,
                dropout_rate=dropout_rate,
                spike_rate=spike_rate,
                truncate_fraction=truncate_fraction,
                obs_lag_steps=obs_lag_steps,
            )
            obs = np.concatenate([s, g], axis=1)  # 6-D: (x,y,vx,vy,gx,gy)
            episodes.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"{gname}_ep{ep_idx:04d}"),
                    timestamps=ts,
                    observations={"proprio": obs},
                    actions=a,
                )
            )
            ep_idx += 1

        for i in range(n_recovery):
            s, a, ts, g = generate_episode(
                seed=seed_offset + ep_idx,
                goal=goal,
                dropout_rate=dropout_rate,
                spike_rate=spike_rate,
                truncate_fraction=truncate_fraction,
                obs_lag_steps=obs_lag_steps,
                perturbation=True,
            )
            obs = np.concatenate([s, g], axis=1)
            episodes.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"{gname}_recovery{ep_idx:04d}"),
                    timestamps=ts,
                    observations={"proprio": obs},
                    actions=a,
                )
            )
            ep_idx += 1

    # Duplicate injection: replace a fraction of episodes with near-copies
    if duplicate_fraction > 0.0 and len(episodes) > 2:
        n_dups = int(len(episodes) * duplicate_fraction)
        rng = np.random.default_rng(seed_offset + 99999)
        src_eps = list(episodes)
        for i in range(n_dups):
            src = src_eps[int(rng.integers(len(src_eps)))]
            obs_n = src.observations["proprio"] + rng.normal(
                0, 0.01, src.observations["proprio"].shape
            ).astype(np.float32)
            a_n = np.clip(
                src.actions + rng.normal(0, 0.02, src.actions.shape).astype(np.float32), -3.0, 3.0
            )
            episodes.append(
                Episode(
                    metadata=EpisodeMetadata(episode_id=f"dup_{ep_idx:04d}"),
                    timestamps=src.timestamps.copy(),
                    observations={"proprio": obs_n},
                    actions=a_n,
                )
            )
            ep_idx += 1

    return EpisodeBatch(
        episodes=episodes,
        dataset_name=dataset_name,
        format="synthetic",
        source_path="synthetic",
    )


# ── goal-conditioned BC ────────────────────────────────────────────────────────


def train_bc(batch: "EpisodeBatch", n_epochs: int = 150, lr: float = 1e-3):
    import torch
    import torch.nn as nn

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    S_list, A_list = [], []
    for ep in batch.episodes:
        obs = ep.observations.get("proprio")
        if obs is not None and len(obs) > 1:
            S_list.append(obs)
            A_list.append(ep.actions)

    if not S_list:
        return None

    S = torch.from_numpy(np.concatenate(S_list)).to(device)
    A = torch.from_numpy(np.concatenate(A_list)).to(device)

    s_mean, s_std = S.mean(0), S.std(0).clamp(min=1e-6)
    a_mean, a_std = A.mean(0), A.std(0).clamp(min=1e-6)
    S_n = (S - s_mean) / s_std

    net = nn.Sequential(
        nn.Linear(S.shape[1], 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, A.shape[1]),
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


def evaluate_bc_per_goal(
    artifacts,
    *,
    n_trials: int = 150,
    n_steps: int = 200,
) -> dict[str, float]:
    """Returns per-goal success rate and overall rate."""
    if artifacts is None:
        return {gn: 0.0 for gn in GOAL_NAMES} | {"overall": 0.0}

    import torch

    net, s_mean, s_std, a_mean, a_std, device = artifacts
    rng = np.random.default_rng(777)
    results = {gn: 0 for gn in GOAL_NAMES}

    for trial in range(n_trials):
        goal_idx = trial % len(GOALS)
        goal = GOALS[goal_idx]
        gname = GOAL_NAMES[goal_idx]
        env = MultiGoalObstacleEnv(goal=goal, seed=int(rng.integers(1_000_000)))

        x = rng.uniform(-0.9, -0.2)
        y = rng.uniform(-0.85, 0.85)
        s = env.reset(start=np.array([x, y], dtype=np.float32))

        with torch.no_grad():
            for _ in range(n_steps):
                obs = np.concatenate([s, goal])
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                obs_n = (obs_t - s_mean) / s_std
                a_n = net(obs_n).squeeze(0)
                a = (a_n * a_std + a_mean).cpu().numpy()
                s, _, done = env.step(a)
                if done:
                    results[gname] += 1
                    break

    per_goal = {gn: results[gn] / (n_trials // len(GOALS)) for gn in GOAL_NAMES}
    per_goal["overall"] = sum(results.values()) / n_trials
    return per_goal


# ── experiment 1: diagnostic validity ─────────────────────────────────────────


def run_diagnostic_validity():
    """
    Each corruption family is swept separately.
    The primary question: does the corresponding Calibra metric respond
    monotonically to its target corruption?
    """
    from calibra.pipeline import Pipeline
    from calibra.score import compute_score

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC VALIDITY SWEEP")
    print("=" * 70)

    families = {
        "dropout": {"dropout_rate": [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]},
        "spikes": {"spike_rate": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]},
        "truncation": {"truncate_fraction": [1.0, 0.80, 0.60, 0.40, 0.25, 0.10]},
        "obs_lag": {"obs_lag_steps": [0, 1, 2, 4, 8, 16]},
        "duplicates": {"duplicate_fraction": [0.0, 0.10, 0.30, 0.60, 0.90]},
    }

    all_results = {}

    for family_name, param_dict in families.items():
        param_key = list(param_dict.keys())[0]
        severities = list(param_dict.values())[0]

        print(f"\n  [{family_name.upper()} SWEEP]")
        print(
            f"  {'Severity':>10}  {'Temporal':>9}  {'Smooth':>9}  {'Coverage':>9}  {'Structure':>10}  {'Gate':>6}  {'Total':>7}"
        )
        print("  " + "-" * 70)

        rows = []
        for sev in severities:
            kwargs: dict = {param_key: sev}
            batch = build_dataset(n_per_goal=60, dataset_name=f"{family_name}_{sev}", **kwargs)
            report = Pipeline().run(batch)
            result = compute_score(report)

            t = result["dimensions"]["temporal_stability"]["score"]
            s = result["dimensions"]["control_smoothness"]["score"]
            c = result["dimensions"]["coverage_diversity"]["score"]
            st = result["dimensions"]["task_structure"]["score"]
            gate = result["integrity_gate"]
            total = result["total_score"]

            print(
                f"  {sev:>10.3g}  {t:>9.2f}  {s:>9.2f}  {c:>9.2f}  {st:>10.2f}  {gate:>6.3f}  {total:>7.1f}"
            )
            rows.append(
                {
                    "severity": sev,
                    "temporal": t,
                    "smoothness": s,
                    "coverage": c,
                    "structure": st,
                    "gate": gate,
                    "total": total,
                }
            )

        all_results[family_name] = rows

    return all_results


# ── experiment 2: coreset benchmark ───────────────────────────────────────────


def run_coreset_benchmark_multigoal(n_seeds: int = 5):
    """
    Coreset benchmark on an IMBALANCED multigoal dataset.

    Dataset: 140 episodes of Goal A (dominant) + 20 episodes of Goal B (rare).
    At 5-10% retention, random selection has ~11-35% chance of dropping ALL
    Goal B episodes, degrading worst-group success.  Calibra diversity selection
    preserves behaviorally distant episodes (Goal B lives in a different
    region of action space) even under extreme compression.

    Key metric: worst-group success rate (min over goals), not just overall.
    """
    import torch  # noqa: F401 — guard early

    from calibra.pipeline import Pipeline
    from calibra.pruning import CoresetSelector
    from calibra.schema.episode import EpisodeBatch

    print("\n" + "=" * 70)
    print("  CORESET BENCHMARK (MULTIGOAL — IMBALANCED 7:1)")
    print("=" * 70)

    N_EPOCHS = 150
    N_EVAL = 150
    KEEP_FRACS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

    N_A, N_B = 140, 20  # 7:1 imbalance
    print(f"\nBuilding imbalanced dataset ({N_A} Goal-A, {N_B} Goal-B) ...")
    full_batch = build_dataset(
        n_per_goal=0,  # override below
        dataset_name="multigoal_imbalanced",
    )

    # Build manually with the imbalanced split
    from calibra.schema.episode import Episode, EpisodeMetadata
    from calibra.schema.episode import EpisodeBatch as _EB

    all_eps = []
    for i in range(N_A):
        s, a, ts, g = generate_episode(seed=i, goal=GOALS[0])
        obs = np.concatenate([s, g], axis=1)
        all_eps.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"A:upper_ep{i:04d}"),
                timestamps=ts,
                observations={"proprio": obs},
                actions=a,
            )
        )
    for i in range(N_B):
        s, a, ts, g = generate_episode(seed=1000 + i, goal=GOALS[1])
        obs = np.concatenate([s, g], axis=1)
        all_eps.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"B:lower_ep{i:04d}"),
                timestamps=ts,
                observations={"proprio": obs},
                actions=a,
            )
        )
    full_batch = _EB(all_eps, "multigoal_imbalanced", "synthetic", "synthetic")
    n_total = full_batch.n_episodes

    print("Running Calibra diagnostics ...")
    report = Pipeline().run(full_batch)

    print("\n[100% baseline]", flush=True)
    t0 = time.perf_counter()
    full_arts = train_bc(full_batch, n_epochs=N_EPOCHS)
    full_time = time.perf_counter() - t0
    full_res = evaluate_bc_per_goal(full_arts, n_trials=N_EVAL)
    print(
        f"  overall={full_res['overall']:.1%}  A={full_res['A:upper']:.1%}  B={full_res['B:lower']:.1%}  time={full_time:.1f}s"
    )

    sweep_results = []

    for frac in KEEP_FRACS:
        k = max(1, round(n_total * frac))
        print(f"\n[keep {frac:.0%} -> {k} episodes]", flush=True)
        row = {"frac": frac, "k": k}

        # ── Calibra coreset ──────────────────────────────────────────────────
        selector = CoresetSelector(
            keep_fraction=frac,
            max_spike_rate=1.0,
            max_vel_disc_rate=1.0,
            max_dropout_fraction=1.0,
            min_ldlj=-1e6,
        )
        prune = selector.select(full_batch, report)
        keep = set(prune.keep_episode_ids)
        cal_ep = [ep for ep in full_batch.episodes if ep.metadata.episode_id in keep]
        cal_batch = EpisodeBatch(cal_ep, f"calibra_{frac:.0%}", "synthetic", "synthetic")

        t0 = time.perf_counter()
        cal_arts = train_bc(cal_batch, n_epochs=N_EPOCHS)
        cal_time = time.perf_counter() - t0
        cal_res = evaluate_bc_per_goal(cal_arts, n_trials=N_EVAL)

        # Count goal coverage
        cal_goals = {
            gn: sum(1 for ep in cal_ep if gn.split(":")[0] in ep.metadata.episode_id)
            for gn in GOAL_NAMES
        }

        print(
            f"  [Calibra] overall={cal_res['overall']:.1%}  A={cal_res['A:upper']:.1%}  B={cal_res['B:lower']:.1%}  time={cal_time:.1f}s  goals={cal_goals}"
        )
        row["cal_overall"] = cal_res["overall"]
        row["cal_A"] = cal_res["A:upper"]
        row["cal_B"] = cal_res["B:lower"]
        row["cal_worst"] = min(cal_res["A:upper"], cal_res["B:lower"])
        row["cal_time"] = round(cal_time, 2)

        # ── Random baseline (averaged over n_seeds) ───────────────────────────
        rand_overalls, rand_As, rand_Bs, rand_times = [], [], [], []
        all_ids = [ep.metadata.episode_id for ep in full_batch.episodes]
        for seed in range(n_seeds):
            _random.seed(seed)
            kept = set(_random.sample(all_ids, k))
            r_ep = [ep for ep in full_batch.episodes if ep.metadata.episode_id in kept]
            r_batch = EpisodeBatch(r_ep, f"rand_{frac:.0%}_s{seed}", "synthetic", "synthetic")
            t0 = time.perf_counter()
            r_arts = train_bc(r_batch, n_epochs=N_EPOCHS)
            rand_times.append(time.perf_counter() - t0)
            r_res = evaluate_bc_per_goal(r_arts, n_trials=N_EVAL)
            rand_overalls.append(r_res["overall"])
            rand_As.append(r_res["A:upper"])
            rand_Bs.append(r_res["B:lower"])

        row["rand_overall"] = float(np.mean(rand_overalls))
        row["rand_overall_std"] = float(np.std(rand_overalls))
        row["rand_A"] = float(np.mean(rand_As))
        row["rand_B"] = float(np.mean(rand_Bs))
        row["rand_worst"] = float(np.mean([min(a, b) for a, b in zip(rand_As, rand_Bs)]))
        row["rand_time"] = float(np.mean(rand_times))

        print(
            f"  [Random]  overall={row['rand_overall']:.1%}+/-{row['rand_overall_std']:.1%}  A={row['rand_A']:.1%}  B={row['rand_B']:.1%}  time={row['rand_time']:.1f}s"
        )
        sweep_results.append(row)

    # ── print summary table ───────────────────────────────────────────────────
    print("\n")
    print("=" * 85)
    print("  CORESET BENCHMARK SUMMARY")
    print(
        f"  Full baseline: overall={full_res['overall']:.1%}  A={full_res['A:upper']:.1%}  B={full_res['B:lower']:.1%}"
    )
    print("=" * 85)
    print(
        f"  {'Keep':>5}  {'N':>4}  {'Cal Overall':>12}  {'Cal Worst':>10}  {'Rand Overall':>13}  {'Rand Worst':>11}  {'Saved':>6}"
    )
    print("-" * 85)
    for r in sweep_results:
        saved = 100 * (1 - r["frac"])
        print(
            f"  {r['frac']:>4.0%}  {r['k']:>4}  "
            f"  {r['cal_overall']:>9.1%}  {r['cal_worst']:>9.1%}  "
            f"  {r['rand_overall']:>10.1%}  {r['rand_worst']:>10.1%}  "
            f"  {saved:>5.0f}%"
        )
    print("=" * 85)

    _save_coreset_figure(full_res, sweep_results)
    return {"full_baseline": full_res, "sweep": sweep_results}


def _save_coreset_figure(full_res: dict, sweep_results: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt

        fracs = [r["frac"] * 100 for r in sweep_results]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = [
            ("cal_overall", "rand_overall", "Overall Success Rate (%)", "Overall"),
            ("cal_worst", "rand_worst", "Worst-Group Success Rate (%)", "Worst-Group (min goal)"),
        ]

        for ax, (cal_key, rand_key, ylabel, title) in zip(axes[:2], metrics):
            cal_vals = [r[cal_key] * 100 for r in sweep_results]
            rand_vals = [r[rand_key] * 100 for r in sweep_results]

            ax.axhline(
                full_res["overall"] * 100, color="#6b7280", lw=1.5, ls="--", label="Full data"
            )
            ax.plot(fracs, cal_vals, "o-", color="#2563eb", lw=2, label="Calibra coreset")
            ax.plot(fracs, rand_vals, "s--", color="#dc2626", lw=1.5, label="Random (avg)")
            ax.set_xlabel("Retention fraction (%)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 110)

        # Per-goal bar chart at 20% retention
        ax = axes[2]
        r20 = next((r for r in sweep_results if abs(r["frac"] - 0.20) < 0.05), sweep_results[0])
        goals_x = np.arange(len(GOAL_NAMES))
        w = 0.3
        cal_per = [r20["cal_A"] * 100, r20["cal_B"] * 100]
        rand_per = [r20["rand_A"] * 100, r20["rand_B"] * 100]
        full_per = [full_res["A:upper"] * 100, full_res["B:lower"] * 100]
        ax.bar(goals_x - w, cal_per, width=w, color="#2563eb", label="Calibra 20%")
        ax.bar(goals_x, rand_per, width=w, color="#dc2626", label="Random 20%")
        ax.bar(goals_x + w, full_per, width=w, color="#6b7280", label="Full data")
        ax.set_xticks(goals_x)
        ax.set_xticklabels(GOAL_NAMES, fontsize=10)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_title("Per-Goal Success at 20% Retention", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 110)

        fig.suptitle("Multigoal Obstacle Coreset Benchmark", fontsize=13)
        fig.tight_layout()
        out = FIG_DIR / "fig_multigoal_coreset.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"\n  Figure saved to {out}")
        plt.close()
    except ImportError:
        print("\n  (matplotlib not installed -- skipping figure)")


# ── entry point ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import torch  # noqa: F401 -- guard early

    run_diagnostic_validity()
    run_coreset_benchmark_multigoal(n_seeds=5)
