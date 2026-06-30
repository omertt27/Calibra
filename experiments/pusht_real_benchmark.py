"""
Gym-based coreset benchmark on gym_pusht/PushT-v0.

Collects demonstrations from the PushT simulator using a scripted expert,
then trains BC-MLP policies on three conditions and evaluates in the same simulator:
  1. Full collected dataset  (100%)
  2. Calibra 30% coreset
  3. Random 30% baseline

Observation space: 5D state [agent_x, agent_y, block_x, block_y, block_angle].
Training and evaluation use the same observation, so there is no obs mismatch.

Evaluation metric: average coverage (fraction of goal T-zone covered by block,
0–1). This is a continuous signal that shows policy quality even without strict
success (95% threshold). A secondary binary success rate at 50% threshold is
also reported.

Usage:
    pip install 'calibra-robotics[lerobot]' gym-pusht gymnasium "pymunk==6.9.0"
    PYTHONPATH=. python experiments/pusht_real_benchmark.py
"""

import argparse
import random
import time
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = pathlib.Path(__file__).parent.parent
FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_COLLECT_EPISODES = 500
MAX_STEPS_PER_EPISODE = 400
N_EVAL_EPISODES = 100
TRAIN_EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3
KEEP_FRACTION = 0.30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Goal pose is fixed in gym_pusht: block center at (256, 256), angle pi/4
GOAL_POS = np.array([256.0, 256.0])


# ── Scripted expert ───────────────────────────────────────────────────────────


def scripted_expert(obs, rng, noise_std=6.0):
    """
    Alignment-based push controller for PushT.

    On every step, checks whether the agent is already positioned behind the
    block relative to the goal direction. If yes, pushes through toward the
    goal. If not, repositions to the approach point (opposite side from goal).

    This avoids the single-evaluation trap of a two-phase FSM: the agent
    continuously adapts as the block moves.

    obs   : [agent_x, agent_y, block_x, block_y, block_angle]  (pixels, [0, 512])
    action: [target_x, target_y]  (pixels, [0, 512]) — PD position target
    """
    agent_pos = obs[:2]
    block_pos = obs[2:4]

    push_vec = GOAL_POS - block_pos
    push_dist = np.linalg.norm(push_vec)

    if push_dist < 8.0:
        # Block is at goal — hold position
        action = agent_pos.copy()
    else:
        push_norm = push_vec / push_dist

        # Alignment: project agent position relative to block onto push direction.
        # Positive alignment means agent is on the "push" side (behind the block).
        agent_rel = agent_pos - block_pos
        alignment = np.dot(agent_rel, -push_norm)

        if alignment > 15.0:
            # Agent is behind the block — push through.
            # Aim PAST the goal so the agent drives through the block.
            target = GOAL_POS + push_norm * 60.0
            action = np.clip(target, 10.0, 502.0)
        else:
            # Agent is on wrong side or not aligned — reposition.
            approach = block_pos - push_norm * 55.0
            action = np.clip(approach, 10.0, 502.0)

    action = action + rng.normal(0.0, noise_std, size=2)
    return np.clip(action, 0.0, 512.0).astype(np.float32)


# ── Data collection ───────────────────────────────────────────────────────────


def collect_gym_data(n_episodes=N_COLLECT_EPISODES, seed=SEED):
    """
    Run the scripted expert in gym_pusht and return a Calibra EpisodeBatch.

    Uses obs_type='state' (5D) so training and evaluation observations match.
    """
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    rng = np.random.default_rng(seed)
    episodes = []
    total_coverage = 0.0

    print(f"  Collecting {n_episodes} episodes from gym_pusht (scripted expert)...")
    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_obs, ep_acts, ep_ts = [], [], []
        best_coverage = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = scripted_expert(obs, rng)
            ep_obs.append(obs.astype(np.float32))
            ep_acts.append(action)
            ep_ts.append(step / 10.0)

            obs, reward, terminated, truncated, info = env.step(action)
            best_coverage = max(best_coverage, info.get("coverage", 0.0))
            if terminated or truncated:
                break

        total_coverage += best_coverage
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(
                    episode_id=str(ep_idx),
                    source_file="gym_pusht/PushT-v0",
                ),
                timestamps=np.array(ep_ts, dtype=np.float64),
                observations={"state": np.array(ep_obs, dtype=np.float32)},
                actions=np.array(ep_acts, dtype=np.float32),
            )
        )

    env.close()
    avg_cov = total_coverage / n_episodes
    print(f"  Expert avg best coverage: {avg_cov * 100:.1f}%")

    return EpisodeBatch(
        episodes=episodes,
        dataset_name="gym_pusht_scripted",
        format="gym",
        source_path="gym_pusht/PushT-v0",
    )


# ── Calibra coreset selection ─────────────────────────────────────────────────


def get_calibra_coreset(batch, keep_fraction=KEEP_FRACTION):
    """Run Calibra pipeline on the collected EpisodeBatch and return indices."""
    from calibra.pipeline import Pipeline
    from calibra.pruning import CoresetSelector

    print("  Running Calibra pipeline...")
    report = Pipeline().run(batch)
    result = CoresetSelector(keep_fraction=keep_fraction).select(batch, report)
    return sorted(int(e) for e in result.keep_episode_ids)


def get_random_baseline(n_total, keep_fraction=KEEP_FRACTION, seed=SEED):
    rng = random.Random(seed)
    n_keep = round(n_total * keep_fraction)
    return sorted(rng.sample(range(n_total), n_keep))


def get_tensors(batch, indices):
    """Concatenate observations and actions for the given episode indices."""
    states = np.concatenate([batch.episodes[i].observations["state"] for i in indices])
    actions = np.concatenate([batch.episodes[i].actions for i in indices])
    return states, actions


# ── Policy and training ───────────────────────────────────────────────────────


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_policy(states, actions, label):
    obs_dim = states.shape[1]
    act_dim = actions.shape[1]
    policy = BCPolicy(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(
        torch.from_numpy(states).to(DEVICE),
        torch.from_numpy(actions).to(DEVICE),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    t0 = time.perf_counter()
    for _ in range(TRAIN_EPOCHS):
        for s_batch, a_batch in loader:
            optimizer.zero_grad()
            loss_fn(policy(s_batch), a_batch).backward()
            optimizer.step()
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] trained in {elapsed:.1f}s on {DEVICE}")
    return policy, elapsed


# ── Evaluation in PushT simulator ────────────────────────────────────────────


def evaluate_policy(policy, n_episodes=N_EVAL_EPISODES, seed=SEED):
    """
    Evaluate policy in PushT and return (avg_coverage, success_rate_50pct).

    avg_coverage   : mean of per-episode best coverage (0–1, continuous).
    success_rate   : fraction of episodes with best coverage ≥ 0.50.

    Using 50% coverage as the success threshold (rather than the gym's 95%)
    gives a meaningful signal for a non-oracle BC policy trained on scripted
    demonstrations. The avg_coverage metric is the primary headline number.
    """
    try:
        import gym_pusht  # noqa: F401
        import gymnasium as gym
    except ImportError:
        print("  WARNING: gym-pusht not installed.")
        return None, None

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    policy.eval()
    rng = np.random.default_rng(seed)
    coverages = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
            done = False
            best_coverage = 0.0
            while not done:
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(DEVICE)
                action = policy(obs_t).cpu().numpy()[0].clip(0.0, 512.0)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                best_coverage = max(best_coverage, info.get("coverage", 0.0))
            coverages.append(best_coverage)

    env.close()
    coverages = np.array(coverages)
    return float(coverages.mean()), float((coverages >= 0.50).mean())


# ── Main ──────────────────────────────────────────────────────────────────────


def run_single_seed(seed: int, keep_fraction: float = KEEP_FRACTION) -> list[dict]:
    """Run the full collect→select→train→eval pipeline for one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n── Seed {seed} ──────────────────────────────────────────────────────")

    batch = collect_gym_data(N_COLLECT_EPISODES, seed)
    n_total = len(batch.episodes)

    calibra_indices = get_calibra_coreset(batch, keep_fraction)
    random_indices = get_random_baseline(n_total, keep_fraction, seed=seed)
    full_indices = list(range(n_total))

    print(f"  Calibra coreset: {len(calibra_indices)} / {n_total} episodes")

    conditions = [
        ("Full dataset (100%)",                        *get_tensors(batch, full_indices)),
        (f"Calibra {int(keep_fraction*100)}% coreset", *get_tensors(batch, calibra_indices)),
        (f"Random  {int(keep_fraction*100)}% baseline", *get_tensors(batch, random_indices)),
    ]

    results = []
    full_time = None
    for label, states, actions in conditions:
        print(f"\n  [{label}] {len(states):,} steps")
        policy, elapsed = train_policy(states, actions, label)
        avg_cov, sr50 = evaluate_policy(policy, seed=seed)

        if full_time is None:
            full_time = elapsed
        compute_savings = 1.0 - (elapsed / full_time) if full_time else 0.0

        results.append({
            "seed": seed,
            "label": label,
            "n_steps": len(states),
            "train_time_s": elapsed,
            "compute_savings": compute_savings,
            "avg_coverage": avg_cov,
            "success_rate_50": sr50,
        })
        cov_str = f"{avg_cov*100:.1f}%" if avg_cov is not None else "N/A"
        sr_str  = f"{sr50*100:.1f}%"    if sr50  is not None else "N/A"
        print(f"    Avg cov: {cov_str}  SR≥50%: {sr_str}  compute saved: {compute_savings*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Calibra PushT coreset benchmark")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[SEED],
        help="Random seeds to evaluate (default: 42). "
             "Multiple seeds compute mean ± std.",
    )
    parser.add_argument(
        "--keep-fraction", type=float, default=KEEP_FRACTION,
        help=f"Coreset keep fraction (default: {KEEP_FRACTION})",
    )
    args = parser.parse_args()

    print("=== Calibra PushT Coreset Benchmark ===")
    print(f"  Seeds: {args.seeds}  |  Keep fraction: {args.keep_fraction*100:.0f}%%")

    all_results: list[dict] = []
    for seed in args.seeds:
        all_results.extend(run_single_seed(seed, args.keep_fraction))

    # ── Per-seed table ────────────────────────────────────────────────────────
    if len(args.seeds) > 1:
        print("\n" + "=" * 90)
        print("  PER-SEED RESULTS")
        print("=" * 90)
        print(f"  {'Seed':>5}  {'Condition':<35} {'Steps':>7} {'Avg Cov':>9} {'SR>=50%':>8}")
        print("  " + "-" * 68)
        for r in all_results:
            cov = f"{r['avg_coverage']*100:.1f}%" if r["avg_coverage"] is not None else "N/A"
            sr  = f"{r['success_rate_50']*100:.1f}%" if r["success_rate_50"] is not None else "N/A"
            print(f"  {r['seed']:>5}  {r['label']:<35} {r['n_steps']:>7,} {cov:>9} {sr:>8}")

    # ── Aggregate table (mean ± std) ──────────────────────────────────────────
    import collections
    by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for r in all_results:
        by_label[r["label"]].append(r)

    print("\n" + "=" * 90)
    seeds_str = " ".join(str(s) for s in args.seeds)
    print(f"  AGGREGATE RESULTS — {len(args.seeds)} seed(s): [{seeds_str}]")
    print("=" * 90)
    print(f"  {'Condition':<35} {'Steps':>7} {'Avg Cov':>14} {'SR>=50%':>14} {'Compute saved':>14}")
    print("  " + "-" * 86)
    for label, runs in by_label.items():
        steps = runs[0]["n_steps"]
        covs = [r["avg_coverage"] for r in runs if r["avg_coverage"] is not None]
        srs  = [r["success_rate_50"] for r in runs if r["success_rate_50"] is not None]
        saves = [r["compute_savings"] for r in runs]

        cov_str  = (f"{np.mean(covs)*100:.1f}±{np.std(covs)*100:.1f}%"  if covs  else "N/A")
        sr_str   = (f"{np.mean(srs)*100:.1f}±{np.std(srs)*100:.1f}%"   if srs   else "N/A")
        save_str = f"{np.mean(saves)*100:.1f}%"
        print(f"  {label:<35} {steps:>7,} {cov_str:>14} {sr_str:>14} {save_str:>14}")
    print("=" * 90)
    print()
    print("To reproduce:")
    print('  pip install "calibra-robotics[lerobot]" gym-pusht gymnasium "pymunk==6.9.0"')
    print(f"  PYTHONPATH=. python experiments/pusht_real_benchmark.py --seeds {seeds_str}")


if __name__ == "__main__":
    main()
