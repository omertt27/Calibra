"""
Real-data coreset benchmark on lerobot/pusht.

Trains a BC-MLP on three conditions and evaluates in the PushT simulator:
  1. Full dataset  (100%)
  2. Calibra 30% coreset
  3. Random 30% baseline

Usage:
    pip install 'calibra-robotics[lerobot]' lerobot gym-pusht
    PYTHONPATH=. python experiments/pusht_real_benchmark.py

Results replace the synthetic prune_performance_benchmark.py numbers with
a result anyone can independently reproduce from a public dataset.
"""

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
N_EVAL_EPISODES = 100
TRAIN_EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3
KEEP_FRACTION = 0.30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pusht_tensors(episode_indices=None):
    """Load PushT actions and observations as numpy arrays."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/pusht")
    all_episodes = list(range(ds.num_episodes)) if episode_indices is None else episode_indices

    states, actions = [], []
    for ep_idx in all_episodes:
        ep = ds.episode_data_index
        start = ep["from"][ep_idx].item()
        end = ep["to"][ep_idx].item()
        for i in range(start, end):
            item = ds[i]
            obs = item["observation.state"].numpy().astype(np.float32)
            act = item["action"].numpy().astype(np.float32)
            states.append(obs)
            actions.append(act)

    return np.array(states), np.array(actions)


# ── Calibra coreset selection ─────────────────────────────────────────────────

def get_calibra_coreset(keep_fraction=KEEP_FRACTION):
    """Run Calibra pruning on lerobot/pusht and return kept episode indices."""
    from calibra.ingestion.registry import load
    from calibra.pipeline import Pipeline
    from calibra.pruning import CoresetSelector

    print("  Loading dataset into Calibra...")
    batch = load("lerobot/pusht")
    print("  Running pipeline...")
    report = Pipeline().run(batch)
    selector = CoresetSelector(keep_fraction=keep_fraction)
    result = selector.select(batch, report)
    return sorted(result.keep_episode_ids)


def get_random_baseline(n_episodes_total, keep_fraction=KEEP_FRACTION):
    rng = random.Random(SEED)
    n_keep = round(n_episodes_total * keep_fraction)
    return sorted(rng.sample(range(n_episodes_total), n_keep))


# ── Policy and training ───────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
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
    for epoch in range(TRAIN_EPOCHS):
        for s_batch, a_batch in loader:
            optimizer.zero_grad()
            loss_fn(policy(s_batch), a_batch).backward()
            optimizer.step()
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] trained in {elapsed:.1f}s on {DEVICE}")
    return policy, elapsed


# ── Evaluation in PushT simulator ────────────────────────────────────────────

def evaluate_policy(policy, n_episodes=N_EVAL_EPISODES):
    """Run policy in the PushT gym environment and return success rate."""
    try:
        import gym_pusht  # noqa: F401
        import gymnasium as gym
    except ImportError:
        print("  WARNING: gym-pusht not installed. Run: pip install gym-pusht gymnasium")
        print("  Skipping simulator evaluation — install gym-pusht and re-run.")
        return None

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    successes = 0
    policy.eval()
    rng = np.random.default_rng(SEED)

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(1e6)))
            done = False
            while not done:
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(DEVICE)
                action = policy(obs_t).cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if info.get("is_success", False) or reward > 0.9:
                successes += 1

    env.close()
    return successes / n_episodes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=== Calibra Real-Data Coreset Benchmark (lerobot/pusht) ===\n")

    # ── Step 1: get episode splits ────────────────────────────────────────────
    print("Step 1: Computing episode splits...")

    print("  Running Calibra coreset selection...")
    calibra_indices = get_calibra_coreset(KEEP_FRACTION)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    n_total = LeRobotDataset("lerobot/pusht").num_episodes

    random_indices = get_random_baseline(n_total, KEEP_FRACTION)
    full_indices = list(range(n_total))

    print(f"  Total episodes: {n_total}")
    print(f"  Calibra coreset: {len(calibra_indices)} episodes")
    print(f"  Random baseline: {len(random_indices)} episodes")

    # ── Step 2: load tensors for each condition ───────────────────────────────
    print("\nStep 2: Loading data tensors...")
    full_states,    full_actions    = load_pusht_tensors(full_indices)
    calibra_states, calibra_actions = load_pusht_tensors(calibra_indices)
    random_states,  random_actions  = load_pusht_tensors(random_indices)

    conditions = [
        ("Full dataset (100%)",          full_states,    full_actions),
        (f"Calibra {int(KEEP_FRACTION*100)}% coreset", calibra_states, calibra_actions),
        (f"Random  {int(KEEP_FRACTION*100)}% baseline", random_states,  random_actions),
    ]

    # ── Step 3: train and evaluate ────────────────────────────────────────────
    print("\nStep 3: Training policies...")
    results = []
    full_time = None

    for label, states, actions in conditions:
        print(f"\n  Condition: {label} ({len(states)} steps)")
        policy, elapsed = train_policy(states, actions, label)

        print(f"  Evaluating in PushT simulator ({N_EVAL_EPISODES} episodes)...")
        success_rate = evaluate_policy(policy)

        if full_time is None:
            full_time = elapsed
        compute_savings = 1.0 - (elapsed / full_time) if full_time else 0.0

        results.append({
            "label": label,
            "n_steps": len(states),
            "train_time_s": elapsed,
            "compute_savings": compute_savings,
            "success_rate": success_rate,
        })

    # ── Step 4: print results table ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS — lerobot/pusht (real data, public, reproducible)")
    print("=" * 72)
    print(f"  {'Condition':<35} {'Steps':>7} {'Success':>8} {'Compute saved':>14}")
    print("  " + "-" * 68)
    for r in results:
        sr = f"{r['success_rate']*100:.1f}%" if r["success_rate"] is not None else "N/A"
        print(f"  {r['label']:<35} {r['n_steps']:>7} {sr:>8} {r['compute_savings']*100:>13.1f}%")
    print("=" * 72)
    print()
    print("To reproduce:")
    print("  pip install 'calibra-robotics[lerobot]' lerobot gym-pusht")
    print("  PYTHONPATH=. python experiments/pusht_real_benchmark.py")


if __name__ == "__main__":
    main()