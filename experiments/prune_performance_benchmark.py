import time
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Import Calibra components
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.pipeline import Pipeline
from calibra.pruning import CoresetSelector

REPO_ROOT = pathlib.Path(__file__).parent.parent
FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Mock Environment and Trajectory Generation ────────────────────────────────

def generate_trajectory(seed, is_corrupted=False, is_redundant=False, redundancy_ref_seed=0):
    """
    Generates a 2D trajectory tracking a circular path.
    If is_corrupted, injects jerk spikes and velocity discontinuities.
    If is_redundant, replicates another trajectory with minimal noise.
    """
    rng = np.random.default_rng(redundancy_ref_seed if is_redundant else seed)
    n_steps = 100
    dt = 0.1
    
    # Clean circular trajectory states
    t = np.linspace(0, 2 * np.pi, n_steps)
    states = np.stack([np.cos(t), np.sin(t)], axis=1)
    
    # Calculate actions (next state - current state)
    actions = np.zeros_like(states)
    actions[:-1] = states[1:] - states[:-1]
    actions[-1] = actions[-2] # boundary condition
    
    # Add noise
    noise_level = 0.005 if is_redundant else 0.02
    states += rng.normal(0, noise_level, size=states.shape)
    actions += rng.normal(0, noise_level, size=actions.shape)
    
    timestamps = np.linspace(0, (n_steps-1)*dt, n_steps)
    
    if is_corrupted:
        # Inject jerk spikes (abrupt action jumps)
        spike_indices = [25, 50, 75]
        for idx in spike_indices:
            actions[idx] += rng.normal(0, 0.6, size=2)
            
        # Inject velocity discontinuity
        actions[10:15] += 0.3
        
    return states.astype(np.float32), actions.astype(np.float32), timestamps


def generate_benchmark_dataset(n_episodes=100):
    """
    Creates a batch of episodes where:
    - 60% are clean, diverse trajectories
    - 20% are near-duplicate redundant trajectories
    - 20% are corrupted (jerk / discontinuity outliers)
    """
    episodes = []
    
    n_clean = int(0.6 * n_episodes)
    n_redundant = int(0.2 * n_episodes)
    n_corrupted = int(0.2 * n_episodes)
    
    # 1. Clean episodes
    for i in range(n_clean):
        states, actions, ts = generate_trajectory(seed=i)
        episodes.append(Episode(
            metadata=EpisodeMetadata(episode_id=f"clean_{i}"),
            timestamps=ts,
            observations={"proprio": states},
            actions=actions
        ))
        
    # 2. Redundant episodes
    for i in range(n_redundant):
        ref_seed = i % 3  # copy one of the first 3 clean episodes
        states, actions, ts = generate_trajectory(seed=100+i, is_redundant=True, redundancy_ref_seed=ref_seed)
        episodes.append(Episode(
            metadata=EpisodeMetadata(episode_id=f"redundant_{i}"),
            timestamps=ts,
            observations={"proprio": states},
            actions=actions
        ))
        
    # 3. Corrupted episodes
    for i in range(n_corrupted):
        states, actions, ts = generate_trajectory(seed=200+i, is_corrupted=True)
        episodes.append(Episode(
            metadata=EpisodeMetadata(episode_id=f"corrupted_{i}"),
            timestamps=ts,
            observations={"proprio": states},
            actions=actions
        ))
        
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="pusht_benchmark",
        format="hdf5",
        source_path="/tmp/pusht_benchmark.h5"
    )

# ── Policy & Training Loop ───────────────────────────────────────────────────

class BehaviorCloningPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.net(x)


def train_policy(episodes, epochs=60, batch_size=64):
    """Train a Behavior Cloning MLP policy on the selected list of episodes."""
    states_all = []
    actions_all = []
    
    for ep in episodes:
        states_all.append(ep.observations["proprio"])
        actions_all.append(ep.actions)
        
    X = torch.tensor(np.concatenate(states_all, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(actions_all, axis=0), dtype=torch.float32)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    policy = BehaviorCloningPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    policy.train()
    start_time = time.perf_counter()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(policy(bx), by)
            loss.backward()
            optimizer.step()
            
    wall_clock = time.perf_counter() - start_time
    return policy, wall_clock


def evaluate_policy(policy, n_rollouts=50):
    """
    Evaluate policy success.
    A rollout is a success if starting from a point, the policy keeps the agent
    on the unit circle (final distance to circle <= 0.15).
    """
    policy.eval()
    successes = 0
    rng = np.random.default_rng(42)
    
    with torch.no_grad():
        for _ in range(n_rollouts):
            # Random starting angle
            angle = rng.uniform(0, 2 * np.pi)
            state = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            
            # Rollout
            for _ in range(20):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = policy(state_tensor).numpy()
                state = state + action # take step
                
            # Check final distance to circle boundary
            dist = np.abs(np.linalg.norm(state) - 1.0)
            if dist <= 0.15:
                successes += 1
                
    return (successes / n_rollouts) * 100.0

# ── Main Experiment ───────────────────────────────────────────────────────────

def main():
    print("Generating synthetic robotics benchmark dataset (100 episodes) ...")
    batch = generate_benchmark_dataset(n_episodes=100)
    
    print("Running Calibra diagnostic pipeline ...")
    report = Pipeline().run(batch)
    
    # 1. Train on 100% full dataset
    print("\nTraining on 100% of episodes...")
    policy_100, t_100 = train_policy(batch.episodes)
    success_100 = evaluate_policy(policy_100)
    print(f"Success rate: {success_100:.1f}%, Training time: {t_100:.3f}s")
    
    # Define splits
    keep_fractions = [0.5, 0.3]
    results = {
        "calibra": {1.0: success_100},
        "random": {1.0: success_100},
        "times_calibra": {1.0: t_100},
        "times_random": {1.0: t_100}
    }
    
    for frac in keep_fractions:
        # A. Calibra Curation
        print(f"\n[Calibra Curation] Pruning to {frac*100:.0f}%...")
        selector = CoresetSelector(
            keep_fraction=frac,
            max_spike_rate=0.03,
            max_vel_disc_rate=0.90,
            min_ldlj=-25.0
        )
        pruned = selector.select(batch, report)
        keep_eps_calibra = [ep for ep in batch.episodes if ep.metadata.episode_id in pruned.keep_episode_ids]
        
        policy_c, t_c = train_policy(keep_eps_calibra)
        success_c = evaluate_policy(policy_c)
        results["calibra"][frac] = success_c
        results["times_calibra"][frac] = t_c
        print(f"Calibra {frac*100:.0f}% -> {len(keep_eps_calibra)} episodes | Success: {success_c:.1f}%, Time: {t_c:.3f}s")
        
        # B. Random Curation
        print(f"[Random Baseline] Sampling {frac*100:.0f}%...")
        rng = np.random.default_rng(42)
        n_keep = max(1, int(len(batch.episodes) * frac))
        keep_indices = rng.choice(len(batch.episodes), size=n_keep, replace=False)
        keep_eps_random = [batch.episodes[i] for i in keep_indices]
        
        policy_r, t_r = train_policy(keep_eps_random)
        success_r = evaluate_policy(policy_r)
        results["random"][frac] = success_r
        results["times_random"][frac] = t_r
        print(f"Random {frac*100:.0f}% -> {len(keep_eps_random)} episodes | Success: {success_r:.1f}%, Time: {t_r:.3f}s")

    # ── Save Figure 5 (Success Curve) ──────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#dddddd")

    fractions_plot = [0.3, 0.5, 1.0]
    calibra_succs = [results["calibra"][f] for f in fractions_plot]
    random_succs = [results["random"][f] for f in fractions_plot]

    ax.plot(np.array(fractions_plot) * 100, calibra_succs, marker="o", color="#2196F3", linewidth=2.5, label="Calibra Curation")
    ax.plot(np.array(fractions_plot) * 100, random_succs, marker="x", color="#FF5722", linestyle="--", linewidth=2.0, label="Random Pruning")

    ax.set_xlabel("Dataset Keep Fraction (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Policy Success Rate (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Coreset Pruning Performance: Calibra vs. Random Curation", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")
    
    plt.xlim(25, 105)
    plt.ylim(30, 105)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_prune_vs_random.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_prune_vs_random.png", bbox_inches="tight", dpi=200)
    print(f"\nFigure saved to {FIG_DIR / 'fig5_prune_vs_random.pdf'}")

    # ── Save Table 2 (LaTeX Table) ───────────────────────────────────────────

    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Policy performance and compute comparison across dataset splits.}}
\\label{{table:prune_performance}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{Condition}} & \\textbf{{Keep \\%}} & \\textbf{{Episodes}} & \\textbf{{Success \\%}} & \\textbf{{Training Time (s)}} & \\textbf{{Compute Saving}} \\\\ \\hline
Full Dataset & 100\\% & 100 & {success_100:.1f}\\% & {t_100:.3f} & Base \\\\ \\hline
Calibra Coreset & 50\\% & 50 & {results["calibra"][0.5]:.1f}\\% & {results["times_calibra"][0.5]:.3f} & {((t_100 - results["times_calibra"][0.5])/t_100)*100:.1f}\\% \\\\
Random Curation & 50\\% & 50 & {results["random"][0.5]:.1f}\\% & {results["times_random"][0.5]:.3f} & {((t_100 - results["times_random"][0.5])/t_100)*100:.1f}\\% \\\\ \\hline
Calibra Coreset & 30\\% & 30 & {results["calibra"][0.3]:.1f}\\% & {results["times_calibra"][0.3]:.3f} & {((t_100 - results["times_calibra"][0.3])/t_100)*100:.1f}\\% \\\\
Random Curation & 30\\% & 30 & {results["random"][0.3]:.1f}\\% & {results["times_random"][0.3]:.3f} & {((t_100 - results["times_random"][0.3])/t_100)*100:.1f}\\% \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(FIG_DIR / "table2_prune_performance.tex", "w") as f:
        f.write(latex_table)
    print(f"LaTeX Table saved to {FIG_DIR / 'table2_prune_performance.tex'}")


if __name__ == "__main__":
    main()
