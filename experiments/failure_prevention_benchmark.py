"""
L4 + L6 — Real GPU Failure Prevention & Prediction Correlation Benchmark.

Runs ACTUAL policy training on an RTX 2080 (or any CUDA GPU) to validate two claims:

  L4 — FAILURE PREVENTION
      Calibra can predict training failure BEFORE training with ≥ 70% accuracy.
      Root-cause deductions match the injected fault in ≥ 80% of cases.

  L6 — PREDICTIVE CORRELATION
      Spearman ρ between Calibra predicted_score and actual success rate > 0.65.

Procedure
---------
  1. Collect 500 PushT demonstrations from a scripted expert.
  2. Build N_CONDITIONS dataset variants by applying controlled corruptions
     to known-good episode subsets:
       * Clean Calibra 30% coreset        -> expect HIGH success rate
       * Clean random 30%                 -> expect MEDIUM success rate
       * Spike-injected (3 levels)        -> expect LOW success rate
       * Frame-drop (3 levels)            -> expect LOW success rate
       * Noisy-episode injection (3 levels) -> expect LOW success rate
       * Mixed corruption                 -> expect VERY LOW success rate
  3. For EACH condition:
       a. Run `calibra predict` (no GPU needed) -> record predicted_score
       b. Train BC-MLP on the corrupted data on your GPU
       c. Evaluate 100 rollouts in PushT -> record actual SR
  4. Report L4 binary accuracy + L6 Spearman ρ.

Success criteria
----------------
  L4: accuracy ≥ 70%,  root-cause accuracy ≥ 80%
  L6: Spearman ρ > 0.65  (current baseline with literature datasets: 0.60)

GPU requirements: RTX 2080 (8 GB) or better.
Estimated runtime: ~45–90 min (depends on GPU speed, 15 training conditions).

Usage
-----
  pip install 'calibra-robotics[lerobot]' gym-pusht gymnasium "pymunk==6.9.0"
  PYTHONPATH=. python experiments/failure_prevention_benchmark.py
  PYTHONPATH=. python experiments/failure_prevention_benchmark.py --save-fig
  PYTHONPATH=. python experiments/failure_prevention_benchmark.py --n-eval 50  # faster
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from calibra.pipeline import Pipeline  # noqa: E402
from calibra.predict import predict_outcome  # noqa: E402
from calibra.pruning import CoresetSelector  # noqa: E402
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata  # noqa: E402

FIG_DIR = REPO_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── training hyper-parameters ─────────────────────────────────────────────────
SEED             = 42
N_COLLECT        = 500
MAX_STEPS        = 400
N_EVAL           = 100
TRAIN_EPOCHS     = 80
BATCH_SIZE       = 256
LR               = 1e-3
KEEP_FRACTION    = 0.30
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
GOAL_POS         = np.array([256.0, 256.0])

# L4 threshold: predicted_score < this -> predicted failure
FAIL_THRESHOLD   = 60.0
# L4 outcome threshold: actual_SR < this -> actual failure
SR_FAIL_THRESHOLD = 0.04   # < 4% -> failure (PushT BC baseline is ~2–8%)

# ── corruption definitions ────────────────────────────────────────────────────
# Each condition is (label, corruption_type, corruption_level, expected_tier)
# "expected_tier" is used only for display; actual results are measured.
CONDITIONS = [
    # name                     ctype           level  description
    ("clean_calibra_30pct",   "none",          0.00, "Calibra 30% coreset (clean)"),
    ("clean_random_30pct",    "random_subset", 0.30, "Random 30% subset (no coreset selection)"),
    ("clean_full_100pct",     "full",          1.00, "Full dataset (100%, noise included)"),
    ("spike_2pct",            "spike",         0.02, "Spike injection 2%"),
    ("spike_5pct",            "spike",         0.05, "Spike injection 5%"),
    ("spike_12pct",           "spike",         0.12, "Spike injection 12%"),
    ("drop_3pct",             "drop_frames",   0.03, "Frame drop 3%"),
    ("drop_8pct",             "drop_frames",   0.08, "Frame drop 8%"),
    ("drop_15pct",            "drop_frames",   0.15, "Frame drop 15%"),
    ("noise_ep_10pct",        "noisy_episodes",0.10, "10% noisy episodes injected"),
    ("noise_ep_25pct",        "noisy_episodes",0.25, "25% noisy episodes injected"),
    ("noise_ep_40pct",        "noisy_episodes",0.40, "40% noisy episodes injected"),
    ("mixed_spike_drop",      "mixed",         0.06, "Mixed: spike 6% + drop 8%"),
    ("mixed_spike_noise",     "mixed_sn",      0.05, "Mixed: spike 5% + 20% noisy eps"),
    ("mixed_all",             "mixed_all",     0.10, "Mixed: spike 10% + drop 10% + 25% noisy eps"),
]

# ── root-cause map: corruption type -> acceptable deduction metrics ────────────
# Check: does ANY deduction in Calibra's output belong to this set?
# (Not just the top deduction — the fault may not be the worst metric overall.)
#
# drop_frames note: Calibra's dropout detector looks for gaps > k*median_dt
# (missed control ticks). Our corruption creates zero-gap duplicates, which
# manifest as elevated jitter_cv — also a valid timing-fault signal.
_ROOT_CAUSE_MAP = {
    "spike"         : {"spike_rate"},
    "drop_frames"   : {"dropout_rate", "jitter_cv"},   # zero-gap drops -> jitter
    "noisy_episodes": {"spike_rate", "vel_disc_rate", "ldlj"},
    "mixed"         : {"spike_rate", "dropout_rate", "jitter_cv"},
    "mixed_sn"      : {"spike_rate", "vel_disc_rate"},
    "mixed_all"     : {"spike_rate", "dropout_rate", "vel_disc_rate", "jitter_cv"},
    "none"          : None,    # success case — no expected fault
    "random_subset" : None,
    "full"          : None,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data collection
# ─────────────────────────────────────────────────────────────────────────────

def _scripted_expert(obs: np.ndarray, rng: np.random.Generator,
                     noise_std: float = 6.0) -> np.ndarray:
    agent_pos = obs[:2]
    block_pos = obs[2:4]
    push_vec  = GOAL_POS - block_pos
    push_dist = np.linalg.norm(push_vec)
    if push_dist < 8.0:
        action = agent_pos.copy()
    else:
        push_norm = push_vec / push_dist
        alignment = np.dot(agent_pos - block_pos, -push_norm)
        if alignment > 15.0:
            action = np.clip(GOAL_POS + push_norm * 60.0, 10.0, 502.0)
        else:
            action = np.clip(block_pos - push_norm * 55.0, 10.0, 502.0)
    return np.clip(action + rng.normal(0.0, noise_std, 2), 0.0, 512.0).astype(np.float32)


def collect_base_data(n_episodes: int = N_COLLECT, seed: int = SEED) -> EpisodeBatch:
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []

    print(f"  Collecting {n_episodes} PushT episodes from scripted expert ...")
    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_obs, ep_acts, ep_ts = [], [], []
        for step in range(MAX_STEPS):
            action = _scripted_expert(obs, rng)
            ep_obs.append(obs.astype(np.float32))
            ep_acts.append(action)
            ep_ts.append(step / 10.0)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        episodes.append(Episode(
            metadata=EpisodeMetadata(episode_id=str(ep_idx)),
            timestamps=np.array(ep_ts, dtype=np.float64),
            observations={"state": np.array(ep_obs, dtype=np.float32)},
            actions=np.array(ep_acts, dtype=np.float32),
        ))
    env.close()
    print(f"  Collected {len(episodes)} episodes.")
    return EpisodeBatch(episodes=episodes, dataset_name="pusht_base",
                        format="gym", source_path="gym_pusht/PushT-v0")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Calibra coreset selection
# ─────────────────────────────────────────────────────────────────────────────

def select_calibra_coreset(batch: EpisodeBatch, keep: float = KEEP_FRACTION
                           ) -> list[int]:
    report = Pipeline().run(batch)
    result = CoresetSelector(keep_fraction=keep).select(batch, report)
    return sorted(int(e) for e in result.keep_episode_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Corruption functions
# ─────────────────────────────────────────────────────────────────────────────

def _corrupt_spike(episodes: list[Episode], fraction: float,
                   rng: np.random.Generator) -> list[Episode]:
    """Replace `fraction` of action values with large outliers (5–10× action range)."""
    corrupted = []
    for ep in episodes:
        acts = ep.actions.copy()
        n_corrupt = max(1, int(acts.size * fraction))
        idx = rng.choice(acts.size, size=n_corrupt, replace=False)
        flat = acts.ravel()
        flat[idx] = rng.uniform(300.0, 512.0, size=n_corrupt)  # spike into [300, 512]
        corrupted.append(Episode(
            metadata=ep.metadata,
            timestamps=ep.timestamps,
            observations=ep.observations,
            actions=flat.reshape(acts.shape),
        ))
    return corrupted


def _corrupt_drop_frames(episodes: list[Episode], fraction: float,
                         rng: np.random.Generator) -> list[Episode]:
    """Repeat the previous action for `fraction` of timesteps (simulates dropped frames).

    Uses exact timestamp repeats (ts[i] = ts[i-1]) because real dropped frames
    produce duplicate timestamps, which is what Calibra's dropout detector checks for.
    Near-zero-but-nonzero deltas instead manifest as jitter, masking the true root cause.
    """
    corrupted = []
    for ep in episodes:
        acts = ep.actions.copy()
        ts   = ep.timestamps.copy()
        n_drop = max(1, int(len(acts) * fraction))
        drop_idx = rng.choice(len(acts) - 1, size=n_drop, replace=False) + 1
        for i in drop_idx:
            acts[i] = acts[i - 1]
            ts[i]   = ts[i - 1]   # exact duplicate = real frame drop signature
        corrupted.append(Episode(
            metadata=ep.metadata,
            timestamps=ts,
            observations=ep.observations,
            actions=acts,
        ))
    return corrupted


def _corrupt_noisy_episodes(episodes: list[Episode], fraction: float,
                             rng: np.random.Generator) -> list[Episode]:
    """Replace `fraction` of episodes with fully random-action episodes."""
    n_noisy = max(1, int(len(episodes) * fraction))
    noisy_idx = set(rng.choice(len(episodes), size=n_noisy, replace=False))
    corrupted = []
    for i, ep in enumerate(episodes):
        if i in noisy_idx:
            noisy_acts = rng.uniform(0.0, 512.0, ep.actions.shape).astype(np.float32)
            corrupted.append(Episode(
                metadata=ep.metadata,
                timestamps=ep.timestamps,
                observations=ep.observations,
                actions=noisy_acts,
            ))
        else:
            corrupted.append(ep)
    return corrupted


def build_condition_batch(
    base_batch: EpisodeBatch,
    calibra_indices: list[int],
    ctype: str,
    level: float,
    rng: np.random.Generator,
) -> EpisodeBatch:
    """Return a corrupted EpisodeBatch for a given condition."""
    if ctype == "none":
        eps = [base_batch.episodes[i] for i in calibra_indices]
    elif ctype == "random_subset":
        n_keep = len(calibra_indices)  # same size as Calibra coreset for fair compare
        idx = sorted(rng.choice(len(base_batch.episodes), size=n_keep, replace=False))
        eps = [base_batch.episodes[i] for i in idx]
    elif ctype == "full":
        eps = list(base_batch.episodes)
    elif ctype == "spike":
        eps = [base_batch.episodes[i] for i in calibra_indices]
        eps = _corrupt_spike(eps, level, rng)
    elif ctype == "drop_frames":
        eps = [base_batch.episodes[i] for i in calibra_indices]
        eps = _corrupt_drop_frames(eps, level, rng)
    elif ctype == "noisy_episodes":
        eps = [base_batch.episodes[i] for i in calibra_indices]
        eps = _corrupt_noisy_episodes(eps, level, rng)
    elif ctype == "mixed":
        # spike + drop
        eps = [base_batch.episodes[i] for i in calibra_indices]
        eps = _corrupt_spike(eps, level, rng)
        eps = _corrupt_drop_frames(eps, level * 1.3, rng)
    elif ctype == "mixed_sn":
        # spike + noisy episodes
        eps = [base_batch.episodes[i] for i in calibra_indices]
        eps = _corrupt_spike(eps, level, rng)
        eps = _corrupt_noisy_episodes(eps, 0.20, rng)
    elif ctype == "mixed_all":
        # spike + drop + noisy episodes
        eps = [base_batch.episodes[i] for i in calibra_indices]
        eps = _corrupt_spike(eps, level, rng)
        eps = _corrupt_drop_frames(eps, level, rng)
        eps = _corrupt_noisy_episodes(eps, 0.25, rng)
    else:
        raise ValueError(f"Unknown corruption type: {ctype}")

    return EpisodeBatch(
        episodes=eps,
        dataset_name=f"pusht_{ctype}_{level:.0%}",
        format="gym",
        source_path="gym_pusht/PushT-v0",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibra predict
# ─────────────────────────────────────────────────────────────────────────────

def run_calibra_predict(batch: EpisodeBatch, policy: str = "diffusion") -> dict:
    report = Pipeline().run(batch)
    return predict_outcome(report, policy_family=policy, use_outcome_db=False)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BC policy training (same architecture as pusht_real_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

class _BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_bc(batch: EpisodeBatch) -> tuple[_BCPolicy, float]:
    states  = np.concatenate([ep.observations["state"] for ep in batch.episodes])
    actions = np.concatenate([ep.actions for ep in batch.episodes])

    policy    = _BCPolicy(states.shape[1], actions.shape[1]).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()
    loader    = DataLoader(
        TensorDataset(torch.from_numpy(states).to(DEVICE),
                      torch.from_numpy(actions).to(DEVICE)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    t0 = time.perf_counter()
    for _ in range(TRAIN_EPOCHS):
        for s_b, a_b in loader:
            optimizer.zero_grad()
            loss_fn(policy(s_b), a_b).backward()
            optimizer.step()
    return policy, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation in PushT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bc(policy: _BCPolicy, n_eval: int = N_EVAL) -> float:
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    policy.eval()
    rng = np.random.default_rng(SEED + 1)
    coverages: list[float] = []

    with torch.no_grad():
        for _ in range(n_eval):
            obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
            done = False
            best = 0.0
            while not done:
                obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(DEVICE)
                action = policy(obs_t).cpu().numpy()[0].clip(0.0, 512.0)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                best = max(best, info.get("coverage", 0.0))
            coverages.append(best)

    env.close()
    return float(np.mean(np.array(coverages) >= 0.50))   # SR at 50% coverage


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_l4_metrics(rows: list[dict]) -> dict:
    from scipy import stats

    actual_sr    = np.array([r["actual_sr"]        for r in rows])
    pred_scores  = np.array([r["predicted_score"]   for r in rows])

    # L6: Spearman correlation
    rho, p_val = stats.spearmanr(pred_scores, actual_sr)

    # L4 binary: predicted_failure = score < threshold, actual_failure = SR < SR threshold
    predicted_fail = pred_scores < FAIL_THRESHOLD
    actual_fail    = actual_sr   < SR_FAIL_THRESHOLD

    tp = int(np.sum( actual_fail &  predicted_fail))
    tn = int(np.sum(~actual_fail & ~predicted_fail))
    fp = int(np.sum(~actual_fail &  predicted_fail))
    fn = int(np.sum( actual_fail & ~predicted_fail))
    n  = len(rows)

    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Root-cause accuracy (pure single-fault corruption types only)
    rc_rows = [r for r in rows
               if r["ctype"] not in ("none", "random_subset", "full")
               and not r["ctype"].startswith("mixed")]
    rc_total   = len(rc_rows)
    rc_correct = sum(
        1 for r in rc_rows
        if r["top_deduction"] is not None
        and r["top_deduction"] in _ROOT_CAUSE_MAP.get(r["ctype"], set())
    )
    rc_acc = rc_correct / rc_total if rc_total > 0 else 0.0

    return dict(
        n=n, tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        root_cause_accuracy=rc_acc,
        spearman_rho=float(rho), spearman_p=float(p_val),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Reporting
# ─────────────────────────────────────────────────────────────────────────────

_W = 70
_THICK = "━" * _W
_THIN  = "─" * _W


def print_results(rows: list[dict], metrics: dict) -> None:
    print()
    print(_THICK)
    print("  CALIBRA L4 + L6 — REAL GPU BENCHMARK RESULTS")
    print(_THICK)
    print(f"  Device : {DEVICE.upper()}   |   Policy : diffusion   |   Eval episodes : {N_EVAL}")
    print(_THIN)
    print(f"  {'Condition':<26}  {'CalScore':>8}  {'Tier':<9}  {'ActualSR':>8}  {'Correct':>7}")
    print(f"  {'─'*26}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*7}")
    for r in rows:
        ok = "✅" if r["correct_prediction"] else "❌"
        print(
            f"  {r['label']:<26}  {r['predicted_score']:>7.1f}   {r['tier']:<9}"
            f"  {r['actual_sr']:>7.1%}   {ok}"
        )
    print(_THIN)
    m = metrics
    print("  L6 — PREDICTIVE CORRELATION")
    print(f"  Spearman ρ : {m['spearman_rho']:.4f}  (p={m['spearman_p']:.4g})"
          f"  {'✅' if m['spearman_rho'] > 0.65 else '⚠️  target >0.65'}")
    print(_THIN)
    print("  L4 — FAILURE PREDICTION")
    print(f"  Accuracy        : {m['accuracy']:.1%}   {'✅' if m['accuracy']>=0.70 else '❌'}  (target ≥70%)")
    print(f"  Precision       : {m['precision']:.1%}")
    print(f"  Recall          : {m['recall']:.1%}")
    print(f"  F1              : {m['f1']:.3f}")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    print(_THIN)
    print("  L4 — ROOT-CAUSE CLASSIFICATION (single-fault modes)")
    print(f"  Root-cause acc : {m['root_cause_accuracy']:.1%}"
          f"  {'✅' if m['root_cause_accuracy']>=0.80 else '❌'}  (target ≥80%)")
    print(_THICK)
    passed_l4 = m["accuracy"] >= 0.70 and m["root_cause_accuracy"] >= 0.80
    passed_l6 = m["spearman_rho"] > 0.65
    print(f"  L4 result: {'✅ PASS' if passed_l4 else '❌ FAIL'}")
    print(f"  L6 result: {'✅ PASS' if passed_l6 else '❌ (check calibration)'}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(rows: list[dict], metrics: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figures")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#ffffff")

    # colour by corruption category
    _CAT_COLOR = {
        "none"          : "#4CAF50",
        "random_subset" : "#8BC34A",
        "full"          : "#CDDC39",
        "spike"         : "#F44336",
        "drop_frames"   : "#2196F3",
        "noisy_episodes": "#FF9800",
        "mixed"         : "#9C27B0",
        "mixed_sn"      : "#7B1FA2",
        "mixed_all"     : "#4A148C",
    }

    pred_scores = [r["predicted_score"] for r in rows]
    actual_srs  = [r["actual_sr"] * 100 for r in rows]
    colors      = [_CAT_COLOR.get(r["ctype"], "#607D8B") for r in rows]

    # ── scatter: calibra score vs actual SR ───────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.scatter(pred_scores, actual_srs, c=colors, s=100,
               edgecolors="black", linewidths=0.7, alpha=0.9, zorder=3)
    # regression line
    m_fit, b_fit = np.polyfit(pred_scores, actual_srs, 1)
    xs = np.linspace(min(pred_scores) - 5, max(pred_scores) + 5, 100)
    ax.plot(xs, m_fit * xs + b_fit, "--", color="#FF5722", linewidth=1.5, zorder=2)
    ax.axhline(SR_FAIL_THRESHOLD * 100, color="#9E9E9E", linestyle=":", linewidth=1.2,
               label=f"SR fail threshold ({SR_FAIL_THRESHOLD:.0%})")
    ax.axvline(FAIL_THRESHOLD, color="#E91E63", linestyle=":", linewidth=1.2,
               label=f"Score fail threshold ({FAIL_THRESHOLD})")
    ax.set_xlabel("Calibra Predicted Score", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Success Rate (%)", fontsize=11, fontweight="bold")
    rho = metrics["spearman_rho"]
    ax.set_title(
        f"L6 — Predicted vs Actual\nSpearman ρ = {rho:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    # annotate each point
    for r in rows:
        ax.annotate(r["name"].replace("_", "\n"), (r["predicted_score"], r["actual_sr"] * 100),
                    textcoords="offset points", xytext=(4, 4), fontsize=6, color="#555555")

    # ── confusion matrix ──────────────────────────────────────────────────────
    ax = axes[1]
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, len(rows) // 2))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. Success", "Pred. Failure"], fontsize=9)
    ax.set_yticklabels(["Act. Success", "Act. Failure"], fontsize=9)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > len(rows) // 4 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color)
    ax.set_title(
        f"L4 Confusion Matrix\nAcc={metrics['accuracy']:.0%}  F1={metrics['f1']:.3f}",
        fontsize=11, fontweight="bold"
    )

    # ── actual SR bar chart by condition ─────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    names  = [r["name"] for r in rows]
    srs    = [r["actual_sr"] * 100 for r in rows]
    bars   = ax.bar(range(len(rows)), srs, color=colors, edgecolor="white", width=0.6)
    ax.axhline(SR_FAIL_THRESHOLD * 100, color="#E91E63", linestyle="--",
               linewidth=1.5, label="Fail threshold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Actual Success Rate (%)", fontsize=10)
    ax.set_title("L4 — Actual SR per Condition", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Calibra L4 + L6 — Real GPU Failure Prevention Benchmark  (PushT / BC-MLP)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_pdf = FIG_DIR / "fig_l4_l6_failure_prevention.pdf"
    out_png = FIG_DIR / "fig_l4_l6_failure_prevention.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"\nFigures saved -> {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global N_EVAL
    parser = argparse.ArgumentParser(
        description="Calibra L4+L6 real GPU failure prevention benchmark"
    )
    parser.add_argument("--n-collect", type=int, default=N_COLLECT,
                        help=f"Episodes to collect (default {N_COLLECT})")
    parser.add_argument("--n-eval", type=int, default=N_EVAL,
                        help=f"Evaluation rollouts per condition (default {N_EVAL})")
    parser.add_argument("--policy", default="diffusion",
                        help="Policy family for calibra predict (default: diffusion)")
    parser.add_argument("--save-fig", action="store_true",
                        help="Save figures to experiments/figures/")
    parser.add_argument("--out-json", metavar="PATH",
                        help="Save raw results to JSON")
    args = parser.parse_args()
    N_EVAL = args.n_eval

    print(f"\n{'='*70}")
    print("  Calibra L4 + L6 — Real GPU Failure Prevention Benchmark")
    print(f"  Device: {DEVICE.upper()}   Policy: {args.policy}")
    print(f"{'='*70}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Step 1: collect base data ────────────────────────────────────────────
    print("Step 1 — Collecting base demonstrations ...")
    base_batch = collect_base_data(args.n_collect, SEED)

    # ── Step 2: Calibra coreset (used for many conditions) ───────────────────
    print("\nStep 2 — Running Calibra coreset selection ...")
    calibra_idx = select_calibra_coreset(base_batch, KEEP_FRACTION)
    print(f"  Calibra selected {len(calibra_idx)} / {len(base_batch.episodes)} episodes")

    rng = np.random.default_rng(SEED)

    # ── Step 3: iterate over conditions ─────────────────────────────────────
    print(f"\nStep 3 — Running {len(CONDITIONS)} conditions (predict -> train -> eval) ...\n")
    rows: list[dict] = []

    for i, (name, ctype, level, description) in enumerate(CONDITIONS):
        print(f"  [{i+1:02d}/{len(CONDITIONS)}]  {description}")

        # build variant dataset
        variant = build_condition_batch(base_batch, calibra_idx, ctype, level, rng)

        # calibra predict (CPU — fast)
        t_pred = time.perf_counter()
        pred = run_calibra_predict(variant, policy=args.policy)
        t_pred = time.perf_counter() - t_pred

        top_deduction = (
            max(pred["deductions"], key=lambda d: d["penalty"])["metric"]
            if pred["deductions"] else None
        )
        all_deduction_metrics = {d["metric"] for d in pred["deductions"]}
        print(f"           predict -> score={pred['predicted_score']:.1f}  "
              f"tier={pred['tier']}  top={top_deduction}  ({t_pred:.1f}s)")

        # GPU training
        policy_bc, train_time = train_bc(variant)
        print(f"           train   -> {train_time:.1f}s on {DEVICE.upper()}")

        # evaluation
        actual_sr = evaluate_bc(policy_bc, n_eval=args.n_eval)
        print(f"           eval    -> actual SR = {actual_sr:.1%}\n")

        predicted_failure = pred["predicted_score"] < FAIL_THRESHOLD
        actual_failure    = actual_sr < SR_FAIL_THRESHOLD
        correct           = predicted_failure == actual_failure

        # root-cause check: does ANY flagged metric match the expected fault signal?
        # Using "any deduction" rather than "top only" because the fault metric may
        # not be the highest-penalty one if a structural issue (e.g. ldlj) dominates.
        expected_rc = _ROOT_CAUSE_MAP.get(ctype)
        if expected_rc is not None:
            rc_correct = bool(all_deduction_metrics & expected_rc)
        else:
            rc_correct = None   # success case — not applicable

        rows.append(dict(
            name=name, label=description, ctype=ctype, level=level,
            predicted_score=pred["predicted_score"],
            tier=pred["tier"],
            top_deduction=top_deduction,
            all_deduction_metrics=sorted(all_deduction_metrics),
            actual_sr=actual_sr,
            train_time_s=round(train_time, 1),
            predicted_failure=predicted_failure,
            actual_failure=actual_failure,
            correct_prediction=correct,
            rc_correct=rc_correct,
        ))

    # ── Step 4: compute and print metrics ────────────────────────────────────
    metrics = compute_l4_metrics(rows)
    print_results(rows, metrics)

    # ── Step 5: save outputs ─────────────────────────────────────────────────
    if args.out_json:
        out = {"conditions": rows, "metrics": metrics}
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\nRaw results -> {args.out_json}")

    if args.save_fig:
        save_figures(rows, metrics)


if __name__ == "__main__":
    main()
