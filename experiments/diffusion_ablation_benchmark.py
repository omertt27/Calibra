"""
Diffusion Policy Ablation Benchmark — third architecture generalisation check
=============================================================================
Companion to `act_ablation_benchmark.py`. Same controlled design: reuse the
*identical* selection harness from `ablation_benchmark.py` (same datasets,
splits, seeds, and byte-identical coreset selection), and change ONLY the
downstream learner:

    BC-MLP  /  ACT  ->  Diffusion Policy

The learner here is a state-conditioned **DDPM** in the spirit of Diffusion
Policy (Chi et al., RSS 2023), adapted to low-dimensional state observations:

  - Predicts a chunk of H future actions by iterative denoising (receding horizon).
  - Noise-prediction network eps_theta(noisy_chunk, state, k): an MLP with a
    sinusoidal diffusion-timestep embedding and residual blocks.
  - Training: DDPM epsilon-matching (predict the noise added at step k).
  - Sampling: DDPM ancestral sampling from N(0, I), conditioned on the state.

Evaluation reports **normalised first-action MSE** on the held-out split — the
same quantity the BC and ACT benchmarks report, so `vs_random` numbers are
directly comparable across all three policies. Output schema matches the others,
so `aggregate_ablation.py` consumes it unchanged.

EQUAL-COMPUTE MODE
------------------
`--max-steps N` trains every condition for exactly N optimizer steps regardless
of subset size, instead of `--n-epochs` passes. This answers the reviewer
question "if total training steps are fixed, do the conclusions still hold?" —
most relevant to the coreset-vs-full-dataset comparison (selection methods at a
fixed budget k already use equal steps under the default epoch schedule).

SCOPE / HONESTY
---------------
State-based Diffusion Policy scored on offline first-action prediction error, not
the image-conditioned Diffusion Policy evaluated by simulator rollout. It answers
"does the selection ranking transfer to a diffusion learner," not "what is
Diffusion Policy's task success rate."

    python experiments/diffusion_ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet \\
        --seeds 5 --json results/diffusion_ablation_aloha.json
    python experiments/aggregate_ablation.py results/diffusion_ablation_*.json
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

from experiments.ablation_benchmark import (  # noqa: E402
    _load,
    _obs_key,
    _split,
    _random_subset,
    _run_calibra_pipeline,
    select_kcenter,
    select_herding,
    select_facility,
    select_quality_only,
    select_diversity_only,
    select_calibra_full,
    print_ablation,
    save_ablation_figure,
    _W,
)
# Windows are identical to the ACT benchmark — reuse them so the two experiments
# see exactly the same (obs, action-chunk) construction.
from experiments.act_ablation_benchmark import _act_windows  # noqa: E402


# ── diffusion model ───────────────────────────────────────────────────────────


def _build_diffusion(state_dim, action_dim, chunk, hidden, time_dim):
    import torch
    import torch.nn as nn

    class SinusoidalTime(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, k):  # k: (b,) int
            device = k.device
            half = self.dim // 2
            freqs = torch.exp(
                -np.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1))
            ang = k.float()[:, None] * freqs[None, :]
            return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    class ResBlock(nn.Module):
        def __init__(self, d, cond):
            super().__init__()
            self.fc1 = nn.Linear(d, d)
            self.fc2 = nn.Linear(d, d)
            self.norm = nn.LayerNorm(d)
            self.cond = nn.Linear(cond, d)
            self.act = nn.SiLU()

        def forward(self, x, c):
            h = self.act(self.fc1(x) + self.cond(c))
            h = self.fc2(h)
            return self.act(self.norm(x + h))

    class EpsNet(nn.Module):
        """eps_theta(noisy_chunk, state, k) -> predicted noise over the chunk."""

        def __init__(self):
            super().__init__()
            self.chunk = chunk
            self.action_dim = action_dim
            flat = chunk * action_dim
            self.time_embed = nn.Sequential(
                SinusoidalTime(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
            self.state_embed = nn.Sequential(nn.Linear(state_dim, time_dim), nn.SiLU())
            cond = time_dim * 2
            self.inp = nn.Linear(flat, hidden)
            self.blocks = nn.ModuleList([ResBlock(hidden, cond) for _ in range(4)])
            self.out = nn.Linear(hidden, flat)

        def forward(self, x, state, k):
            b = x.shape[0]
            xf = x.reshape(b, -1)
            c = torch.cat([self.time_embed(k), self.state_embed(state)], dim=-1)
            h = self.inp(xf)
            for blk in self.blocks:
                h = blk(h, c)
            return self.out(h).reshape(b, self.chunk, self.action_dim)

    return EpsNet()


def _ddpm_schedule(n_steps, device):
    import torch
    betas = torch.linspace(1e-4, 0.02, n_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


# ── train / eval ──────────────────────────────────────────────────────────────


def _train_diffusion(batch, cfg, seed=None):
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    OBS, CH, MASK = _act_windows(batch, cfg["chunk"], cfg["stride"],
                                 cfg["max_windows"], seed=(seed or 0))
    state_dim, action_dim = OBS.shape[1], CH.shape[2]
    S = torch.from_numpy(OBS).to(device)
    A = torch.from_numpy(CH).to(device)
    M = torch.from_numpy(MASK).to(device)

    s_mean, s_std = S.mean(0), S.std(0).clamp(min=1e-6)
    flat = A[M]
    a_mean, a_std = flat.mean(0), flat.std(0).clamp(min=1e-6)
    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std

    T = cfg["n_diffusion_steps"]
    _, _, alpha_bars = _ddpm_schedule(T, device)

    net = _build_diffusion(state_dim, action_dim, cfg["chunk"],
                           cfg["hidden"], cfg["time_dim"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    N, bs = len(S_n), cfg["batch_size"]
    net.train()

    def _step(idx):
        x0 = A_n[idx]
        k = torch.randint(0, T, (len(idx),), device=device)
        ab = alpha_bars[k][:, None, None]
        noise = torch.randn_like(x0)
        xk = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        pred = net(xk, S_n[idx], k)
        m = M[idx].unsqueeze(-1)
        loss = (((pred - noise) ** 2) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    max_steps = cfg.get("max_steps")
    if max_steps:  # equal-compute mode: fixed optimizer steps for every condition
        done = 0
        while done < max_steps:
            perm = torch.randperm(N, device=device)
            for i in range(0, N, bs):
                if done >= max_steps:
                    break
                _step(perm[i:i + bs])
                done += 1
    else:
        for _ in range(cfg["n_epochs"]):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, bs):
                _step(perm[i:i + bs])

    return dict(net=net, s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std,
                device=device, cfg=cfg, alpha_bars=alpha_bars)


def _eval_diffusion(art, test_batch):
    """Normalised first-action MSE via DDPM ancestral sampling from the state."""
    import torch

    net, s_mean, s_std, device, cfg, alpha_bars = (
        art[k] for k in ("net", "s_mean", "s_std", "device", "cfg", "alpha_bars"))
    a_mean, a_std = art["a_mean"], art["a_std"]
    OBS, CH, MASK = _act_windows(test_batch, cfg["chunk"], cfg["stride"],
                                 cfg["max_windows"], seed=0)
    S = torch.from_numpy(OBS).to(device)
    A = torch.from_numpy(CH).to(device)
    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std

    T = cfg["n_diffusion_steps"]
    n_samp = int(cfg.get("eval_samples", 1) or 1)
    betas, alphas, abars = _ddpm_schedule(T, device)
    net.eval()
    chunk, adim = A_n.shape[1], A_n.shape[2]
    first_parts = []
    bs = 1024
    g = torch.Generator(device=device).manual_seed(0)

    def _sample_first(s):
        """One DDPM ancestral sample; return the first predicted action (b, adim)."""
        b = s.shape[0]
        x = torch.randn(b, chunk, adim, device=device, generator=g)
        for k in reversed(range(T)):
            kk = torch.full((b,), k, device=device, dtype=torch.long)
            eps = net(x, s, kk)
            ab = abars[k]
            x0 = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            if k > 0:
                ab_prev = abars[k - 1]
                coef_x0 = torch.sqrt(ab_prev) * betas[k] / (1 - ab)
                coef_xt = torch.sqrt(alphas[k]) * (1 - ab_prev) / (1 - ab)
                mean = coef_x0 * x0 + coef_xt * x
                var = betas[k] * (1 - ab_prev) / (1 - ab)
                x = mean + torch.sqrt(var.clamp(min=0)) * torch.randn(
                    b, chunk, adim, device=device, generator=g)
            else:
                x = x0
        return x[:, 0]

    with torch.no_grad():
        for i in range(0, len(S_n), bs):
            s = S_n[i:i + bs]
            # Average the sampled first action over n_samp draws (n_samp=1 -> the
            # single-sample benchmark; n_samp>1 -> sampler-noise robustness check).
            acc = torch.zeros(s.shape[0], adim, device=device)
            for _ in range(n_samp):
                acc += _sample_first(s)
            pred_first = acc / n_samp
            first_parts.append(((pred_first - A_n[i:i + bs, 0]) ** 2).mean(dim=1))
        first_mse = float(torch.cat(first_parts).mean().item())
    return first_mse


# ── ablation run ──────────────────────────────────────────────────────────────


def run_diffusion_ablation(train_batch, test_batch, cfg, keep_fraction=0.30,
                           n_random_seeds=5, only_conditions=None):
    k = max(1, round(len(train_batch.episodes) * keep_fraction))

    print("  Running Calibra pipeline ...", flush=True)
    t0 = time.perf_counter()
    report = _run_calibra_pipeline(train_batch)
    print(f"  Pipeline done in {time.perf_counter()-t0:.1f}s", flush=True)

    conditions = [
        ("Full dataset",        lambda: train_batch),
        ("K-Center greedy",     lambda: select_kcenter(train_batch, k, seed=0)),
        ("Herding",             lambda: select_herding(train_batch, k)),
        ("Facility Location",   lambda: select_facility(train_batch, k)),
        ("Quality-filter only", lambda: select_quality_only(train_batch, report, k)),
        ("Diversity-only",      lambda: select_diversity_only(train_batch, report, k)),
        ("Calibra full",        lambda: select_calibra_full(train_batch, report, k)),
    ]
    # Random is always run (it is the vs-random baseline); --conditions filters the
    # rest, e.g. for the equal-compute check on the key methods only.
    if only_conditions:
        want = {c.strip() for c in only_conditions}
        conditions = [(lbl, fn) for (lbl, fn) in conditions if lbl in want]
    seeds = list(range(n_random_seeds))

    def _mse(sub, s):
        return _eval_diffusion(_train_diffusion(sub, cfg, seed=s), test_batch)

    random_mses = [_mse(_random_subset(train_batch, k, seed=s * 17 + 42), s) for s in seeds]
    random_mean, random_std = float(np.mean(random_mses)), float(np.std(random_mses))

    rows = [{
        "condition": "Random", "n_episodes": k,
        "test_mse": random_mean, "test_mse_std": random_std,
        "per_seed_mse": random_mses, "vs_random": 0.0,
    }]
    print(f"  Random (k={k}, {n_random_seeds} seeds): mse={random_mean:.5f} +/- {random_std:.5f}", flush=True)

    for label, make_subset in conditions:
        sub = make_subset()
        per_seed = [_mse(sub, s) for s in seeds]
        mse, sd = float(np.mean(per_seed)), float(np.std(per_seed))
        delta = (random_mean - mse) / random_mean * 100
        rows.append({
            "condition": label, "n_episodes": len(sub.episodes),
            "test_mse": mse, "test_mse_std": sd,
            "per_seed_mse": per_seed, "vs_random": delta,
        })
        marker = "+++ " if label == "Calibra full" else "    "
        print(f"  {marker}{label:<26} k={len(sub.episodes):>3}  "
              f"mse={mse:.5f}+/-{sd:.5f}  vs_random={delta:+.1f}%", flush=True)
    return rows


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    p = argparse.ArgumentParser(prog="diffusion_ablation_benchmark")
    p.add_argument("--dataset", default="lerobot/aloha_mobile_cabinet")
    p.add_argument("--keep", "-k", type=float, default=0.30)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None,
                   help="equal-compute mode: fixed optimizer steps per condition")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=30000)
    p.add_argument("--n-diffusion-steps", type=int, default=100)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-samples", type=int, default=1,
                   help="average N sampled first-actions per obs (>1 = sampler-noise robustness check)")
    p.add_argument("--conditions", default=None,
                   help="comma-separated subset of conditions to run (Random is always included)")
    p.add_argument("--save-fig", action="store_true")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args(argv)

    cfg = dict(
        chunk=args.chunk, stride=args.stride, max_windows=args.max_windows,
        n_diffusion_steps=args.n_diffusion_steps, hidden=args.hidden,
        time_dim=args.time_dim, lr=args.lr, batch_size=args.batch_size,
        n_epochs=args.n_epochs, max_steps=args.max_steps,
        eval_samples=args.eval_samples,
    )
    only_conditions = args.conditions.split(",") if args.conditions else None

    print("=" * _W)
    print("  CALIBRA DIFFUSION-POLICY ABLATION BENCHMARK  (policy = state-conditioned DDPM)")
    print("=" * _W)
    print(f"  Dataset : {args.dataset}")
    sched = f"max_steps={args.max_steps} (equal-compute)" if args.max_steps else f"epochs={args.n_epochs}"
    print(f"  Keep    : {args.keep:.0%}   Seeds: {args.seeds}   {sched}")
    print(f"  Diff    : chunk={args.chunk} T={args.n_diffusion_steps} hidden={args.hidden}")
    print()

    print("[1/3] Loading dataset ...")
    batch = _load(args.dataset)
    ep0 = batch.episodes[0]
    sk = _obs_key(ep0)
    state_dim = ep0.observations[sk].shape[1] if sk else 0
    print(f"  {batch.n_episodes} episodes  state_dim={state_dim}  action_dim={ep0.actions.shape[1]}")

    print("[2/3] Train/test split (80/20) ...")
    train_batch, test_batch = _split(batch)
    print(f"  train={train_batch.n_episodes}  test={test_batch.n_episodes}")

    print(f"\n[3/3] Running Diffusion ablation (keep={args.keep:.0%}, {args.seeds} seeds) ...")
    rows = run_diffusion_ablation(train_batch, test_batch, cfg,
                                  keep_fraction=args.keep, n_random_seeds=args.seeds,
                                  only_conditions=only_conditions)
    print_ablation(batch.dataset_name + " [Diffusion]", args.keep, rows)

    output = {
        "dataset": args.dataset,
        "policy": "diffusion",
        "diffusion_config": cfg,
        "keep_fraction": args.keep,
        "n_epochs": args.n_epochs,
        "n_seeds": args.seeds,
        "train_episodes": train_batch.n_episodes,
        "test_episodes": test_batch.n_episodes,
        "ablation": rows,
    }

    if args.save_fig:
        save_ablation_figure(batch.dataset_name + "_diffusion", args.keep, rows)

    if args.json:
        out_path = pathlib.Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
