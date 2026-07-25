"""
ACT Ablation Benchmark — does coreset selection survive a different policy?
===========================================================================
The BC-MLP ablation (`ablation_benchmark.py`) answers "which selection method is
best." This script answers the *generalisation* question a reviewer asks first:

    "Do those findings still hold when the downstream learner is ACT, not a BC-MLP?"

To make that a clean controlled experiment, this benchmark **reuses the exact
selection harness** from `ablation_benchmark.py` — same datasets, same train/test
split, same seeds, same coreset conditions (Random, K-Center, Herding, Facility
Location, Quality-only, Diversity-only, Calibra full, Full dataset). The ONLY
variable changed versus the BC benchmark is the policy architecture:

    BC-MLP  ->  ACT  (Action Chunking Transformer, Zhao et al. RSS 2023)

The ACT implementation here reproduces ACT's two defining ideas and its training
objective, adapted to the low-dimensional *state* observations this harness uses
(no image backbone):

  1. Action chunking  — predict a chunk of H future actions from one observation.
  2. CVAE             — a transformer encoder infers a style latent z from the
                        demonstrated action chunk (train only); a DETR-style
                        transformer decoder reconstructs the chunk from z + state.
  3. Objective        — L1 reconstruction (masked) + beta * KL(z || N(0, I)).

At evaluation, z is set to the prior mean (0) and the chunk is decoded from the
state alone. The reported `test_mse` is the **normalised first-action MSE** on the
held-out split — the same quantity the BC benchmark reports (the first action of
the chunk is what ACT actually executes before temporal ensembling), so the
per-condition `vs_random` numbers are directly comparable across the two policies.

SCOPE / HONESTY
---------------
This is a *state-based* ACT trained on offline action-prediction error, not the
full image-conditioned ACT evaluated by simulator rollout. It answers "does the
selection ranking transfer to a chunking-transformer learner," which is the point
of the ablation. It is NOT a claim about image-based ACT task success rates.

Output JSON matches `ablation_benchmark.py`'s schema, so `aggregate_ablation.py`
consumes ACT and BC results with the same tooling:

    python experiments/act_ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet \\
        --seeds 5 --json results/act_ablation_aloha.json
    python experiments/aggregate_ablation.py results/act_ablation_*.json
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

# Reuse the *identical* selection + data harness from the BC benchmark so the
# only variable across the two experiments is the policy architecture.
from experiments.ablation_benchmark import (  # noqa: E402
    _W,
    _load,
    _obs_key,
    _random_subset,
    _run_calibra_pipeline,
    _split,
    print_ablation,
    save_ablation_figure,
    select_calibra_full,
    select_diversity_only,
    select_facility,
    select_herding,
    select_kcenter,
    select_quality_only,
)

# ── ACT model ─────────────────────────────────────────────────────────────────


def _build_act(state_dim, action_dim, chunk, d_model, nhead, enc_layers, dec_layers, latent_dim):
    import torch
    import torch.nn as nn

    class ACT(nn.Module):
        """State-conditioned CVAE action-chunking transformer (low-dim ACT)."""

        def __init__(self):
            super().__init__()
            self.chunk = chunk
            self.action_dim = action_dim
            self.d_model = d_model
            self.latent_dim = latent_dim

            # --- CVAE encoder (train only): infers z from the action chunk + state
            self.enc_action_proj = nn.Linear(action_dim, d_model)
            self.enc_state_proj = nn.Linear(state_dim, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.enc_pos = nn.Parameter(torch.zeros(1, chunk + 2, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.cvae_encoder = nn.TransformerEncoder(enc_layer, enc_layers)
            self.latent_head = nn.Linear(d_model, latent_dim * 2)  # mu, logvar

            # --- Decoder: reconstruct chunk from z + state (DETR-style queries)
            self.latent_proj = nn.Linear(latent_dim, d_model)
            self.dec_state_proj = nn.Linear(state_dim, d_model)
            self.query_embed = nn.Parameter(torch.zeros(1, chunk, d_model))
            dec_layer = nn.TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
            self.action_head = nn.Linear(d_model, action_dim)

            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.enc_pos, std=0.02)
            nn.init.normal_(self.query_embed, std=0.02)

        def encode(self, state, chunk_actions):
            # tokens: [cls, state, a_0 .. a_{H-1}]
            b = state.shape[0]
            cls = self.cls_token.expand(b, -1, -1)
            st = self.enc_state_proj(state).unsqueeze(1)
            act = self.enc_action_proj(chunk_actions)
            seq = torch.cat([cls, st, act], dim=1) + self.enc_pos
            h = self.cvae_encoder(seq)
            mu_logvar = self.latent_head(h[:, 0])
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            return mu, logvar

        def decode(self, state, z):
            b = state.shape[0]
            mem = (self.dec_state_proj(state) + self.latent_proj(z)).unsqueeze(1)
            q = self.query_embed.expand(b, -1, -1)
            h = self.decoder(q, mem)
            return self.action_head(h)  # (b, chunk, action_dim)

        def forward(self, state, chunk_actions=None):
            if chunk_actions is not None:  # training: sample z from posterior
                mu, logvar = self.encode(state, chunk_actions)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                pred = self.decode(state, z)
                return pred, mu, logvar
            # inference: prior mean
            z = torch.zeros(state.shape[0], self.latent_dim, device=state.device)
            return self.decode(state, z)

    return ACT()


# ── windowed data (respects episode boundaries) ───────────────────────────────


def _act_windows(batch, chunk, stride, max_windows, seed):
    """Build (obs, action_chunk, valid_mask) windows within each episode.

    Returns OBS (N, state_dim), CH (N, chunk, action_dim), MASK (N, chunk) bool.
    A window starts at every `stride`-th step; the chunk is right-padded with
    zeros past the episode end and masked out of the loss.
    """
    obs_l, ch_l, mask_l = [], [], []
    for ep in batch.episodes:
        key = _obs_key(ep)
        if key is None:
            continue
        s, a = ep.observations[key], ep.actions
        ml = min(len(s), len(a))
        if ml < 2:
            continue
        s, a = s[:ml], a[:ml]
        adim = a.shape[1]
        for t in range(0, ml, stride):
            end = min(t + chunk, ml)
            n = end - t
            if n < 1:
                continue
            block = np.zeros((chunk, adim), dtype=np.float32)
            block[:n] = a[t:end]
            m = np.zeros(chunk, dtype=bool)
            m[:n] = True
            obs_l.append(s[t])
            ch_l.append(block)
            mask_l.append(m)
    if not obs_l:
        raise ValueError("No usable windows found.")
    OBS = np.asarray(obs_l, dtype=np.float32)
    CH = np.asarray(ch_l, dtype=np.float32)
    MASK = np.asarray(mask_l, dtype=bool)
    if max_windows and len(OBS) > max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(OBS), size=max_windows, replace=False)
        OBS, CH, MASK = OBS[idx], CH[idx], MASK[idx]
    return OBS, CH, MASK


# ── train / eval ──────────────────────────────────────────────────────────────


def _train_act(batch, cfg, seed=None):
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    OBS, CH, MASK = _act_windows(
        batch, cfg["chunk"], cfg["stride"], cfg["max_windows"], seed=(seed or 0)
    )
    state_dim, action_dim = OBS.shape[1], CH.shape[2]

    S = torch.from_numpy(OBS).to(device)
    A = torch.from_numpy(CH).to(device)
    M = torch.from_numpy(MASK).to(device)

    # Normalise using only valid (unmasked) actions.
    s_mean, s_std = S.mean(0), S.std(0).clamp(min=1e-6)
    flat = A[M]
    a_mean, a_std = flat.mean(0), flat.std(0).clamp(min=1e-6)
    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std

    net = _build_act(
        state_dim,
        action_dim,
        cfg["chunk"],
        cfg["d_model"],
        cfg["nhead"],
        cfg["enc_layers"],
        cfg["dec_layers"],
        cfg["latent_dim"],
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    beta = cfg["kl_weight"]
    N, bs = len(S_n), cfg["batch_size"]
    net.train()
    for _ in range(cfg["n_epochs"]):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            pred, mu, logvar = net(S_n[idx], A_n[idx])
            m = M[idx].unsqueeze(-1)
            l1 = (torch.abs(pred - A_n[idx]) * m).sum() / m.sum().clamp(min=1)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = l1 + beta * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return dict(
        net=net, s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std, device=device, cfg=cfg
    )


def _eval_act(art, test_batch):
    """Normalised first-action MSE on held-out windows (comparable to BC MSE)."""
    import torch

    net, s_mean, s_std, a_mean, a_std, device, cfg = (
        art[k] for k in ("net", "s_mean", "s_std", "a_mean", "a_std", "device", "cfg")
    )
    OBS, CH, MASK = _act_windows(
        test_batch, cfg["chunk"], cfg["stride"], cfg["max_windows"], seed=0
    )
    S = torch.from_numpy(OBS).to(device)
    A = torch.from_numpy(CH).to(device)
    S_n = (S - s_mean) / s_std
    A_n = (A - a_mean) / a_std
    net.eval()
    with torch.no_grad():
        # first action of the chunk is what ACT executes -> compare to BC per-step
        first_mse_parts = []
        chunk_mse_parts = []
        M = torch.from_numpy(MASK).to(device)
        bs = 2048
        for i in range(0, len(S_n), bs):
            pred = net(S_n[i : i + bs])
            tgt = A_n[i : i + bs]
            mm = M[i : i + bs].unsqueeze(-1)
            first_mse_parts.append(((pred[:, 0] - tgt[:, 0]) ** 2).mean(dim=1))
            cm = (((pred - tgt) ** 2) * mm).sum() / mm.sum().clamp(min=1)
            chunk_mse_parts.append(cm.expand(pred.shape[0]))
        first_mse = float(torch.cat(first_mse_parts).mean().item())
        chunk_mse = float(torch.cat(chunk_mse_parts).mean().item())
    return first_mse, chunk_mse


# ── ablation run (mirrors ablation_benchmark.run_ablation, ACT learner) ─────────


def run_act_ablation(train_batch, test_batch, cfg, keep_fraction=0.30, n_random_seeds=5):
    k = max(1, round(len(train_batch.episodes) * keep_fraction))

    print("  Running Calibra pipeline ...", flush=True)
    t0 = time.perf_counter()
    report = _run_calibra_pipeline(train_batch)
    print(f"  Pipeline done in {time.perf_counter() - t0:.1f}s", flush=True)

    conditions = [
        ("Full dataset", lambda: train_batch),
        ("K-Center greedy", lambda: select_kcenter(train_batch, k, seed=0)),
        ("Herding", lambda: select_herding(train_batch, k)),
        ("Facility Location", lambda: select_facility(train_batch, k)),
        ("Quality-filter only", lambda: select_quality_only(train_batch, report, k)),
        ("Diversity-only", lambda: select_diversity_only(train_batch, report, k)),
        ("Calibra full", lambda: select_calibra_full(train_batch, report, k)),
    ]

    seeds = list(range(n_random_seeds))

    def _mse(sub, s):
        first, _ = _eval_act(_train_act(sub, cfg, seed=s), test_batch)
        return first

    random_mses = [_mse(_random_subset(train_batch, k, seed=s * 17 + 42), s) for s in seeds]
    random_mean, random_std = float(np.mean(random_mses)), float(np.std(random_mses))

    rows = [
        {
            "condition": "Random",
            "n_episodes": k,
            "test_mse": random_mean,
            "test_mse_std": random_std,
            "per_seed_mse": random_mses,
            "vs_random": 0.0,
        }
    ]
    print(
        f"  Random (k={k}, {n_random_seeds} seeds): mse={random_mean:.5f} +/- {random_std:.5f}",
        flush=True,
    )

    for label, make_subset in conditions:
        sub = make_subset()
        per_seed = [_mse(sub, s) for s in seeds]
        mse, sd = float(np.mean(per_seed)), float(np.std(per_seed))
        delta = (random_mean - mse) / random_mean * 100
        rows.append(
            {
                "condition": label,
                "n_episodes": len(sub.episodes),
                "test_mse": mse,
                "test_mse_std": sd,
                "per_seed_mse": per_seed,
                "vs_random": delta,
            }
        )
        marker = "+++ " if label == "Calibra full" else "    "
        print(
            f"  {marker}{label:<26} k={len(sub.episodes):>3}  "
            f"mse={mse:.5f}+/-{sd:.5f}  vs_random={delta:+.1f}%",
            flush=True,
        )

    return rows


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    p = argparse.ArgumentParser(prog="act_ablation_benchmark")
    p.add_argument("--dataset", default="lerobot/aloha_mobile_cabinet")
    p.add_argument("--keep", "-k", type=float, default=0.30)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--chunk", type=int, default=16, help="ACT action-chunk length H")
    p.add_argument("--stride", type=int, default=4, help="window stride (dedup overlap)")
    p.add_argument(
        "--max-windows",
        type=int,
        default=40000,
        help="cap windows per training run to bound compute",
    )
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--save-fig", action="store_true")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args(argv)

    cfg = dict(
        chunk=args.chunk,
        stride=args.stride,
        max_windows=args.max_windows,
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )

    print("=" * _W)
    print("  CALIBRA ACT ABLATION BENCHMARK  (policy = Action Chunking Transformer)")
    print("=" * _W)
    print(f"  Dataset : {args.dataset}")
    print(f"  Keep    : {args.keep:.0%}   Seeds: {args.seeds}   Epochs: {args.n_epochs}")
    print(
        f"  ACT     : chunk={args.chunk} d_model={args.d_model} "
        f"enc/dec={args.enc_layers}/{args.dec_layers} latent={args.latent_dim} "
        f"kl={args.kl_weight}"
    )
    print()

    print("[1/3] Loading dataset ...")
    batch = _load(args.dataset)
    ep0 = batch.episodes[0]
    sk = _obs_key(ep0)
    state_dim = ep0.observations[sk].shape[1] if sk else 0
    print(
        f"  {batch.n_episodes} episodes  state_dim={state_dim}  action_dim={ep0.actions.shape[1]}"
    )

    print("[2/3] Train/test split (80/20) ...")
    train_batch, test_batch = _split(batch)
    print(f"  train={train_batch.n_episodes}  test={test_batch.n_episodes}")

    print(f"\n[3/3] Running ACT ablation (keep={args.keep:.0%}, {args.seeds} seeds) ...")
    rows = run_act_ablation(
        train_batch, test_batch, cfg, keep_fraction=args.keep, n_random_seeds=args.seeds
    )
    print_ablation(batch.dataset_name + " [ACT]", args.keep, rows)

    output = {
        "dataset": args.dataset,
        "policy": "act",
        "act_config": cfg,
        "keep_fraction": args.keep,
        "n_epochs": args.n_epochs,
        "n_seeds": args.seeds,
        "train_episodes": train_batch.n_episodes,
        "test_episodes": test_batch.n_episodes,
        "ablation": rows,
    }

    if args.save_fig:
        save_ablation_figure(batch.dataset_name + "_act", args.keep, rows)

    if args.json:
        out_path = pathlib.Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
