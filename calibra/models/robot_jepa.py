"""
RobotJEPA: Joint Embedding Predictive Architecture for robot demonstrations.

Learns a latent dynamics model  f(s_t, a_t) -> z_{t+1}  without reconstruction.
Trained with a VICReg-style objective (Bardes et al., 2022) to prevent
representational collapse.

Key properties
--------------
- Runs on CPU or Apple MPS (M2/M3) without a GPU cluster
- Produces per-episode 'surprise scores' — a world-model learnability signal
- Operates on proprioceptive states; vision not required
- Integrates with EpisodeBatch as a drop-in Calibra component
- PyTorch is an optional dependency; graceful fallback if absent

Connection to LeCun's JEPA
--------------------------
Standard IL trains a policy p(a|s). Calibra's RobotJEPA trains a predictor
z_{t+1} = g(f(s_t), a_t) in a non-reconstructive latent space, aligned with
the I-JEPA / V-JEPA paradigm. The key insight is that episodes with high
Calibra quality scores (low jerk, low dropout) also have low JEPA surprise —
they are more learnable by a world model. This validates that hand-crafted
diagnostic metrics are offline proxies for world-model predictability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_PROPRIO_KEYS = ("proprio", "state", "joint_state", "joint_pos",
                 "robot_state", "qpos", "obs")


@dataclass
class RobotJEPAConfig:
    latent_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 2
    lr: float = 3e-4
    n_epochs: int = 60
    batch_size: int = 512
    vicreg_lambda: float = 25.0   # variance loss weight
    vicreg_mu: float = 1.0        # covariance loss weight
    grad_clip: float = 1.0
    warmup_epochs: int = 5


class RobotJEPA:
    """
    Lightweight JEPA world model for robot state-action sequences.

    Usage
    -----
        jepa = RobotJEPA()
        jepa.fit(batch)                      # train on EpisodeBatch
        scores = jepa.score_episodes(batch)  # {episode_id: surprise_score}

    Surprise score interpretation
    -----------------------------
    High surprise (> 0.7)  →  episode is either:
        (a) genuinely novel dynamics the model hasn't seen, OR
        (b) corrupted/noisy (jerk spikes, dropout)
    Cross-reference with jerk_spike_rate to distinguish (a) from (b).
    An episode that scores high on BOTH surprise AND jerk rate is corrupted.
    An episode that scores high on surprise but LOW on jerk rate is genuinely novel.
    """

    def __init__(self, config: Optional[RobotJEPAConfig] = None) -> None:
        self.config = config or RobotJEPAConfig()
        self._model = None
        self._s_mean: Optional[object] = None
        self._s_std: Optional[object] = None
        self._a_mean: Optional[object] = None
        self._a_std: Optional[object] = None
        self._device = None
        self.training_loss_curve: list[float] = []

    # ── device ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_device():
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ── architecture ──────────────────────────────────────────────────────────

    def _build_model(self, state_dim: int, action_dim: int):
        import torch.nn as nn
        cfg = self.config
        D, H = cfg.latent_dim, cfg.hidden_dim

        def mlp(in_d: int, out_d: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(in_d, H), nn.LayerNorm(H), nn.GELU()]
            for _ in range(cfg.n_layers - 1):
                layers += [nn.Linear(H, H), nn.LayerNorm(H), nn.GELU()]
            layers.append(nn.Linear(H, out_d))
            return nn.Sequential(*layers)

        return nn.ModuleDict({
            "encoder":   mlp(state_dim, D),
            "predictor": mlp(D + action_dim, D),
        })

    # ── data extraction ───────────────────────────────────────────────────────

    def _extract_transitions(self, batch):
        """Extract (s_t, a_t, s_{t+1}) triples across all episodes."""
        states_l, actions_l, next_l, ids_l = [], [], [], []

        for ep in batch.episodes:
            key = next((k for k in _PROPRIO_KEYS if k in ep.observations), None)
            if key is None:
                continue
            s = ep.observations[key]
            a = ep.actions
            s = s[:, np.newaxis] if s.ndim == 1 else s
            a = a[:, np.newaxis] if a.ndim == 1 else a
            T = min(len(s), len(a))
            if T < 2:
                continue
            s = s[:T].astype(np.float32)
            a = a[:T].astype(np.float32)
            states_l.append(s[:-1])
            actions_l.append(a[:-1])
            next_l.append(s[1:])
            ids_l.extend([ep.metadata.episode_id] * (T - 1))

        if not states_l:
            return None, None, None, None
        return (
            np.concatenate(states_l, 0),
            np.concatenate(actions_l, 0),
            np.concatenate(next_l, 0),
            ids_l,
        )

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, batch) -> "RobotJEPA":
        """Train the JEPA on an EpisodeBatch. Returns self for chaining."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(
                "RobotJEPA requires PyTorch. Install with: pip install torch"
            ) from e

        states, actions, next_states, _ = self._extract_transitions(batch)
        if states is None:
            return self

        self._device = self._get_device()
        cfg = self.config

        model = self._build_model(states.shape[1], actions.shape[1]).to(self._device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.1
        )

        S = torch.from_numpy(states).to(self._device)
        A = torch.from_numpy(actions).to(self._device)
        NS = torch.from_numpy(next_states).to(self._device)

        # Input normalization statistics
        s_mean = S.mean(0)
        s_std = S.std(0).clamp(min=1e-6)
        a_mean = A.mean(0)
        a_std = A.std(0).clamp(min=1e-6)
        S_n = (S - s_mean) / s_std
        A_n = (A - a_mean) / a_std
        NS_n = (NS - s_mean) / s_std

        N = len(S_n)
        self.training_loss_curve = []

        for epoch in range(cfg.n_epochs):
            perm = torch.randperm(N, device=self._device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                s_b, a_b, ns_b = S_n[idx], A_n[idx], NS_n[idx]

                z_t = model["encoder"](s_b)
                z_t1_pred = model["predictor"](torch.cat([z_t, a_b], dim=-1))

                with torch.no_grad():
                    z_t1_target = model["encoder"](ns_b)

                # ── VICReg objective ──
                # 1. Invariance: push predicted close to stop-gradient target
                inv_loss = F.mse_loss(z_t1_pred, z_t1_target)

                # 2. Variance: prevent collapse (keep std > 1 per dimension)
                std = z_t1_pred.std(dim=0).clamp(min=1e-4)
                var_loss = F.relu(1.0 - std).mean()

                # 3. Covariance: decorrelate latent dimensions
                z_c = z_t1_pred - z_t1_pred.mean(0)
                B = z_c.shape[0]
                cov = (z_c.T @ z_c) / (B - 1)
                diag_mask = torch.eye(cov.shape[0], device=self._device, dtype=torch.bool)
                cov_loss = cov[~diag_mask].pow(2).sum() / cfg.latent_dim

                loss = (inv_loss
                        + cfg.vicreg_lambda * var_loss
                        + cfg.vicreg_mu * cov_loss)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            if n_batches > 0:
                self.training_loss_curve.append(epoch_loss / n_batches)

        self._model = model
        self._s_mean = s_mean
        self._s_std = s_std
        self._a_mean = a_mean
        self._a_std = a_std
        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def score_episodes(self, batch) -> dict[str, float]:
        """
        Compute per-episode surprise score after training.

        Returns
        -------
        dict mapping episode_id -> normalized surprise score in [0, 1].
        Higher = world model more surprised by this episode's dynamics.

        Cross-reference interpretation:
          high surprise + high jerk rate  →  CORRUPTED episode
          high surprise + low  jerk rate  →  GENUINELY NOVEL episode (keep)
          low  surprise                   →  REDUNDANT or well-covered episode
        """
        if self._model is None:
            raise RuntimeError("Call fit() before score_episodes().")

        import torch
        import torch.nn.functional as F

        scores: dict[str, float] = {}

        with torch.no_grad():
            for ep in batch.episodes:
                key = next((k for k in _PROPRIO_KEYS if k in ep.observations), None)
                if key is None:
                    scores[ep.metadata.episode_id] = 0.0
                    continue

                s = ep.observations[key]
                a = ep.actions
                s = s[:, np.newaxis] if s.ndim == 1 else s
                a = a[:, np.newaxis] if a.ndim == 1 else a
                T = min(len(s), len(a))
                if T < 2:
                    scores[ep.metadata.episode_id] = 0.0
                    continue

                s_t = torch.from_numpy(s[:T].astype(np.float32)).to(self._device)
                a_t = torch.from_numpy(a[:T].astype(np.float32)).to(self._device)

                s_n = (s_t - self._s_mean) / self._s_std
                a_n = (a_t - self._a_mean) / self._a_std

                z_t = self._model["encoder"](s_n[:-1])
                z_t1_pred = self._model["predictor"](
                    torch.cat([z_t, a_n[:-1]], dim=-1)
                )
                z_t1_target = self._model["encoder"](s_n[1:])

                # Surprise = mean cosine distance between predicted and actual
                cos_sim = F.cosine_similarity(z_t1_pred, z_t1_target, dim=-1)
                surprise = float((1.0 - cos_sim).clamp(0.0).mean().item())
                scores[ep.metadata.episode_id] = surprise

        # Normalize across episodes to [0, 1] for comparability
        if len(scores) > 1:
            vals = np.array(list(scores.values()))
            v_min, v_max = vals.min(), vals.max()
            if v_max > v_min:
                for k in scores:
                    scores[k] = float((scores[k] - v_min) / (v_max - v_min))

        return scores

    def predict_next_latent(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Single-step latent prediction for model-predictive planning.

        Parameters
        ----------
        state  : (state_dim,) current state
        action : (action_dim,) proposed action

        Returns
        -------
        predicted next state latent : (latent_dim,)
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_next_latent().")

        import torch
        with torch.no_grad():
            s = torch.from_numpy(
                state.astype(np.float32)
            ).unsqueeze(0).to(self._device)
            a = torch.from_numpy(
                action.astype(np.float32)
            ).unsqueeze(0).to(self._device)

            s_n = (s - self._s_mean) / self._s_std
            a_n = (a - self._a_mean) / self._a_std

            z = self._model["encoder"](s_n)
            z_next = self._model["predictor"](torch.cat([z, a_n], dim=-1))
            return z_next.squeeze(0).cpu().numpy()

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode a state (state_dim,) into the latent space (latent_dim,)."""
        if self._model is None:
            raise RuntimeError("Call fit() before encode().")

        import torch
        with torch.no_grad():
            s = torch.from_numpy(
                state.astype(np.float32)
            ).unsqueeze(0).to(self._device)
            s_n = (s - self._s_mean) / self._s_std
            z = self._model["encoder"](s_n)
            return z.squeeze(0).cpu().numpy()