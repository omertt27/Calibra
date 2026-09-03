"""
Reference implementation of the metadata-conditioning wiring for the ADR-011
research gate (see METADATA_CONDITIONING_BENCHMARK.md).

This file is model-agnostic and has **no `lerobot` dependency** — it does the
dataset side (which episodes each arm trains on, and the per-episode
conditioning vector) and marks the two points where a real ACT / Diffusion
Policy implementation has to consume that vector. Copy it into the partner's
training repo and fill in `inject_into_act` / `inject_into_diffusion`.

    from metadata_conditioning_reference import prepare_arm

    spec = prepare_arm("calibra_meta/", arm="D", seed=0)
    #   spec.episode_ids      -> which episodes to train on
    #   spec.cond[episode_id] -> np.ndarray conditioning vector (or zeros)
    #   spec.weight[episode_id] -> loss weight (1.0 unless DOWNWEIGHT)
    #   spec.actual_retention_pct -> log this next to the nominal --keep target
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Arms of the frozen matrix. `select` says which episodes train; `metadata`
# says whether the conditioning vector is real or zeros.
ARMS = {
    "A": dict(select="all_non_drop", metadata=False),
    "B": dict(select="keep", metadata=False),
    "C": dict(select="all_non_drop", metadata=True),
    "D": dict(select="keep_plus_annotate", metadata=True),
    "R": dict(select="random_like_D", metadata=False),
    "R+": dict(select="random_like_D", metadata=True),
    "D0": dict(select="keep", metadata=True),  # optional KEEP-only + metadata
}

# Columns binned into quartiles and one-hot encoded as the conditioning vector.
_COND_COLUMNS = ("quality_risk", "coverage_value")
_N_BINS = 4


@dataclass
class ArmSpec:
    arm: str
    episode_ids: list[str]
    cond: dict[str, np.ndarray]
    weight: dict[str, float]
    nominal_keep_pct: float
    actual_retention_pct: float
    cond_dim: int


# ── sidecar loading ──────────────────────────────────────────────────────────


def load_sidecar(sidecar_dir: str) -> list[dict]:
    """Read calibra_annotations.jsonl into a list of row dicts."""
    p = Path(sidecar_dir) / "calibra_annotations.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {p}")
    return rows


def _nominal_keep_pct(sidecar_dir: str, rows: list[dict]) -> float:
    """Best-effort nominal --keep target: read the raw curation report if present,
    else fall back to |KEEP| / |all|."""
    raw = Path(sidecar_dir) / "calibra_curation_report.json"
    if raw.exists():
        rep = json.loads(raw.read_text())
        n0 = rep.get("original_n_episodes") or len(rows)
        n_keep = sum(1 for d in rep.get("dispositions", []) if d.get("disposition") == "KEEP")
        if n0:
            return 100.0 * n_keep / n0
    n_keep = sum(1 for r in rows if r["calibra_disposition"] == "KEEP")
    return 100.0 * n_keep / len(rows)


# ── arm membership ───────────────────────────────────────────────────────────


def _select_ids(rows: list[dict], arm: str, seed: int) -> list[str]:
    disp = {r["episode_id"]: r["calibra_disposition"] for r in rows}
    non_drop = [e for e, d in disp.items() if d != "DROP"]
    keep = [e for e, d in disp.items() if d == "KEEP"]
    keep_plus_annotate = [e for e, d in disp.items() if d in ("KEEP", "ANNOTATE")]

    kind = ARMS[arm]["select"]
    if kind == "all_non_drop":
        return non_drop
    if kind == "keep":
        return keep
    if kind == "keep_plus_annotate":
        return keep_plus_annotate
    if kind == "random_like_D":
        # random subset of the non-DROP pool, same size as arm D
        rng = random.Random(seed)
        pool = list(non_drop)
        rng.shuffle(pool)
        return sorted(pool[: len(keep_plus_annotate)])
    raise ValueError(f"unknown select kind {kind!r}")


# ── conditioning vector ──────────────────────────────────────────────────────


def _bin_edges(values: list[float]) -> np.ndarray:
    """Quartile edges over the training set. Interior edges only (len == _N_BINS-1)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return np.array([0.25, 0.5, 0.75])
    qs = np.linspace(0, 1, _N_BINS + 1)[1:-1]
    return np.quantile(v, qs)


def _onehot(value: Optional[float], edges: np.ndarray) -> np.ndarray:
    vec = np.zeros(_N_BINS, dtype=np.float32)
    if value is None:
        return vec  # unknown → all-zero, distinct from any bin
    vec[int(np.searchsorted(edges, value))] = 1.0
    return vec


def build_conditioning(
    rows: list[dict], train_ids: list[str], *, use_metadata: bool
) -> tuple[dict[str, np.ndarray], int]:
    """
    Per-episode conditioning vector: one-hot quartile bins of each _COND_COLUMNS
    column, concatenated. Bins are fit **on the training set of this arm**.
    use_metadata=False → every vector is zeros (same shape, so the model code is
    identical across arms).
    """
    by_id = {r["episode_id"]: r for r in rows}
    train = [by_id[e] for e in train_ids]
    dim = _N_BINS * len(_COND_COLUMNS)

    if not use_metadata:
        return {e: np.zeros(dim, dtype=np.float32) for e in train_ids}, dim

    edges = {c: _bin_edges([r.get(c) for r in train]) for c in _COND_COLUMNS}
    cond = {}
    for r in train:
        parts = [_onehot(r.get(c), edges[c]) for c in _COND_COLUMNS]
        cond[r["episode_id"]] = np.concatenate(parts)
    return cond, dim


def clean_conditioning_vector() -> np.ndarray:
    """What to pass at inference: bin 0 (cleanest / lowest quality_risk, and
    lowest coverage_value bin — adjust if you'd rather ask for high coverage)."""
    parts = []
    for _ in _COND_COLUMNS:
        v = np.zeros(_N_BINS, dtype=np.float32)
        v[0] = 1.0
        parts.append(v)
    return np.concatenate(parts)


# ── weights ──────────────────────────────────────────────────────────────────


def build_weights(rows: list[dict], train_ids: list[str]) -> dict[str, float]:
    by_id = {r["episode_id"]: r for r in rows}
    out = {}
    for e in train_ids:
        w = by_id[e].get("weight")
        out[e] = 1.0 if w is None else float(w)
    return out


# ── top level ────────────────────────────────────────────────────────────────


def prepare_arm(sidecar_dir: str, arm: str, seed: int = 0) -> ArmSpec:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}, got {arm!r}")
    rows = load_sidecar(sidecar_dir)
    total = len(rows)

    train_ids = _select_ids(rows, arm, seed)
    use_meta = ARMS[arm]["metadata"]
    cond, dim = build_conditioning(rows, train_ids, use_metadata=use_meta)
    weight = build_weights(rows, train_ids)

    return ArmSpec(
        arm=arm,
        episode_ids=sorted(train_ids),
        cond=cond,
        weight=weight,
        nominal_keep_pct=round(_nominal_keep_pct(sidecar_dir, rows), 2),
        actual_retention_pct=round(100.0 * len(train_ids) / total, 2),
        cond_dim=dim,
    )


# ── model injection points (fill these in the partner's repo) ─────────────────


def inject_into_act(policy_config, cond_dim: int):  # pragma: no cover - stub
    """
    ACT: add a learned projection of the per-episode conditioning vector to the
    encoder's conditioning/style token stream.

      self.calibra_proj = nn.Linear(cond_dim, policy_config.dim_model)
      # in forward(), before the transformer encoder:
      cond_token = self.calibra_proj(calibra_cond)          # (B, dim_model)
      encoder_in = torch.cat([encoder_in, cond_token[:, None, :]], dim=1)

    At inference pass `clean_conditioning_vector()` for every sample.
    """
    raise NotImplementedError("wire into your ACT implementation")


def inject_into_diffusion(policy_config, cond_dim: int):  # pragma: no cover - stub
    """
    Diffusion Policy: concatenate an MLP embedding of the conditioning vector to
    the global conditioning fed to the denoiser.

      self.calibra_mlp = nn.Sequential(
          nn.Linear(cond_dim, 128), nn.ReLU(), nn.Linear(128, cond_embed_dim)
      )
      global_cond = torch.cat([global_cond, self.calibra_mlp(calibra_cond)], dim=-1)

    At inference pass `clean_conditioning_vector()`.
    """
    raise NotImplementedError("wire into your Diffusion Policy implementation")


if __name__ == "__main__":  # small demo against a real sidecar dir
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "calibra_meta"
    for a in ("A", "B", "C", "D", "R", "R+"):
        s = prepare_arm(d, a, seed=0)
        print(
            f"{a:3}  n={len(s.episode_ids):5}  "
            f"nominal_keep={s.nominal_keep_pct:5.1f}%  actual={s.actual_retention_pct:5.1f}%  "
            f"cond_dim={s.cond_dim}  meta={ARMS[a]['metadata']}"
        )
