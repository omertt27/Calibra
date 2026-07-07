"""
Aggregate coreset-selection results across datasets
===================================================
Consumes the per-dataset JSONs written by ablation_benchmark.py (each condition
carries a `per_seed_mse` list) and reports the summary that actually decides a
method comparison:

  - Mean rank of each method across datasets (lower = better).
  - Win / Tie / Loss record of Calibra full vs every other method.
  - Mean improvement vs the Random baseline.
  - Paired significance (per-seed) of Calibra full vs each method, per dataset.
  - Oracle upper bound: the best fixed method chosen per dataset — the headroom
    an adaptive, regime-aware selector could recover.

The point is not "who wins one dataset" but "who is best on average, and is the
gap real given seed noise." A method that is second-best everywhere can beat a
method that wins once and loses twice.

Usage
-----
    python experiments/aggregate_ablation.py results/ablation_*.json
    python experiments/aggregate_ablation.py            # defaults to results/ablation_*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

try:
    from scipy import stats as _stats
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# Selection methods compared at the fixed budget k. "Full dataset" uses 100% of
# the data (different budget) so it is reported separately as a reference, not
# ranked against the coreset methods.
_SELECTION_METHODS = [
    "Random",
    "K-Center greedy",
    "Herding",
    "Facility Location",
    "Quality-filter only",
    "Diversity-only",
    "Calibra full",
]
_REFERENCE = "Full dataset"
_FOCUS = "Calibra full"
_W = 78


def _load(paths: list[str]) -> dict[str, dict]:
    """Return {dataset_name: {condition: {"mean":.., "seeds":[..]}}}."""
    out: dict[str, dict] = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        ds = data.get("dataset", pathlib.Path(p).stem)
        rows = {}
        for r in data.get("ablation", []):
            # Normalize the random label (older runs used "Random 30% (n=..)")
            name = r["condition"]
            if name.startswith("Random"):
                name = "Random"
            rows[name] = {
                "mean": float(r["test_mse"]),
                "seeds": [float(x) for x in r.get("per_seed_mse", [])] or None,
                "n_episodes": r.get("n_episodes"),
            }
        out[ds] = rows
    return out


def _paired_p(a: list[float], b: list[float]) -> float | None:
    """Paired test on per-seed MSE vectors (a = method, b = reference). Two-sided."""
    if not a or not b or len(a) != len(b) or len(a) < 2:
        return None
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return 1.0
    if _HAVE_SCIPY:
        # Wilcoxon is underpowered at n=5; paired t-test is the conventional choice.
        return float(_stats.ttest_rel(a, b).pvalue)
    d = a - b
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
    # crude normal-approx two-sided p (no t-table dependency)
    from math import erf, sqrt
    return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aggregate_ablation")
    ap.add_argument("paths", nargs="*", help="ablation JSON files")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level for W/T/L")
    args = ap.parse_args(argv)

    paths = args.paths or sorted(glob.glob("results/ablation_*.json"))
    if not paths:
        print("No ablation JSONs found. Run ablation_benchmark.py first.", file=sys.stderr)
        return 1
    data = _load(paths)
    datasets = list(data)
    methods = [m for m in _SELECTION_METHODS if any(m in data[d] for d in datasets)]

    print("=" * _W)
    print("  AGGREGATE CORESET COMPARISON")
    print("=" * _W)
    print(f"  Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"  Seeds per cell: {max((len(data[d].get(_FOCUS,{}).get('seeds') or []) for d in datasets), default=0)}")
    print()

    # ── per-dataset ranks (lower MSE = better rank 1) ──────────────────────────
    ranks: dict[str, list[int]] = {m: [] for m in methods}
    vs_random: dict[str, list[float]] = {m: [] for m in methods}
    for d in datasets:
        present = [(m, data[d][m]["mean"]) for m in methods if m in data[d]]
        order = sorted(present, key=lambda kv: kv[1])
        rank_of = {m: i + 1 for i, (m, _) in enumerate(order)}
        rnd = data[d].get("Random", {}).get("mean")
        for m, mse in present:
            ranks[m].append(rank_of[m])
            if rnd:
                vs_random[m].append((rnd - mse) / rnd * 100)

    print(f"  {'Method':<22}{'Mean rank':>10}{'Mean vs-random':>16}{'Rank / dataset':>22}")
    print(f"  {'-'*70}")
    for m in sorted(methods, key=lambda x: np.mean(ranks[x]) if ranks[x] else 99):
        mr = np.mean(ranks[m]) if ranks[m] else float("nan")
        vr = np.mean(vs_random[m]) if vs_random[m] else float("nan")
        rk = " ".join(str(r) for r in ranks[m])
        mark = ">>>" if m == _FOCUS else "   "
        print(f"  {mark}{m:<19}{mr:>10.2f}{vr:>15.1f}%   [{rk}]")
    print()

    # ── Oracle upper bound ─────────────────────────────────────────────────────
    oracle_ranks, oracle_vr, oracle_pick = [], [], []
    for d in datasets:
        present = [(m, data[d][m]["mean"]) for m in methods if m in data[d]]
        best = min(present, key=lambda kv: kv[1])
        oracle_pick.append((d, best[0]))
        oracle_ranks.append(1)
        rnd = data[d].get("Random", {}).get("mean")
        if rnd:
            oracle_vr.append((rnd - best[1]) / rnd * 100)
    print(f"  Oracle (best method per dataset): mean vs-random = {np.mean(oracle_vr):+.1f}% "
          f"(mean rank 1.00)")
    for d, m in oracle_pick:
        print(f"      {d:<34} -> {m}")
    print(f"  Headroom for an adaptive selector = gap between Oracle and best fixed method above.")
    print()

    # ── Calibra full: W/T/L + paired significance vs each method ────────────────
    print(f"  {_FOCUS} vs. each method  (paired per-seed t-test, alpha={args.alpha})")
    print(f"  {'-'*70}")
    print(f"  {'Opponent':<22}{'W-T-L':>8}{'  per-dataset (Δ%, p)':<40}")
    for m in methods:
        if m == _FOCUS:
            continue
        w = t = l = 0
        cells = []
        for d in datasets:
            if m not in data[d] or _FOCUS not in data[d]:
                continue
            fa = data[d][_FOCUS]
            oa = data[d][m]
            # improvement of focus over opponent, % (positive = focus better)
            imp = (oa["mean"] - fa["mean"]) / oa["mean"] * 100
            p = _paired_p(fa.get("seeds"), oa.get("seeds"))
            sig = (p is not None and p < args.alpha)
            if not sig:
                t += 1
            elif imp > 0:
                w += 1
            else:
                l += 1
            ps = f"p={p:.3f}" if p is not None else "p=n/a"
            cells.append(f"{d.split('/')[-1][:10]}:{imp:+.0f}%,{ps}")
        print(f"  {m:<22}{f'{w}-{t}-{l}':>8}  " + "  ".join(cells))
    print()
    print("  W/T/L: Win = Calibra full significantly better; Tie = not significant; Loss = sig. worse.")
    print("=" * _W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
