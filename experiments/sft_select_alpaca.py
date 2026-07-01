"""
experiments/sft_select_alpaca.py

Select a diverse, clean SFT coreset from Alpaca fingerprints using
`calibra.llm.select.SFTCoresetSelector` — the same quality-filter +
greedy-max-coverage pipeline exposed by `calibra sft-select`, applied here to
pre-computed fingerprints/embeddings (from sft_fingerprint_alpaca.py) for
benchmark reproducibility.

Prints a sanity comparison: selected coreset vs random 9K on all quality
and diversity metrics. This directly answers: "Is Calibra-SFT selecting
cleaner and more diverse examples than random?"

Usage:
    python experiments/sft_select_alpaca.py \\
        --fingerprints results/alpaca_fingerprints.parquet \\
        --embeddings results/alpaca_embeddings.npy \\
        --keep 9000 \\
        --out results/calibra_sft_9k.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from calibra.llm.fingerprint import FingerprintResult  # noqa: E402
from calibra.llm.select import SFTCoresetSelector, _avg_nn_distance  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Select a Calibra-SFT coreset from Alpaca fingerprints."
    )
    p.add_argument("--fingerprints", default="results/alpaca_fingerprints.parquet")
    p.add_argument("--embeddings", default="results/alpaca_embeddings.npy")
    p.add_argument("--keep", type=int, default=9000, metavar="N",
                   help="Number of examples to keep (default: 9000)")
    p.add_argument("--out", default="results/calibra_sft_9k.json")
    p.add_argument("--min-coherence", type=float, default=0.1,
                   help="Remove examples with coherence below this (default: 0.10)")
    p.add_argument("--max-rep", type=float, default=0.4,
                   help="Remove examples with repetition_rate above this (default: 0.40)")
    p.add_argument("--min-length", type=int, default=5,
                   help="Remove examples with response_length_words below this (default: 5)")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="MiniBatch size for approximate selection (default: 1000)")
    p.add_argument("--exact", action="store_true",
                   help="Use exact greedy k-center — fast on MPS/CUDA, slow on CPU for large K")
    args = p.parse_args()

    fp_path = REPO_ROOT / args.fingerprints
    emb_path = REPO_ROOT / args.embeddings
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────
    print(f"Loading fingerprints from {fp_path} ...", flush=True)
    df = pq.read_table(fp_path).to_pydict()
    n = len(df["idx"])
    coherence = np.array(df["coherence"], dtype=np.float32)
    repetition = np.array(df["repetition_rate"], dtype=np.float32)
    template_ratio = np.zeros(n, dtype=np.float32)  # not thresholded by this benchmark's CLI
    lengths = np.array(df["response_length_words"], dtype=np.int32)
    orig_indices = np.array(df["idx"], dtype=np.int32)

    print(f"Loading embeddings from {emb_path} ...", flush=True)
    embs = np.load(emb_path)  # (N, D)
    assert len(embs) == n, f"Shape mismatch: {len(embs)} embeddings vs {n} fingerprints"

    fp = FingerprintResult(
        coherence=coherence,
        repetition_rate=repetition,
        template_ratio=template_ratio,
        response_length_words=lengths,
        embeddings=embs,
    )

    # ── Stage 1 + Stage 2 via the shared library selector ─────────────────────
    keep_fraction = min(1.0, args.keep / max(n, 1))
    selector = SFTCoresetSelector(
        keep_fraction=keep_fraction,
        min_coherence=args.min_coherence,
        max_repetition_rate=args.max_rep,
        max_template_ratio=1.0,  # not part of this benchmark's fingerprint (see template_ratio above)
        min_length_words=args.min_length,
        use_approximate=not args.exact,
        batch_size=args.batch_size,
    )
    print(f"  Stage 1 + Stage 2: keep_fraction={keep_fraction:.4f} "
          f"({'approximate' if not args.exact else 'exact'}) ...", flush=True)
    result = selector.select(fp)

    selected_local = np.array(result.keep_indices, dtype=np.int32)
    keep_orig = orig_indices[selected_local].tolist()

    print(f"  Stage 1: {n:,} total  →  {result.n_quality_failures:,} quality failures  →  "
          f"{n - result.n_quality_failures:,} quality-passing", flush=True)

    # ── Sanity check: selected vs random ──────────────────────────────────────
    rng = np.random.default_rng(42)
    random_local = rng.choice(n, size=min(args.keep, n), replace=False)

    sel_coh = float(coherence[selected_local].mean())
    sel_rep = float(repetition[selected_local].mean())
    sel_len = float(lengths[selected_local].mean())

    rnd_coh = float(coherence[random_local].mean())
    rnd_rep = float(repetition[random_local].mean())
    rnd_len = float(lengths[random_local].mean())

    print("Computing coverage scores (sampling 2K) ...", flush=True)
    sel_cov = _avg_nn_distance(embs[selected_local])
    rnd_cov = _avg_nn_distance(embs[random_local])

    # ── Write JSON ─────────────────────────────────────────────────────────────
    out_result = {
        "method": result.method + " (sft_adapter)",
        "n_original": n,
        "n_quality_fail": result.n_quality_failures,
        "n_quality_pass": n - result.n_quality_failures,
        "n_kept": len(keep_orig),
        "n_diversity_pruned": result.n_diversity_pruned,
        "keep_fraction_actual": round(len(keep_orig) / max(n, 1), 6),
        "thresholds": {
            "min_coherence": args.min_coherence,
            "max_repetition_rate": args.max_rep,
            "min_response_length_words": args.min_length,
        },
        "quality_fail_indices": orig_indices[result.quality_fail_indices].tolist(),
        "keep_indices": keep_orig,
        "aggregate_fingerprint": result.aggregate_fingerprint,
        "sanity": {
            "selected": {
                "mean_coherence": round(sel_coh, 4),
                "mean_repetition_rate": round(sel_rep, 4),
                "mean_response_length_words": round(sel_len, 1),
                "coverage_avg_nn_dist": round(sel_cov, 4),
            },
            "random_9k": {
                "mean_coherence": round(rnd_coh, 4),
                "mean_repetition_rate": round(rnd_rep, 4),
                "mean_response_length_words": round(rnd_len, 1),
                "coverage_avg_nn_dist": round(rnd_cov, 4),
            },
        },
    }

    with open(out_path, "w") as f:
        json.dump(out_result, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────────────
    W = 62
    print("\n" + "━" * W)
    print("  CALIBRA-SFT SELECTION SUMMARY")
    print("━" * W)
    print(f"  Original examples    : {n:,}")
    print(f"  Quality failures     : {result.n_quality_failures:,}   (Stage 1)")
    print(f"  Diversity pruned     : {result.n_diversity_pruned:,}  (Stage 2)")
    print(f"  Coreset size         : {len(keep_orig):,}   ({out_result['keep_fraction_actual']:.1%} of original)")
    print(f"  Method               : {out_result['method']}")
    print("─" * W)
    coh_arrow  = "↑" if sel_coh  > rnd_coh  else ("↓" if sel_coh  < rnd_coh  else "=")
    rep_arrow  = "↓" if sel_rep  < rnd_rep  else ("↑" if sel_rep  > rnd_rep  else "=")
    len_arrow  = "↑" if sel_len  > rnd_len  else ("↓" if sel_len  < rnd_len  else "=")
    cov_arrow  = "↑" if sel_cov  > rnd_cov  else ("↓" if sel_cov  < rnd_cov  else "=")
    print(f"  {'Metric':<30} {'Selected':>10}  {'Random 9K':>10}  {'':>2}")
    print("─" * W)
    print(f"  {'Mean coherence':<30} {sel_coh:>10.4f}  {rnd_coh:>10.4f}  {coh_arrow}")
    print(f"  {'Mean repetition_rate':<30} {sel_rep:>10.4f}  {rnd_rep:>10.4f}  {rep_arrow}")
    print(f"  {'Mean response length (words)':<30} {sel_len:>10.1f}  {rnd_len:>10.1f}  {len_arrow}")
    print(f"  {'Coverage (avg NN dist)':<30} {sel_cov:>10.4f}  {rnd_cov:>10.4f}  {cov_arrow}")
    print("━" * W)
    print(f"\n  {coh_arrow}↑ coherence = responses more on-topic")
    print(f"  {rep_arrow}↓ repetition = responses less self-repetitive")
    print(f"  {cov_arrow}↑ coverage = examples more spread across instruction space")
    print(f"\n  Coreset index → {out_path}")
    print(f"  Load with:  json.load(open('{out_path}'))['keep_indices']")


if __name__ == "__main__":
    main()
