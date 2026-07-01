"""
experiments/sft_fingerprint_alpaca.py

Compute per-example fingerprints and embeddings for Alpaca-52K using the
shared `calibra.llm.fingerprint` module (the reusable SFT analogue of
Calibra's robotics diagnostic pipeline — see calibra/llm/fingerprint.py for
the robotics↔SFT metric mapping).

Outputs:
  results/alpaca_fingerprints.parquet  — scalar features + text per example
  results/alpaca_embeddings.npy        — (N, 384) full-example embeddings for
                                         Stage 2 diversity selection

Usage:
    python experiments/sft_fingerprint_alpaca.py
    python experiments/sft_fingerprint_alpaca.py \\
        --dataset tatsu-lab/alpaca \\
        --out results/alpaca_fingerprints.parquet \\
        --embeddings results/alpaca_embeddings.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from calibra.llm.fingerprint import compute_fingerprints  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute per-example fingerprints and embeddings for Alpaca-52K."
    )
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--out", default="results/alpaca_fingerprints.parquet")
    p.add_argument("--embeddings", default="results/alpaca_embeddings.npy")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformer model")
    p.add_argument("--batch-size", type=int, default=256, help="Encoding batch size")
    args = p.parse_args()

    out_path = REPO_ROOT / args.out
    emb_path = REPO_ROOT / args.embeddings
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────────────
    print(f"Loading {args.dataset!r} ...", flush=True)
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    n = len(ds)
    print(f"  {n:,} examples")

    # Combine instruction + input (Alpaca has optional context in 'input')
    instructions: list[str] = []
    outputs: list[str] = []
    for ex in ds:
        instr = ex["instruction"]
        if ex.get("input", "").strip():
            instr = instr + "\n" + ex["input"]
        instructions.append(instr)
        outputs.append(ex["output"])

    # ── Fingerprint (shared library logic) ──────────────────────────────────────
    print(f"Loading embedding model {args.model!r} and computing fingerprints ...", flush=True)
    fp = compute_fingerprints(
        instructions, outputs, model_name=args.model, batch_size=args.batch_size
    )

    # ── Save fingerprints ──────────────────────────────────────────────────────
    print(f"Saving fingerprints to {out_path} ...", flush=True)
    table = pa.table(
        {
            "idx": pa.array(range(n), type=pa.int32()),
            "instruction": pa.array(instructions),
            "output": pa.array(outputs),
            "coherence": pa.array(fp.coherence),
            "repetition_rate": pa.array(fp.repetition_rate),
            "template_ratio": pa.array(fp.template_ratio),
            "response_length_words": pa.array(fp.response_length_words),
        }
    )
    pq.write_table(table, out_path)
    print(f"  Saved {out_path}")

    # ── Full-example embeddings for diversity selection ────────────────────────
    np.save(emb_path, fp.embeddings)
    print(f"  Saved {emb_path}  shape={fp.embeddings.shape}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    n_boilerplate = int((fp.template_ratio > 0).sum())
    n_short = int((fp.response_length_words < 5).sum())
    n_low_coherence = int((fp.coherence < 0.1).sum())
    n_high_repetition = int((fp.repetition_rate > 0.4).sum())

    print("\nFingerprint summary:")
    print(f"  coherence         mean={fp.coherence.mean():.3f}  std={fp.coherence.std():.3f}  "
          f"low (<0.10): {n_low_coherence}")
    print(f"  repetition_rate   mean={fp.repetition_rate.mean():.3f}  "
          f"std={fp.repetition_rate.std():.3f}  high (>0.40): {n_high_repetition}")
    print(f"  template_ratio    mean={fp.template_ratio.mean():.4f}  "
          f"n_with_boilerplate={n_boilerplate}")
    print(f"  response_length   mean={fp.response_length_words.mean():.1f} words  "
          f"median={int(np.median(fp.response_length_words))}  short (<5): {n_short}")


if __name__ == "__main__":
    main()
