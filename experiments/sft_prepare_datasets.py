"""
experiments/sft_prepare_datasets.py

Prepare training JSONL files for each SFT benchmark condition.

Conditions produced:
  random_9k       — 9K random examples from Alpaca-52K (fixed seed 42)
  calibra_sft_9k  — Calibra-selected 9K from results/calibra_sft_9k.json
  alpagasus_9k    — AlpaGasus-filtered 9K (requires local file; see notes below)
  lima_1k         — LIMA 1K (requires HF login: huggingface-cli login)

AlpaGasus note:
  The AlpaGasus authors did not publish a public HuggingFace dataset.
  To get their 9K:
    Option A: download their score file from the paper's GitHub release and
              place it at results/datasets/alpagasus_scores.json
              (format: list of {"instruction":..., "input":..., "output":..., "score": 4.5})
    Option B: rate Alpaca-52K yourself with GPT-3.5-turbo (see paper §3)
              and keep examples with score >= 4.5
  The script skips alpagasus_9k silently if neither file is found.

Output format (one JSON line per example):
  {"messages": [
      {"role": "user", "content": "<instruction>\\n<input>"},
      {"role": "assistant", "content": "<output>"}
  ]}

Usage:
    python experiments/sft_prepare_datasets.py
    python experiments/sft_prepare_datasets.py --out-dir results/datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _to_messages(instruction: str, inp: str, output: str) -> dict:
    user_content = instruction.strip()
    if inp and inp.strip():
        user_content += "\n" + inp.strip()
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output.strip()},
        ]
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} examples → {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare SFT benchmark training datasets.")
    p.add_argument("--out-dir", default="results/datasets")
    p.add_argument("--alpaca-dataset", default="tatsu-lab/alpaca")
    p.add_argument("--calibra-json", default="results/calibra_sft_9k.json")
    p.add_argument("--alpagasus-file", default="results/datasets/alpagasus_scores.json",
                   help="Local AlpaGasus scores file (skipped if missing)")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--random-n", type=int, default=9000)
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out_dir

    # ── Load Alpaca-52K ────────────────────────────────────────────────────────
    print(f"Loading {args.alpaca_dataset!r} ...", flush=True)
    from datasets import load_dataset

    alpaca = load_dataset(args.alpaca_dataset, split="train")
    alpaca_list = list(alpaca)
    n = len(alpaca_list)
    print(f"  {n:,} examples")

    # ── Random 9K ─────────────────────────────────────────────────────────────
    print(f"\nPreparing random_{args.random_n // 1000}k ...", flush=True)
    rng = np.random.default_rng(args.random_seed)
    random_indices = rng.choice(n, size=args.random_n, replace=False).tolist()
    random_records = [
        _to_messages(alpaca_list[i]["instruction"], alpaca_list[i]["input"], alpaca_list[i]["output"])
        for i in random_indices
    ]
    _write_jsonl(out_dir / f"random_{args.random_n // 1000}k" / "train.jsonl", random_records)
    # Save indices for reproducibility
    (out_dir / f"random_{args.random_n // 1000}k" / "indices.json").write_text(
        json.dumps({"seed": args.random_seed, "indices": random_indices})
    )

    # ── Calibra-SFT 9K ────────────────────────────────────────────────────────
    calibra_path = REPO_ROOT / args.calibra_json
    if calibra_path.exists():
        print("\nPreparing calibra_sft_9k ...", flush=True)
        calibra_data = json.loads(calibra_path.read_text())
        calibra_indices = calibra_data["keep_indices"]
        calibra_records = [
            _to_messages(
                alpaca_list[i]["instruction"],
                alpaca_list[i]["input"],
                alpaca_list[i]["output"],
            )
            for i in calibra_indices
        ]
        _write_jsonl(out_dir / "calibra_sft_9k" / "train.jsonl", calibra_records)
        # Save selection metadata alongside training data
        (out_dir / "calibra_sft_9k" / "selection_meta.json").write_text(
            json.dumps(
                {
                    k: v
                    for k, v in calibra_data.items()
                    if k not in ("keep_indices", "quality_fail_indices")
                },
                indent=2,
            )
        )
    else:
        print(f"\nWARNING: {calibra_path} not found — run sft_select_alpaca.py first", file=sys.stderr)

    # ── AlpaGasus 9K ──────────────────────────────────────────────────────────
    alpagasus_path = REPO_ROOT / args.alpagasus_file
    if alpagasus_path.exists():
        print("\nPreparing alpagasus_9k ...", flush=True)
        raw = json.loads(alpagasus_path.read_text())
        # Accept either list of examples with 'score' key, or list of examples without
        if isinstance(raw, list) and len(raw) > 0 and "score" in raw[0]:
            filtered = [ex for ex in raw if float(ex.get("score", 0)) >= 4.5]
        else:
            filtered = raw  # assume already filtered
        alpagasus_records = [
            _to_messages(ex["instruction"], ex.get("input", ""), ex["output"])
            for ex in filtered
        ]
        _write_jsonl(out_dir / "alpagasus_9k" / "train.jsonl", alpagasus_records)
    else:
        print(
            f"\nSkipping alpagasus_9k: {alpagasus_path} not found.\n"
            "  To include AlpaGasus, place their scored examples (JSON list with 'score' field)\n"
            "  at that path and re-run. See script docstring for details.",
            file=sys.stderr,
        )

    # ── LIMA 1K ───────────────────────────────────────────────────────────────
    print("\nAttempting LIMA 1K ...", flush=True)
    try:
        lima = load_dataset("GAIR/lima", split="train")
        lima_records = []
        for ex in lima:
            conversations = ex["conversations"]
            if len(conversations) >= 2:
                lima_records.append(
                    _to_messages(conversations[0], "", conversations[1])
                )
        _write_jsonl(out_dir / "lima_1k" / "train.jsonl", lima_records)
    except Exception as e:
        print(
            f"  Skipping LIMA: {e}\n"
            "  Run 'huggingface-cli login' and request access at "
            "https://huggingface.co/datasets/GAIR/lima",
            file=sys.stderr,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("Dataset files written to:", out_dir)
    for cond_dir in sorted(out_dir.iterdir()):
        train_file = cond_dir / "train.jsonl"
        if train_file.exists():
            n_lines = sum(1 for _ in open(train_file))
            print(f"  {cond_dir.name:<20} {n_lines:>6,} examples")
    print("─" * 50)
    print("\nNext: copy results/datasets/ to your GPU box and run sft_train.py")


if __name__ == "__main__":
    main()
