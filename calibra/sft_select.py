"""
calibra sft-select — coreset selection for SFT / instruction-tuning datasets.

Loads instruction-tuning examples from a local JSON/JSONL file or a HuggingFace
Hub dataset id, fingerprints them, and selects a quality-filtered, diverse
coreset using `calibra.llm.select`.

Usage
-----
    calibra sft-select tatsu-lab/alpaca --keep 0.3 --out coreset_index.json
    calibra sft-select /data/my_sft.jsonl --keep 0.5 --instruction-key prompt --output-key response
    calibra sft-select tatsu-lab/alpaca --keep 0.3 --export-dataset ./coreset.jsonl

Output
------
The selection result is written as JSON to --out (default: coreset_index.json):
  keep_indices, quality_fail_indices, diversity_pruned_indices,
  quality_scores, diversity_scores, aggregate_fingerprint, n_original, n_kept,
  keep_fraction_actual, method

`aggregate_fingerprint` feeds `calibra sft-outcome` after training.

Exit codes
----------
    0  Selection completed successfully.
    1  Error loading the dataset or running selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from calibra.llm import SFTCoresetSelector, compute_fingerprints


def _load_examples(path: str, split: str) -> list[dict[str, Any]]:
    local_path = Path(path)
    if local_path.exists():
        if local_path.suffix == ".jsonl":
            with open(local_path) as f:
                return [json.loads(line) for line in f if line.strip()]
        if local_path.suffix == ".json":
            with open(local_path) as f:
                return json.load(f)
        raise ValueError(f"unsupported local file extension: {local_path.suffix}")

    from datasets import load_dataset

    ds = load_dataset(path, split=split)
    return list(ds)


def run_sft_select(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="calibra sft-select",
        description="Select a high-quality, diverse coreset from an instruction-tuning dataset.",
    )
    p.add_argument("path", help="Local JSON/JSONL path or HuggingFace Hub dataset id")
    p.add_argument("--split", default="train", help="HF dataset split to load (default: train)")
    p.add_argument(
        "--instruction-key", default="instruction", help="Field name for the instruction/prompt"
    )
    p.add_argument("--output-key", default="output", help="Field name for the response")
    p.add_argument(
        "--keep", "-k", type=float, default=0.5, metavar="FRACTION",
        help="Target fraction of examples to keep (default: 0.5)",
    )
    p.add_argument(
        "--out", "-o", metavar="PATH", default="coreset_index.json",
        help="Output JSON file path (default: coreset_index.json)",
    )
    p.add_argument(
        "--quality-only", action="store_true",
        help="Stage 1 only — filter quality failures but skip diversity selection",
    )
    p.add_argument("--min-coherence", type=float, default=0.10)
    p.add_argument("--max-repetition-rate", type=float, default=0.40)
    p.add_argument("--max-template-ratio", type=float, default=0.50)
    p.add_argument("--min-length-words", type=int, default=5)
    p.add_argument(
        "--approximate", action="store_true",
        help="Use approximate MiniBatch diversity selection for 100k+ examples",
    )
    p.add_argument(
        "--export-dataset", metavar="PATH", default=None,
        help="Write the selected examples as JSONL to PATH",
    )
    p.add_argument(
        "--json", "-j", action="store_true",
        help="Print full JSON result to stdout in addition to writing --out",
    )
    args = p.parse_args(argv)

    if not (0.0 < args.keep <= 1.0):
        print("error: --keep must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    log(f"Loading {args.path!r} ...")
    try:
        examples = _load_examples(args.path, args.split)
    except Exception as exc:
        print(f"error loading dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    log(f"  {len(examples)} examples")
    log("Computing fingerprints ...")

    instructions: list[str] = []
    outputs: list[str] = []
    for ex in examples:
        instr = ex[args.instruction_key]
        extra_input = ex.get("input", "")
        if extra_input and extra_input.strip():
            instr = f"{instr}\n{extra_input}"
        instructions.append(instr)
        outputs.append(ex[args.output_key])

    try:
        fp = compute_fingerprints(instructions, outputs)
    except Exception as exc:
        print(f"error computing fingerprints: {exc}", file=sys.stderr)
        sys.exit(1)

    log("Running coreset selection ...")
    selector = SFTCoresetSelector(
        keep_fraction=args.keep,
        min_coherence=args.min_coherence,
        max_repetition_rate=args.max_repetition_rate,
        max_template_ratio=args.max_template_ratio,
        min_length_words=args.min_length_words,
        quality_only=args.quality_only,
        use_approximate=args.approximate,
    )
    result = selector.select(fp)

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    log(f"Wrote {out_path}")
    print(result.summary())

    if args.export_dataset:
        export_path = Path(args.export_dataset)
        with open(export_path, "w") as f:
            for i in result.keep_indices:
                f.write(json.dumps(examples[i]) + "\n")
        log(f"Exported {len(result.keep_indices)} examples to {export_path}")

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))

    sys.exit(0)
