"""
experiments/sft_gen_answers.py

Generate MT-Bench answers from a fine-tuned model checkpoint.
Designed to run on a GPU box. Handles the 2-turn MT-Bench format.

MT-Bench: 80 questions × 8 categories × 2 turns = 160 scored answers.
Turn 2 is generated in context (the model sees turn 1 Q+A before answering turn 2).

Usage (GPU box):
    python experiments/sft_gen_answers.py \\
        --model results/checkpoints/calibra_sft_9k/final \\
        --condition calibra_sft_9k \\
        --out results/mtbench/calibra_sft_9k_answers.jsonl

    # For all conditions after training:
    for cond in random_9k calibra_sft_9k alpagasus_9k; do
        python experiments/sft_gen_answers.py \\
            --model results/checkpoints/$cond/final \\
            --condition $cond
    done

Questions are fetched from FastChat's published MT-Bench data and cached locally
at results/mtbench/mt_bench_questions.jsonl.

Output format (one JSON line per question):
    {
        "question_id": 1,
        "category": "writing",
        "condition": "calibra_sft_9k",
        "turn_1_q": "...",
        "turn_1_a": "...",
        "turn_2_q": "...",
        "turn_2_a": "..."
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

_MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main"
    "/fastchat/llm_judge/data/mt_bench/question.jsonl"
)
_MT_BENCH_CACHE = REPO_ROOT / "results" / "mtbench" / "mt_bench_questions.jsonl"

_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_NEW_TOKENS = 1024


def _load_mt_bench_questions() -> list[dict]:
    if _MT_BENCH_CACHE.exists():
        questions = [json.loads(l) for l in _MT_BENCH_CACHE.read_text().splitlines() if l.strip()]
        print(f"  Loaded {len(questions)} MT-Bench questions from cache")
        return questions

    print(f"  Fetching MT-Bench questions from FastChat repo ...", flush=True)
    _MT_BENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(_MT_BENCH_URL, timeout=30) as resp:
            content = resp.read().decode()
        _MT_BENCH_CACHE.write_text(content)
        questions = [json.loads(l) for l in content.splitlines() if l.strip()]
        print(f"  Fetched and cached {len(questions)} questions → {_MT_BENCH_CACHE}")
        return questions
    except Exception as e:
        print(
            f"ERROR: could not fetch MT-Bench questions: {e}\n"
            f"Please manually download the file from the FastChat repository\n"
            f"and save it to {_MT_BENCH_CACHE}",
            file=sys.stderr,
        )
        sys.exit(1)


def _generate(model, tokenizer, messages: list[dict], temperature: float, max_new_tokens: int) -> str:
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate MT-Bench answers from a fine-tuned model.")
    p.add_argument("--model", required=True, help="Path to fine-tuned checkpoint directory")
    p.add_argument("--condition", required=True, help="Condition label (used in output filename)")
    p.add_argument("--out-dir", default="results/mtbench")
    p.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE)
    p.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--question-ids", type=int, nargs="+", default=None,
                   help="Subset of question IDs to run (default: all 80)")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.condition}_answers.jsonl"

    # ── Load model ─────────────────────────────────────────────────────────────
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model!r} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    print(f"  Model on {next(model.parameters()).device}")

    # ── Load questions ─────────────────────────────────────────────────────────
    questions = _load_mt_bench_questions()
    if args.question_ids:
        questions = [q for q in questions if q["question_id"] in set(args.question_ids)]
    print(f"Generating answers for {len(questions)} questions ...")

    # ── Skip already-completed questions ──────────────────────────────────────
    done_ids: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])
        print(f"  Resuming: {len(done_ids)} questions already done, {len(questions) - len(done_ids)} remaining")

    # ── Generate ───────────────────────────────────────────────────────────────
    with open(out_path, "a") as f_out:
        for i, q in enumerate(questions):
            qid = q["question_id"]
            if qid in done_ids:
                continue

            turns = q["turns"]
            category = q.get("category", "unknown")

            # Turn 1
            t1_messages = [{"role": "user", "content": turns[0]}]
            t1_answer = _generate(model, tokenizer, t1_messages, args.temperature, args.max_new_tokens)

            # Turn 2 — model sees turn 1 Q+A in context
            t2_messages = [
                {"role": "user", "content": turns[0]},
                {"role": "assistant", "content": t1_answer},
                {"role": "user", "content": turns[1]},
            ]
            t2_answer = _generate(model, tokenizer, t2_messages, args.temperature, args.max_new_tokens)

            record = {
                "question_id": qid,
                "category": category,
                "condition": args.condition,
                "turn_1_q": turns[0],
                "turn_1_a": t1_answer,
                "turn_2_q": turns[1],
                "turn_2_a": t2_answer,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"  [{i + 1 - len(done_ids):>2}/{len(questions) - len(done_ids)}] "
                  f"Q{qid} ({category}) done", flush=True)

    total = sum(1 for l in out_path.read_text().splitlines() if l.strip())
    print(f"\nAnswers written: {total} questions → {out_path}")
    print(f"Next: python experiments/sft_score_mtbench.py --answers {out_path}")


if __name__ == "__main__":
    main()
