"""
experiments/sft_score_mtbench.py

Score MT-Bench answers with GPT-4 and report per-category and overall scores.

Requires: OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY=sk-...

    # Score one condition:
    python experiments/sft_score_mtbench.py \\
        --answers results/mtbench/calibra_sft_9k_answers.jsonl \\
        --out results/mtbench/calibra_sft_9k_scores.json

    # Compare all conditions:
    python experiments/sft_score_mtbench.py --compare

Scoring:
  Each turn is judged independently by GPT-4 on a 1–10 scale using the
  standard MT-Bench single-answer grading prompt (same as FastChat).
  The overall score is the mean of all 160 turn scores (80 Q × 2 turns).
  Per-category scores are means over the 10 questions per category.

Cost: ~$2–5 per condition (160 GPT-4 API calls per condition).
  Scoring is resumable: already-scored turns are loaded from the --out file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── MT-Bench judge prompts ─────────────────────────────────────────────────────
# Standard single-answer grading prompts from FastChat (lm-sys/FastChat).

_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your "
    "evaluation should consider factors such as the helpfulness, relevance, "
    "accuracy, depth, creativity, and level of detail of the response. Begin "
    "your evaluation by providing a short explanation. Be as objective as "
    "possible. After providing your explanation, you must rate the response on "
    "a scale of 1 to 10 by strictly following this format: \"[[rating]]\", "
    "for example: \"Rating: [[5]]\"."
)

_TURN1_TEMPLATE = """\
[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

_TURN2_TEMPLATE = """\
[Prior conversation]
User: {q1}
Assistant: {a1}

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

_RATING_RE = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")


def _extract_rating(text: str) -> float | None:
    m = _RATING_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def _judge_turn(
    client,
    user_prompt: str,
    model: str = "gpt-4-turbo",
    retries: int = 3,
) -> tuple[float | None, str]:
    """Call GPT-4 judge and return (score, raw_response). Returns (None, raw) on parse failure."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            score = _extract_rating(raw)
            return score, raw
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  API error ({e}), retrying in {wait}s ...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  API error after {retries} attempts: {e}", file=sys.stderr)
                return None, str(e)
    return None, ""


def _score_file(
    answers_path: Path,
    out_path: Path,
    judge_model: str = "gpt-4-turbo",
) -> dict:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)

    # Load answers
    answers = {}
    for line in answers_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            answers[rec["question_id"]] = rec

    # Load already-scored results to enable resuming
    scored: dict[int, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        scored = {r["question_id"]: r for r in existing.get("per_question", [])}
        print(f"  Resuming: {len(scored)} questions already scored")

    total = len(answers)
    remaining = [qid for qid in answers if qid not in scored]
    print(f"  Scoring {len(remaining)} remaining questions with {judge_model} ...")

    for i, qid in enumerate(remaining):
        rec = answers[qid]

        # Turn 1
        t1_prompt = _TURN1_TEMPLATE.format(question=rec["turn_1_q"], answer=rec["turn_1_a"])
        t1_score, t1_raw = _judge_turn(client, t1_prompt, model=judge_model)

        # Turn 2
        t2_prompt = _TURN2_TEMPLATE.format(
            q1=rec["turn_1_q"], a1=rec["turn_1_a"],
            question=rec["turn_2_q"], answer=rec["turn_2_a"],
        )
        t2_score, t2_raw = _judge_turn(client, t2_prompt, model=judge_model)

        scored[qid] = {
            "question_id": qid,
            "category": rec["category"],
            "turn_1_score": t1_score,
            "turn_2_score": t2_score,
            "turn_1_judge_raw": t1_raw,
            "turn_2_judge_raw": t2_raw,
        }

        # Save after every question (safe resuming)
        _save_scores(out_path, answers_path, scored, judge_model)

        t1_str = f"{t1_score:.1f}" if t1_score is not None else "?"
        t2_str = f"{t2_score:.1f}" if t2_score is not None else "?"
        print(f"  [{i + 1:>2}/{len(remaining)}] Q{qid} ({rec['category']}) "
              f"T1={t1_str}  T2={t2_str}", flush=True)

    _save_scores(out_path, answers_path, scored, judge_model)
    result = json.loads(out_path.read_text())
    return result


def _save_scores(
    out_path: Path,
    answers_path: Path,
    scored: dict[int, dict],
    judge_model: str,
) -> None:
    per_q = list(scored.values())

    # Per-category averages
    categories: dict[str, list[float]] = {}
    all_scores: list[float] = []
    for q in per_q:
        cat = q["category"]
        for turn_score in [q["turn_1_score"], q["turn_2_score"]]:
            if turn_score is not None:
                categories.setdefault(cat, []).append(turn_score)
                all_scores.append(turn_score)

    per_category = {cat: round(sum(v) / len(v), 3) for cat, v in categories.items()}
    overall = round(sum(all_scores) / len(all_scores), 3) if all_scores else None

    result = {
        "condition": answers_path.stem.replace("_answers", ""),
        "judge_model": judge_model,
        "n_questions_scored": len(per_q),
        "n_turns_scored": len(all_scores),
        "overall_score": overall,
        "per_category": dict(sorted(per_category.items())),
        "per_question": per_q,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def _print_table(results: list[dict]) -> None:
    """Print a comparison table across all scored conditions."""
    categories = ["writing", "roleplay", "reasoning", "math", "coding",
                  "extraction", "stem", "humanities"]

    col_w = 22
    header = f"{'Category':<16}" + "".join(f"{r['condition'][:col_w]:>{col_w}}" for r in results)
    print("\n" + "━" * (16 + col_w * len(results)))
    print("  MT-BENCH SCORES")
    print("━" * (16 + col_w * len(results)))
    print(header)
    print("─" * (16 + col_w * len(results)))
    for cat in categories:
        row = f"{cat:<16}"
        for r in results:
            v = r["per_category"].get(cat)
            row += f"{v:>{col_w}.2f}" if v is not None else f"{'—':>{col_w}}"
        print(row)
    print("─" * (16 + col_w * len(results)))
    overall_row = f"{'OVERALL':<16}"
    for r in results:
        v = r["overall_score"]
        overall_row += f"{v:>{col_w}.2f}" if v is not None else f"{'—':>{col_w}}"
    print(overall_row)
    print("━" * (16 + col_w * len(results)))


def main() -> None:
    p = argparse.ArgumentParser(description="Score MT-Bench answers with GPT-4.")
    p.add_argument("--answers", help="Path to answers JSONL (from sft_gen_answers.py)")
    p.add_argument("--out", help="Output JSON path (default: derived from --answers path)")
    p.add_argument("--judge-model", default="gpt-4-turbo",
                   help="OpenAI model for judging (default: gpt-4-turbo)")
    p.add_argument("--compare", action="store_true",
                   help="Print a comparison table of all scored conditions in results/mtbench/")
    args = p.parse_args()

    mtbench_dir = REPO_ROOT / "results" / "mtbench"

    if args.compare:
        score_files = sorted(mtbench_dir.glob("*_scores.json"))
        if not score_files:
            print("No score files found in results/mtbench/", file=sys.stderr)
            sys.exit(1)
        results = [json.loads(f.read_text()) for f in score_files]
        _print_table(results)
        return

    if not args.answers:
        p.error("--answers is required unless --compare is used")

    answers_path = Path(args.answers)
    if not answers_path.is_absolute():
        answers_path = REPO_ROOT / answers_path
    if not answers_path.exists():
        print(f"ERROR: {answers_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else (
        mtbench_dir / answers_path.name.replace("_answers.jsonl", "_scores.json")
    )
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    result = _score_file(answers_path, out_path, judge_model=args.judge_model)

    print(f"\n{'─' * 40}")
    print(f"  Condition : {result['condition']}")
    print(f"  Questions : {result['n_questions_scored']}")
    print(f"  Turns     : {result['n_turns_scored']}")
    print(f"  OVERALL   : {result['overall_score']}")
    print(f"{'─' * 40}")
    for cat, score in result["per_category"].items():
        print(f"  {cat:<14}: {score:.2f}")
    print(f"\nScores → {out_path}")
    print(f"Run with --compare to see all conditions side by side.")


if __name__ == "__main__":
    main()