"""
experiments/sft_train.py

SFT fine-tuning script for the MT-Bench benchmark.
Designed to run on a GPU box (CUDA). MPS is supported for smoke tests only.

Usage (GPU box):
    pip install trl>=0.8 peft accelerate bitsandbytes

    # Train one condition:
    python experiments/sft_train.py \\
        --condition calibra_sft_9k \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --out-dir results/checkpoints

    # All three conditions in sequence:
    for cond in random_9k calibra_sft_9k alpagasus_9k; do
        python experiments/sft_train.py --condition $cond
    done

    # Smoke test on MPS (100 steps, batch_size=1):
    python experiments/sft_train.py --condition random_9k --smoke-test

Hyperparameters are fixed across all conditions — only the training data changes.
This is intentional: it isolates data quality as the independent variable.

Model: Qwen/Qwen2.5-1.5B-Instruct (full fine-tune, no LoRA)
  Full fine-tune on 1.5B fits on a single A100 40GB in bf16.
  For 7B-class models, add --lora to enable LoRA (rank=64, alpha=128).

Loss: computed only on assistant responses (user turns are masked),
  using DataCollatorForCompletionOnlyLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="SFT training for Calibra-LLM benchmark.")
    p.add_argument("--condition", required=True,
                   choices=["random_9k", "calibra_sft_9k", "alpagasus_9k", "lima_1k", "full_52k"],
                   help="Which training condition to run")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace model ID")
    p.add_argument("--data-dir", default="results/datasets",
                   help="Directory containing {condition}/train.jsonl")
    p.add_argument("--out-dir", default="results/checkpoints",
                   help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch size = per_device_batch × grad_accum")
    p.add_argument("--lora", action="store_true",
                   help="Use LoRA (rank=64) instead of full fine-tune (needed for 7B+ models)")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 100 steps with batch_size=1 to verify setup (MPS-safe)")
    args = p.parse_args()

    import torch

    data_path = REPO_ROOT / args.data_dir / args.condition / "train.jsonl"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found — run sft_prepare_datasets.py first", file=sys.stderr)
        sys.exit(1)

    out_path = REPO_ROOT / args.out_dir / args.condition
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────────────
    from datasets import Dataset

    records = [json.loads(line) for line in open(data_path)]
    train_dataset = Dataset.from_list(records)
    print(f"Loaded {len(train_dataset):,} examples from {data_path}")

    # ── Load model and tokenizer ───────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model!r} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2" if _flash_available() else "eager",
    )

    # ── LoRA (optional) ───────────────────────────────────────────────────────
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ── Format: apply chat template, mask user turns ───────────────────────────
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    # Find the assistant response marker in this model's chat template
    _probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Heuristic: find where assistant content starts
    if "<|im_start|>assistant" in _probe:
        response_template = "<|im_start|>assistant\n"
    elif "[/INST]" in _probe:
        response_template = "[/INST]"
    elif "<|start_header_id|>assistant" in _probe:
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        # Fallback: find the assistant content in the template and use what precedes it
        response_template = "assistant\n"

    def _format(examples: dict) -> list[str]:
        return [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ── Training config ────────────────────────────────────────────────────────
    if args.smoke_test:
        train_kwargs = dict(max_steps=100, per_device_train_batch_size=1, gradient_accumulation_steps=1)
        print("SMOKE TEST: 100 steps, batch_size=1")
    else:
        train_kwargs = dict(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum,
        )

    training_args = SFTConfig(
        output_dir=str(out_path),
        **train_kwargs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and not args.smoke_test,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        formatting_func=_format,
        data_collator=collator,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"Training {args.condition} on {args.model} ...", flush=True)
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = out_path / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    config_out = {
        "condition": args.condition,
        "model": args.model,
        "n_train": len(train_dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "effective_batch_size": args.per_device_batch * args.grad_accum,
        "lora": args.lora,
        "lora_rank": args.lora_rank if args.lora else None,
        "checkpoint": str(final_path),
    }
    (out_path / "train_config.json").write_text(json.dumps(config_out, indent=2))

    print(f"\nDone. Checkpoint → {final_path}")
    print(f"Next: python experiments/sft_gen_answers.py --model {final_path} --condition {args.condition}")


def _flash_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()