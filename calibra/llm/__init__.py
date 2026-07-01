"""
calibra.llm — dataset curation for SFT / instruction-tuning data.

Public API
----------
    calibra.llm.select(examples, keep_fraction=0.5)          -> SFTSelectionResult
    calibra.llm.select_hf_dataset(dataset, keep_fraction=0.5) -> datasets.Dataset

`select_hf_dataset` is the intended integration point: it returns a filtered
`datasets.Dataset` via the library's own `.select()` method, so it drops into
an existing HuggingFace data-loading pipeline as a single call rather than a
one-off script:

    from calibra.llm import select_hf_dataset

    raw_ds = load_dataset("tatsu-lab/alpaca", split="train")
    train_ds = select_hf_dataset(raw_ds, keep_fraction=0.3)
    # train_ds is a plain datasets.Dataset — use it exactly as before.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from calibra.llm.fingerprint import FingerprintResult, compute_fingerprints
from calibra.llm.select import SFTCoresetSelector, SFTSelectionResult

__all__ = [
    "FingerprintResult",
    "compute_fingerprints",
    "SFTCoresetSelector",
    "SFTSelectionResult",
    "select",
    "select_hf_dataset",
]


def select(
    examples: Sequence[Mapping[str, Any]],
    instruction_key: str = "instruction",
    output_key: str = "output",
    keep_fraction: float = 0.5,
    **selector_kwargs: Any,
) -> SFTSelectionResult:
    """
    Fingerprint and select a diverse, quality-filtered coreset from a sequence
    of instruction-tuning examples (e.g. a list of dicts or a HF `datasets.Dataset`).

    Combines the optional `input` field into the instruction, matching the
    Alpaca-style schema `{instruction, input, output}`.
    """
    instructions: list[str] = []
    outputs: list[str] = []
    for ex in examples:
        instr = ex[instruction_key]
        extra_input = ex.get("input", "")
        if extra_input and extra_input.strip():
            instr = f"{instr}\n{extra_input}"
        instructions.append(instr)
        outputs.append(ex[output_key])

    fp = compute_fingerprints(instructions, outputs)
    selector = SFTCoresetSelector(keep_fraction=keep_fraction, **selector_kwargs)
    return selector.select(fp)


def select_hf_dataset(
    dataset: Any,
    instruction_key: str = "instruction",
    output_key: str = "output",
    keep_fraction: float = 0.5,
    **selector_kwargs: Any,
) -> Any:
    """
    Curate a HuggingFace `datasets.Dataset` in place: returns `dataset.select(keep_indices)`.

    This is the drop-in integration point for an existing HF data-loading pipeline —
    no need to materialize fingerprints or coreset indices separately.
    """
    result = select(
        dataset,
        instruction_key=instruction_key,
        output_key=output_key,
        keep_fraction=keep_fraction,
        **selector_kwargs,
    )
    return dataset.select(result.keep_indices)
