"""
calibra.llm.fingerprint — per-example fingerprints for SFT/instruction-tuning data.

This is the text analogue of Calibra's robotics diagnostic metrics:

  Robotics                    SFT
  ─────────────────────────   ──────────────────────────────────────────
  jerk spike rate          →  repetition_rate (bigram repetition in response)
  LDLJ / smoothness        →  coherence (cos_sim between instruction & response)
  coverage entropy         →  embedding diversity (farthest-point in SBERT space)
  temporal dropout         →  response_length < threshold (degenerate sample)
  velocity discontinuity   →  template_ratio (boilerplate opener fraction)

Usage
-----
    from calibra.llm.fingerprint import compute_fingerprints

    result = compute_fingerprints(instructions, outputs)
    result.coherence          # (N,) float32
    result.repetition_rate    # (N,) float32
    result.template_ratio     # (N,) float32
    result.response_length_words  # (N,) int32
    result.embeddings         # (N, D) float32 — averaged instruction/response embeddings
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ── boilerplate detection ──────────────────────────────────────────────────────

_BOILERPLATE_RE = re.compile(
    r"^(sure,?\s+here(?:'s| is| are)\b"
    r"|of course[,!]"
    r"|certainly[,!]"
    r"|as an ai\b"
    r"|i'?d be happy to\b"
    r"|great[,!]\s+here\b"
    r"|absolutely[,!]"
    r"|thanks for asking\b"
    r"|no problem[,!]"
    r"|i\s+hope\s+this\s+helps\b"
    r"|hello[,!]"
    r"|hi[,!]\s+there\b)",
    re.IGNORECASE,
)


def _repetition_rate(text: str) -> float:
    """Fraction of bigrams that are duplicates. 0 = no repetition, 1 = fully repetitive."""
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


def _template_ratio(text: str) -> float:
    """Fraction of response words that belong to a boilerplate opener."""
    text = text.strip()
    m = _BOILERPLATE_RE.match(text)
    if not m:
        return 0.0
    matched_word_count = len(m.group(0).split())
    total_word_count = max(1, len(text.split()))
    return matched_word_count / total_word_count


# ── result schema ─────────────────────────────────────────────────────────────


@dataclass
class FingerprintResult:
    """
    Per-example fingerprints for a set of (instruction, response) pairs.

    Attributes
    ----------
    coherence              : cosine similarity between instruction and response
                              embeddings. High = response addresses the instruction.
    repetition_rate         : fraction of duplicate bigrams in the response.
    template_ratio          : fraction of response words that are a boilerplate opener.
    response_length_words   : response length in whitespace-split words.
    embeddings              : (N, D) unit-normalized average of instruction/response
                              embeddings — used as the behavioral representation for
                              diversity selection (analogous to action-space stats
                              in the robotics pipeline).
    """

    coherence: np.ndarray
    repetition_rate: np.ndarray
    template_ratio: np.ndarray
    response_length_words: np.ndarray
    embeddings: np.ndarray


def compute_fingerprints(
    instructions: Sequence[str],
    outputs: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> FingerprintResult:
    """
    Compute per-example fingerprints and embeddings for SFT/instruction data.

    Requires `sentence-transformers` (install with `pip install calibra-robotics[llm]`).
    """
    if len(instructions) != len(outputs):
        raise ValueError(
            f"instructions and outputs must have equal length, got "
            f"{len(instructions)} and {len(outputs)}"
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "compute_fingerprints requires sentence-transformers. "
            "Install with `pip install calibra-robotics[llm]`."
        ) from exc

    model = SentenceTransformer(model_name)

    instr_embs: np.ndarray = model.encode(
        list(instructions),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    resp_embs: np.ndarray = model.encode(
        list(outputs),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    coherence = (instr_embs * resp_embs).sum(axis=1).astype(np.float32)
    repetition_rates = np.array([_repetition_rate(o) for o in outputs], dtype=np.float32)
    template_ratios = np.array([_template_ratio(o) for o in outputs], dtype=np.float32)
    response_lengths = np.array([len(o.split()) for o in outputs], dtype=np.int32)

    full_embs = (instr_embs + resp_embs) / 2.0
    norms = np.linalg.norm(full_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    full_embs = (full_embs / norms).astype(np.float32)

    return FingerprintResult(
        coherence=coherence,
        repetition_rate=repetition_rates,
        template_ratio=template_ratios,
        response_length_words=response_lengths,
        embeddings=full_embs,
    )
