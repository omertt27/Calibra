"""
Human-reviewed evidence schema (ADR-012).

Provides the vocabulary for the fourth layer of the detection pipeline:

  Detection      → "What looks unusual?"             (anomalies.py)
  Calibration    → "How often on benign data?"        (calibration.py)
  Decision       → "What should I do about it?"       (comparison.py: Disposition)
  Interpretation → "Is this actually corruption?"     (this module)

FindingCharacterization is the result of a human reviewing a flagged episode
and classifying it as true corruption, unusual-but-valid, or ambiguous.

The most defensible first implementation is a hand-labeled dataset, NOT an
automatic classifier. Calibra ships a small curated set (see
data/reviewed_findings.jsonl) and the benchmark script can extend it.

Per-detector precision is computed from this dataset:

  | Detector      | Corruption | Unusual-valid | Ambiguous |
  |---|---|---|---|
  | jitter_cv     |        61% |           32% |        7% |
  | spike_rate    |        78% |           16% |        6% |
  | discontinuity |        91% |            6% |        3% |

This directly answers the first HF commenter: "Can Calibra distinguish
corrupted data from rare but useful data?"

Usage
-----
    from calibra.schema.evidence import ReviewedFindingDataset

    rfd = ReviewedFindingDataset.load("data/reviewed_findings.jsonl")
    rfd.precision_table()   # → list[dict] per detector
    rfd.to_markdown()       # → Markdown table string
    rfd.summary()           # → human-readable text block
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class FindingCharacterization(str, Enum):
    """
    Human-reviewed classification of a flagged episode.

    These labels require manual inspection — do not infer them automatically
    without a validated model trained on the reviewed dataset.

    TRUE_CORRUPTION  — The episode has a genuine recording or execution problem
                       (dropped frames, jitter burst, hardware reset, etc.) that
                       would likely degrade downstream policy training.

    UNUSUAL_VALID    — The episode is unusual relative to the dataset distribution,
                       but appears to represent legitimate robot behavior
                       (rare task variant, cautious motion, novel condition).
                       May be worth ANNOTATE rather than DROP.

    AMBIGUOUS        — The reviewer cannot determine from the available evidence
                       whether this is corruption or valid behavior. Needs more
                       context or a second reviewer.
    """

    TRUE_CORRUPTION = "true_corruption"
    UNUSUAL_VALID = "unusual_valid"
    AMBIGUOUS = "ambiguous"


@dataclass
class ReviewedFinding:
    """
    A single human-reviewed flagged episode.

    Captures enough provenance to reconstruct the review context:
    which dataset, which episode, which detector fired, and what the
    reviewer concluded.
    """

    dataset: str  # "lerobot/pusht"
    episode_id: str  # episode identifier within the dataset
    detector: str  # "jitter_cv" | "spike_rate" | ...
    characterization: FindingCharacterization
    reviewer_note: Optional[str] = None  # free-text explanation
    reviewed_at: Optional[str] = None  # ISO date "YYYY-MM-DD"
    reviewer_id: Optional[str] = None  # opaque reviewer token
    # Raw signal context, if available
    observed_value: Optional[float] = None  # the metric value that triggered
    median_value: Optional[float] = None  # dataset median for this metric
    deviation_mads: Optional[float] = None  # how many MADs from median

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "episode_id": self.episode_id,
            "detector": self.detector,
            "characterization": self.characterization.value,
            "reviewer_note": self.reviewer_note,
            "reviewed_at": self.reviewed_at,
            "reviewer_id": self.reviewer_id,
            "observed_value": self.observed_value,
            "median_value": self.median_value,
            "deviation_mads": self.deviation_mads,
        }

    @staticmethod
    def from_dict(d: dict) -> "ReviewedFinding":
        return ReviewedFinding(
            dataset=d["dataset"],
            episode_id=d["episode_id"],
            detector=d["detector"],
            characterization=FindingCharacterization(d["characterization"]),
            reviewer_note=d.get("reviewer_note"),
            reviewed_at=d.get("reviewed_at"),
            reviewer_id=d.get("reviewer_id"),
            observed_value=d.get("observed_value"),
            median_value=d.get("median_value"),
            deviation_mads=d.get("deviation_mads"),
        )


class ReviewedFindingDataset:
    """
    A collection of human-reviewed flagged episodes.

    Backed by a JSONL file (one ReviewedFinding per line). Provides methods
    to compute per-detector precision breakdowns and export them as tables.

    Example
    -------
        rfd = ReviewedFindingDataset.load("data/reviewed_findings.jsonl")
        rfd.add(ReviewedFinding(
            dataset="lerobot/pusht",
            episode_id="42",
            detector="jitter_cv",
            characterization=FindingCharacterization.TRUE_CORRUPTION,
            reviewer_note="Large jitter burst at steps 184-203. Likely USB dropout.",
            reviewed_at=str(date.today()),
        ))
        rfd.save("data/reviewed_findings.jsonl")
        print(rfd.to_markdown())
    """

    def __init__(self, findings: list[ReviewedFinding] | None = None) -> None:
        self._findings: list[ReviewedFinding] = list(findings or [])

    def add(self, finding: ReviewedFinding) -> None:
        self._findings.append(finding)

    def __len__(self) -> int:
        return len(self._findings)

    def findings(self) -> list[ReviewedFinding]:
        return list(self._findings)

    @staticmethod
    def load(path: str) -> "ReviewedFindingDataset":
        """Load from a JSONL file."""
        p = Path(path)
        if not p.exists():
            return ReviewedFindingDataset()
        findings = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    findings.append(ReviewedFinding.from_dict(json.loads(line)))
        return ReviewedFindingDataset(findings)

    def save(self, path: str) -> None:
        """Write to a JSONL file (overwrites)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for finding in self._findings:
                f.write(json.dumps(finding.to_dict()) + "\n")

    def precision_table(self) -> list[dict]:
        """
        Per-detector characterization breakdown.

        Returns a list of dicts, one per detector:
            {
                "detector": str,
                "n_reviewed": int,
                "true_corruption_rate": float,
                "unusual_valid_rate": float,
                "ambiguous_rate": float,
                "datasets": list[str],
            }
        """
        from collections import Counter, defaultdict

        by_det: dict[str, list[ReviewedFinding]] = defaultdict(list)
        for f in self._findings:
            by_det[f.detector].append(f)

        rows = []
        for det, items in sorted(by_det.items()):
            n = len(items)
            counts = Counter(f.characterization for f in items)
            datasets = sorted({f.dataset for f in items})
            rows.append(
                {
                    "detector": det,
                    "n_reviewed": n,
                    "true_corruption_rate": counts[FindingCharacterization.TRUE_CORRUPTION] / n,
                    "unusual_valid_rate": counts[FindingCharacterization.UNUSUAL_VALID] / n,
                    "ambiguous_rate": counts[FindingCharacterization.AMBIGUOUS] / n,
                    "datasets": datasets,
                }
            )
        return rows

    def to_markdown(self) -> str:
        """
        Render the precision table as a GitHub-flavored Markdown table.

        Example output:
            | Detector | n | Corruption | Unusual-valid | Ambiguous |
            |---|---|---|---|---|
            | jitter_cv | 32 | 61% | 32% | 7% |
        """
        rows = self.precision_table()
        if not rows:
            return "_No reviewed findings yet._"

        lines = [
            "| Detector | n | Corruption | Unusual-valid | Ambiguous |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['detector']} "
                f"| {r['n_reviewed']} "
                f"| {r['true_corruption_rate']:.0%} "
                f"| {r['unusual_valid_rate']:.0%} "
                f"| {r['ambiguous_rate']:.0%} |"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """
        Human-readable summary of reviewed findings.

        Example output:
            Reviewed Findings: 87 total across 3 detectors
            ─────────────────────────────────────────────
              jitter_cv    (n=32)  corruption 61%  unusual-valid 32%  ambiguous  7%
              spike_rate   (n=41)  corruption 78%  unusual-valid 16%  ambiguous  6%
              vel_disc     (n=14)  corruption 71%  unusual-valid 22%  ambiguous  7%
        """
        rows = self.precision_table()
        if not rows:
            return "Reviewed Findings: none"
        n_total = sum(r["n_reviewed"] for r in rows)
        lines = [
            f"Reviewed Findings: {n_total} total across {len(rows)} detector(s)",
            "─" * 52,
        ]
        for r in rows:
            det = r["detector"]
            n = r["n_reviewed"]
            corr = f"{r['true_corruption_rate']:.0%}"
            uv = f"{r['unusual_valid_rate']:.0%}"
            amb = f"{r['ambiguous_rate']:.0%}"
            lines.append(
                f"  {det:<18} (n={n:<3})  "
                f"corruption {corr:<5}  unusual-valid {uv:<5}  ambiguous {amb}"
            )
        return "\n".join(lines)
