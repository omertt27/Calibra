"""
Training-ready annotation sidecar (ADR-011 annotate mode).

A model-agnostic, per-episode metadata file that a training pipeline joins to
its dataset by `episode_id` / `episode_index`. The disposition tells a vanilla
trainer what to do (KEEP / DROP); the characterization columns are conditioning
inputs for a trainer that can consume them. Model-specific recipes (ACT /
Diffusion / VLA) sit ABOVE this schema — see `docs/annotate.md` — and never add
fields to it.

`AnnotationManifest.write(dir)` produces:
  calibra_annotations.jsonl         — one EpisodeAnnotation per line
  calibra_annotations.manifest.json — schema_version, field docs, counts, source
and, when `parquet=True`, also:
  calibra_annotations.parquet       — the same rows, columnar (needs pyarrow)

`schema_version` here is independent of CalibraReport's but follows the same
rule: a field rename or a semantic change requires a bump.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

ANNOTATION_SCHEMA_VERSION = "1.1.0"

_ROWS_FILE = "calibra_annotations.jsonl"
_MANIFEST_FILE = "calibra_annotations.manifest.json"
_PARQUET_FILE = "calibra_annotations.parquet"

# What every column means and how a trainer should use it. Written into the
# manifest so the sidecar is self-describing.
FIELD_DOCS: dict[str, str] = {
    "episode_index": "Position of the episode in the source dataset (0-based). Join key.",
    "episode_id": "Stable episode identifier from the source dataset. Join key.",
    "calibra_disposition": (
        "KEEP = train on as-is. DROP = exclude (integrity failure or pure "
        "redundancy). DOWNWEIGHT = include at reduced sample weight. ANNOTATE = "
        "vanilla training would drop this, but keep it if you condition on the "
        "columns below. REVIEW = needs a human. RECOLLECT = reserved."
    ),
    "calibra_score": (
        "0-100 per-episode cleanliness = 100·(1 - quality_risk). 100 = clean, "
        "0 = maximally risky. Quality-risk only, NOT a per-episode split of the "
        "dataset-level Calibra Score (which also blends coverage and task "
        "structure) — averaging these rows does not reproduce it."
    ),
    "quality_risk": (
        "0-1. Higher = more likely a real recording/execution problem. Absolute "
        "(fixed jerk/velocity/dropout/LDLJ thresholds), NOT normalized to this "
        "dataset — scripted/planner datasets can score uniformly mediocre while "
        "being healthy relative to themselves. Compare episodes within the file."
    ),
    "coverage_value": "0-1. Higher = more unique behavioral coverage this episode adds.",
    "anomaly_score": (
        "0-1. Higher = more statistically unusual on some metric (not itself "
        "bad). Weak signal below ~100 episodes."
    ),
    "redundancy": (
        "0-1 = 1 - coverage_value. Higher = this episode adds little unique "
        "behavioral coverage relative to the rest of the set. The complement of "
        "coverage_value, not an independent feature — and NOT pairwise "
        "duplicate detection."
    ),
    "success": "Episode success flag from the source dataset metadata, if present. null = unknown.",
    "integrity_flags": "List of metric names that failed an integrity/quality threshold.",
    "n_steps": "Episode length in timesteps (a dataset fact, not a Calibra signal).",
    "weight": "Training sample weight. null → treat as 1.0. Set only for DOWNWEIGHT rows.",
}


class EpisodeAnnotation(BaseModel):
    """One row of the sidecar — a single episode's disposition + characterization."""

    episode_index: int
    episode_id: str
    calibra_disposition: str  # a Disposition value
    calibra_score: Optional[float] = None
    quality_risk: Optional[float] = None
    coverage_value: Optional[float] = None
    anomaly_score: Optional[float] = None
    redundancy: Optional[float] = None
    success: Optional[bool] = None
    integrity_flags: list[str] = []
    n_steps: Optional[int] = None
    weight: Optional[float] = None


def _parquet_schema():
    """Explicit pyarrow schema so all-null columns still get a real type."""
    import pyarrow as pa

    return pa.schema(
        [
            ("episode_index", pa.int64()),
            ("episode_id", pa.string()),
            ("calibra_disposition", pa.string()),
            ("calibra_score", pa.float64()),
            ("quality_risk", pa.float64()),
            ("coverage_value", pa.float64()),
            ("anomaly_score", pa.float64()),
            ("redundancy", pa.float64()),
            ("success", pa.bool_()),
            ("integrity_flags", pa.list_(pa.string())),
            ("n_steps", pa.int64()),
            ("weight", pa.float64()),
        ]
    )


class AnnotationManifest(BaseModel):
    """
    The full sidecar: a header describing the schema and source, plus the
    per-episode rows. `write()` splits it into a rows file (JSONL) and a
    header file (JSON) so the rows stream and the header stays small.
    """

    schema_version: str = ANNOTATION_SCHEMA_VERSION
    calibra_version: str
    generated_at: str
    source_dataset: str
    dataset_format: Optional[str] = None
    n_episodes: int
    disposition_counts: dict[str, int] = {}
    field_docs: dict[str, str] = {}
    annotations: list[EpisodeAnnotation] = []

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def write(self, out_dir: str, *, parquet: bool = False) -> list[str]:
        """
        Write the JSONL rows file and the JSON manifest. With `parquet=True`
        also write a columnar `.parquet` of the same rows (requires pyarrow;
        raises ImportError with an install hint if it is missing). Returns the
        written paths.
        """
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        rows_path = d / _ROWS_FILE
        with rows_path.open("w", encoding="utf-8") as f:
            for row in self.annotations:
                f.write(row.model_dump_json() + "\n")

        manifest_path = d / _MANIFEST_FILE
        manifest_path.write_text(
            self.model_dump_json(indent=2, exclude={"annotations"}), encoding="utf-8"
        )
        written = [str(rows_path), str(manifest_path)]

        if parquet:
            written.append(str(self._write_parquet(d / _PARQUET_FILE)))
        return written

    def _write_parquet(self, path: Path) -> Path:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(
                "Parquet annotation output needs pyarrow. "
                "Install it with:  pip install 'calibra-robotics[lerobot]'"
            ) from exc

        table = pa.Table.from_pylist(
            [r.model_dump() for r in self.annotations], schema=_parquet_schema()
        )
        pq.write_table(table, path)
        return path

    @staticmethod
    def load(out_dir: str) -> "AnnotationManifest":
        d = Path(out_dir)
        header = json.loads((d / _MANIFEST_FILE).read_text(encoding="utf-8"))
        rows: list[dict] = []
        with (d / _ROWS_FILE).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        header["annotations"] = rows
        return AnnotationManifest.model_validate(header)
