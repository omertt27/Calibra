"""
Build the ADR-011 annotate-mode sidecar from a CurationReport.

`calibra prune --annotate DIR` calls `write_annotations()`. The sidecar is a
projection of the decision layer: every episode gets a row with its
disposition and characterization, model-agnostic. Conditioning recipes for
specific policy families live in `docs/annotate.md`, above this schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from calibra.schema.annotations import (
    FIELD_DOCS,
    AnnotationManifest,
    EpisodeAnnotation,
)
from calibra.schema.comparison import CurationReport


def build_annotation_manifest(
    curation: CurationReport,
    *,
    source_dataset: str,
    dataset_format: Optional[str] = None,
) -> AnnotationManifest:
    """Turn a CurationReport's per-episode dispositions into an AnnotationManifest."""
    from calibra import __version__

    rows = [
        EpisodeAnnotation(
            episode_index=d.episode_index,
            episode_id=d.episode_id,
            calibra_disposition=d.disposition.value,
            calibra_score=d.calibra_score,
            quality_risk=d.quality_risk,
            coverage_value=d.coverage_value,
            anomaly_score=d.anomaly_score,
            redundancy=d.redundancy,
            success=d.success,
            integrity_flags=list(d.integrity_flags),
            n_steps=d.n_steps,
            weight=d.weight,
        )
        for d in curation.dispositions
    ]
    return AnnotationManifest(
        calibra_version=__version__,
        generated_at=AnnotationManifest.now(),
        source_dataset=source_dataset,
        dataset_format=dataset_format,
        n_episodes=len(rows),
        disposition_counts=curation.disposition_counts(),
        field_docs=dict(FIELD_DOCS),
        annotations=rows,
    )


def write_annotations(
    curation: CurationReport,
    out_dir: str,
    *,
    source_dataset: str,
    dataset_format: Optional[str] = None,
    parquet: bool = False,
) -> list[str]:
    """
    Write the annotate-mode sidecar to `out_dir`:
      calibra_annotations.jsonl          — per-episode rows
      calibra_annotations.manifest.json  — schema, field docs, disposition counts
      calibra_curation_report.json       — the raw CurationReport, for tooling
      calibra_annotations.parquet        — columnar rows (only when parquet=True)

    `parquet=True` needs pyarrow (`pip install 'calibra-robotics[lerobot]'`).
    Returns the list of written paths.
    """
    manifest = build_annotation_manifest(
        curation, source_dataset=source_dataset, dataset_format=dataset_format
    )
    paths = manifest.write(out_dir, parquet=parquet)

    raw = Path(out_dir) / "calibra_curation_report.json"
    raw.write_text(curation.model_dump_json(indent=2), encoding="utf-8")
    paths.append(str(raw))
    return paths
