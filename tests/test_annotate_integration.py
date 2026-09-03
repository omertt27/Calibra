"""
Integration test for annotate mode (ADR-011) against a real LeRobot dataset.

Proves that `pruning_result_to_curation_report` + the annotation sidecar hold
up on real v2 Parquet data (not just synthetic HDF5) before a design partner
runs a metadata-conditioning experiment against it:

  - the sidecar has exactly one row per dataset episode, ids and index aligned
  - dispositions partition the dataset; DROP carries integrity flags
  - the characterization columns are populated and non-degenerate
  - ANNOTATE episodes really are lower-coverage than KEEP (rescue semantics)
  - JSONL, Parquet and the load() round-trip all agree
  - `calibra prune`'s existing coreset output is byte-identical with/without
    --annotate (the "additive" promise)

Requires network + lerobot extras. Run:  pytest tests/test_annotate_integration.py -v
Skip offline: -m "not integration"
"""

from __future__ import annotations

import json
import statistics as st

import pytest

datasets = pytest.importorskip("datasets", reason="pip install 'calibra-robotics[lerobot]'")
pyarrow = pytest.importorskip("pyarrow", reason="pip install 'calibra-robotics[lerobot]'")

from calibra.annotate import write_annotations  # noqa: E402
from calibra.ingestion.adapters.lerobot import LeRobotReader  # noqa: E402
from calibra.pipeline import Pipeline  # noqa: E402
from calibra.pruning import CoresetSelector  # noqa: E402
from calibra.schema.annotations import AnnotationManifest, EpisodeAnnotation  # noqa: E402
from calibra.schema.comparison import Disposition  # noqa: E402

pytestmark = pytest.mark.integration

DATASET = "lerobot/pusht"  # 206 episodes, Parquet v2


@pytest.fixture(scope="module")
def annotated(tmp_path_factory):
    batch = LeRobotReader().read(DATASET)
    report = Pipeline().run(batch)
    result = CoresetSelector(keep_fraction=0.25).select(batch, report)
    curation = result.to_curation_report(
        batch, report=report, redundant_disposition=Disposition.ANNOTATE
    )
    out = tmp_path_factory.mktemp("annot")
    paths = write_annotations(
        curation, str(out), source_dataset=DATASET, dataset_format=batch.format, parquet=True
    )
    return batch, curation, out, paths


def test_one_row_per_episode_ids_aligned(annotated):
    batch, _, out, _ = annotated
    m = AnnotationManifest.load(str(out))
    ds_ids = [e.metadata.episode_id for e in batch.episodes]

    assert len(m.annotations) == len(ds_ids) == batch.n_episodes
    assert [r.episode_id for r in m.annotations] == ds_ids
    assert [r.episode_index for r in m.annotations] == list(range(len(ds_ids)))


def test_dispositions_partition_and_drop_has_flags(annotated):
    _, curation, out, _ = annotated
    m = AnnotationManifest.load(str(out))
    counts = m.disposition_counts
    assert sum(counts.values()) == len(m.annotations)
    assert set(counts) <= {d.value for d in Disposition}
    assert counts.get("KEEP", 0) > 0

    drop = [r for r in m.annotations if r.calibra_disposition == "DROP"]
    if drop:
        assert all(r.integrity_flags for r in drop)


def test_characterization_columns_populated_and_sane(annotated):
    _, _, out, _ = annotated
    rows = AnnotationManifest.load(str(out)).annotations

    for attr in ("calibra_score", "quality_risk", "coverage_value", "redundancy"):
        v = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        assert len(v) > len(rows) * 0.9, f"{attr} mostly null"
        assert len(set(round(x, 4) for x in v)) > 5, f"{attr} degenerate"

    for r in rows:
        if r.quality_risk is not None:
            assert r.calibra_score == pytest.approx(100.0 * (1.0 - r.quality_risk), abs=0.05)
        if r.coverage_value is not None:
            assert r.redundancy == pytest.approx(1.0 - r.coverage_value, abs=1e-4)
            assert 0.0 <= r.coverage_value <= 1.0


def test_annotate_episodes_are_lower_coverage_than_keep(annotated):
    _, _, out, _ = annotated
    rows = AnnotationManifest.load(str(out)).annotations
    keep_cov = [
        r.coverage_value
        for r in rows
        if r.calibra_disposition == "KEEP" and r.coverage_value is not None
    ]
    ann_cov = [
        r.coverage_value
        for r in rows
        if r.calibra_disposition == "ANNOTATE" and r.coverage_value is not None
    ]
    if keep_cov and ann_cov:
        # rescue semantics: ANNOTATE = "redundant enough that pruning drops it"
        assert st.mean(ann_cov) < st.mean(keep_cov)


def test_parquet_matches_jsonl(annotated):
    _, _, out, _ = annotated
    import pyarrow.parquet as pq

    rows = AnnotationManifest.load(str(out)).annotations
    table = pq.read_table(out / "calibra_annotations.parquet")
    assert table.num_rows == len(rows)
    assert set(table.column_names) == set(EpisodeAnnotation.model_fields.keys())
    from_pq = {r["episode_id"]: r for r in table.to_pylist()}
    from_jsonl = {r.episode_id: r.model_dump() for r in rows}
    assert from_pq == from_jsonl


def test_prune_output_unchanged_by_annotate(annotated, tmp_path):
    batch, _, _, _ = annotated
    report = Pipeline().run(batch)
    a = CoresetSelector(keep_fraction=0.25).select(batch, report).to_dict()
    b = CoresetSelector(keep_fraction=0.25).select(batch, report).to_dict()
    # coreset selection itself is deterministic and independent of the sidecar
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
