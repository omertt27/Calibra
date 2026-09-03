"""
Tests for ADR-011 annotate mode: the annotation sidecar schema, the builder,
and the write/load round-trip.

All fixtures are synthetic — no real dataset files required.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calibra.annotate import build_annotation_manifest, write_annotations
from calibra.pipeline import Pipeline
from calibra.pruning import CoresetSelector, pruning_result_to_curation_report
from calibra.schema.annotations import (
    ANNOTATION_SCHEMA_VERSION,
    FIELD_DOCS,
    AnnotationManifest,
    EpisodeAnnotation,
)
from calibra.schema.comparison import (
    CurationReport,
    Disposition,
    EpisodeCharacterization,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_ep(episode_id: str, n_steps: int = 60) -> Episode:
    """Episode with a smooth (bounded-jerk) trajectory so it clears Stage 1."""
    rng = np.random.default_rng(abs(hash(episode_id)) % (2**32))
    t = np.arange(n_steps) * 0.02
    actions = np.zeros((n_steps, 6), dtype=np.float32)
    for d in range(6):
        for h in range(1, 3):
            freq = h * rng.uniform(0.05, 0.12)
            amp = rng.uniform(0.05, 0.15) / h
            actions[:, d] += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        actions[:, d] += rng.uniform(-0.3, 0.3)
    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=t,
        observations={"state": rng.random((n_steps, 6)).astype(np.float32)},
        actions=actions,
    )


def _curation(dispositions: list[EpisodeCharacterization]) -> CurationReport:
    retained = sum(
        1 for d in dispositions if d.disposition in {Disposition.KEEP, Disposition.ANNOTATE}
    )
    return CurationReport(
        original_n_episodes=len(dispositions),
        retained_n_episodes=retained,
        dispositions=dispositions,
    )


def _sample_dispositions() -> list[EpisodeCharacterization]:
    return [
        EpisodeCharacterization(
            episode_index=0,
            episode_id="ep_0",
            disposition=Disposition.KEEP,
            n_steps=60,
            calibra_score=99.0,
            quality_risk=0.01,
            anomaly_score=0.2,
            coverage_value=0.8,
            redundancy=0.2,
            success=True,
        ),
        EpisodeCharacterization(
            episode_index=1,
            episode_id="ep_1",
            disposition=Disposition.DROP,
            n_steps=55,
            calibra_score=10.0,
            quality_risk=0.9,
            anomaly_score=0.95,
            integrity_flags=["jerk_spike"],
            reasons=["jerk spike rate exceeded threshold"],
            success=False,
        ),
        EpisodeCharacterization(
            episode_index=2,
            episode_id="ep_2",
            disposition=Disposition.ANNOTATE,
            n_steps=60,
            calibra_score=95.0,
            quality_risk=0.05,
            anomaly_score=0.1,
            coverage_value=0.15,
            redundancy=0.85,
            reasons=["diversity_pruned"],
        ),
    ]


# ── schema / builder ─────────────────────────────────────────────────────────


def test_build_manifest_rows_match_dispositions():
    curation = _curation(_sample_dispositions())
    manifest = build_annotation_manifest(
        curation, source_dataset="/data/ds.h5", dataset_format="hdf5"
    )

    assert manifest.schema_version == ANNOTATION_SCHEMA_VERSION
    assert manifest.n_episodes == 3
    assert manifest.source_dataset == "/data/ds.h5"
    assert manifest.dataset_format == "hdf5"
    assert manifest.disposition_counts == {"KEEP": 1, "DROP": 1, "ANNOTATE": 1}
    assert manifest.field_docs == FIELD_DOCS
    assert manifest.calibra_version

    row = {r.episode_id: r for r in manifest.annotations}
    assert row["ep_1"].calibra_disposition == "DROP"
    assert row["ep_1"].integrity_flags == ["jerk_spike"]
    assert row["ep_2"].calibra_disposition == "ANNOTATE"
    assert row["ep_0"].coverage_value == pytest.approx(0.8)
    # characterization columns propagate from the CurationReport
    assert row["ep_0"].calibra_score == pytest.approx(99.0)
    assert row["ep_2"].redundancy == pytest.approx(0.85)
    assert row["ep_0"].success is True
    assert row["ep_1"].success is False
    assert row["ep_2"].success is None


def test_field_docs_cover_every_column():
    row = build_annotation_manifest(
        _curation(_sample_dispositions()), source_dataset="x"
    ).annotations[0]
    assert set(FIELD_DOCS) == set(row.model_dump().keys())
    assert "1.0" in FIELD_DOCS["weight"]  # null-weight convention documented


# ── write / load round-trip ──────────────────────────────────────────────────


def test_write_splits_rows_and_header(tmp_path):
    curation = _curation(_sample_dispositions())
    paths = write_annotations(
        curation, str(tmp_path), source_dataset="/data/ds.h5", dataset_format="hdf5"
    )

    names = {p.rsplit("/", 1)[-1] for p in paths}
    assert names == {
        "calibra_annotations.jsonl",
        "calibra_annotations.manifest.json",
        "calibra_curation_report.json",
    }

    jsonl = (tmp_path / "calibra_annotations.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 3
    assert all(json.loads(line)["episode_id"].startswith("ep_") for line in jsonl)

    header = json.loads((tmp_path / "calibra_annotations.manifest.json").read_text())
    assert "annotations" not in header  # rows live only in the jsonl
    assert header["disposition_counts"] == {"KEEP": 1, "DROP": 1, "ANNOTATE": 1}

    raw = json.loads((tmp_path / "calibra_curation_report.json").read_text())
    assert len(raw["dispositions"]) == 3


def test_parquet_output_matches_jsonl(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")

    manifest = build_annotation_manifest(
        _curation(_sample_dispositions()), source_dataset="/data/ds.h5"
    )
    paths = manifest.write(str(tmp_path), parquet=True)
    assert str(tmp_path / "calibra_annotations.parquet") in paths

    table = pq.read_table(tmp_path / "calibra_annotations.parquet")
    assert table.num_rows == 3
    assert set(table.column_names) == set(EpisodeAnnotation.model_fields.keys())
    by_id = {r["episode_id"]: r for r in table.to_pylist()}
    assert by_id["ep_1"]["calibra_disposition"] == "DROP"
    assert by_id["ep_1"]["integrity_flags"] == ["jerk_spike"]
    assert by_id["ep_0"]["calibra_score"] == pytest.approx(99.0)
    assert by_id["ep_2"]["redundancy"] == pytest.approx(0.85)
    # all-null column keeps a real (non-null) arrow type from the explicit schema
    assert str(table.schema.field("weight").type) == "double"


def test_parquet_requested_without_pyarrow_raises(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pyarrow)
    manifest = build_annotation_manifest(_curation(_sample_dispositions()), source_dataset="x")
    with pytest.raises(ImportError, match="pyarrow"):
        manifest.write(str(tmp_path), parquet=True)


def test_load_round_trips(tmp_path):
    manifest = build_annotation_manifest(
        _curation(_sample_dispositions()), source_dataset="/data/ds.h5"
    )
    manifest.write(str(tmp_path))

    loaded = AnnotationManifest.load(str(tmp_path))
    assert loaded.schema_version == manifest.schema_version
    assert loaded.n_episodes == manifest.n_episodes
    assert loaded.disposition_counts == manifest.disposition_counts
    assert [r.model_dump() for r in loaded.annotations] == [
        r.model_dump() for r in manifest.annotations
    ]


# ── end-to-end from a prune run ──────────────────────────────────────────────


def test_annotate_marks_redundant_as_annotate(tmp_path):
    episodes = [_make_ep(f"ep_{i}") for i in range(10)]
    batch = EpisodeBatch(
        episodes=episodes, dataset_name="d", format="hdf5", source_path="/tmp/d.h5"
    )
    report = Pipeline().run(batch)
    result = CoresetSelector(
        keep_fraction=0.4,
        max_spike_rate=1.0,
        max_vel_disc_rate=1.0,
        max_dropout_fraction=1.0,
        min_ldlj=-1e6,
        strategy="diversity",
    ).select(batch, report)

    curation = pruning_result_to_curation_report(
        result, batch, report=report, redundant_disposition=Disposition.ANNOTATE
    )
    write_annotations(curation, str(tmp_path), source_dataset="/tmp/d.h5", dataset_format="hdf5")
    loaded = AnnotationManifest.load(str(tmp_path))

    # nothing failed quality (max_spike_rate=1.0) → diversity-pruned → ANNOTATE
    assert "ANNOTATE" in loaded.disposition_counts
    assert loaded.disposition_counts.get("DROP", 0) == 0
    keep = loaded.disposition_counts["KEEP"]
    annot = loaded.disposition_counts["ANNOTATE"]
    assert keep + annot == 10
    assert keep == len(result.keep_episode_ids)
    # ANNOTATE rows are the ones vanilla pruning would have removed
    annotated_ids = {
        r.episode_id for r in loaded.annotations if r.calibra_disposition == "ANNOTATE"
    }
    assert annotated_ids == set(result.diversity_pruned_ids)
    # characterization came through
    assert all(r.anomaly_score is not None for r in loaded.annotations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
