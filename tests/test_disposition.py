"""
Tests for ADR-011 dataset decision layer: the Disposition enum,
EpisodeCharacterization, and CurationReport's dispositions ⇄ indices sync.

All fixtures are synthetic — no real dataset files required.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibra.comparison.curator import EpisodeCurator
from calibra.pipeline import Pipeline
from calibra.schema.comparison import (
    CurationReport,
    Disposition,
    EpisodeCharacterization,
)
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_ep(n_steps: int = 100, seed: int = 0, episode_id: str = "ep_0") -> Episode:
    rng = np.random.default_rng(seed)
    dt = 0.1
    timestamps = np.arange(n_steps, dtype=np.float64) * dt
    return Episode(
        metadata=EpisodeMetadata(episode_id=episode_id),
        timestamps=timestamps,
        observations={"proprio": rng.random((n_steps, 4)).astype(np.float32)},
        actions=rng.random((n_steps, 4)).astype(np.float32),
    )


def _batch_with_one_short(n_episodes: int = 6, short_steps: int = 10) -> EpisodeBatch:
    episodes = [_make_ep(n_steps=short_steps, seed=0, episode_id="ep_0")]
    episodes += [
        _make_ep(n_steps=100, seed=i + 1, episode_id=f"ep_{i + 1}") for i in range(n_episodes - 1)
    ]
    return EpisodeBatch(
        episodes=episodes, dataset_name="mixed", format="hdf5", source_path="/tmp/mixed.h5"
    )


# ── Disposition enum ──────────────────────────────────────────────────────────


def test_disposition_is_closed_str_enum():
    assert Disposition.KEEP.value == "KEEP"
    assert Disposition("DROP") is Disposition.DROP
    assert {d.value for d in Disposition} == {
        "KEEP",
        "DROP",
        "DOWNWEIGHT",
        "ANNOTATE",
        "REVIEW",
        "RECOLLECT",
    }


def test_disposition_json_round_trips_as_string():
    rec = EpisodeCharacterization(
        episode_index=3, episode_id="ep_3", disposition=Disposition.ANNOTATE
    )
    dumped = rec.model_dump_json()
    assert '"disposition":"ANNOTATE"' in dumped
    assert EpisodeCharacterization.model_validate_json(dumped).disposition is Disposition.ANNOTATE


def test_characterization_numeric_fields_default_none():
    rec = EpisodeCharacterization(episode_index=0, episode_id="ep_0")
    assert rec.disposition is Disposition.KEEP
    assert rec.quality_risk is None
    assert rec.coverage_value is None
    assert rec.weight is None
    assert rec.integrity_flags == []
    assert rec.reasons == []


# ── CurationReport dispositions ⇄ indices sync ────────────────────────────────


def test_dispositions_derive_legacy_indices():
    """Passing only dispositions fills in retained/dropped indices."""
    dispositions = [
        EpisodeCharacterization(episode_index=0, episode_id="a", disposition=Disposition.DROP),
        EpisodeCharacterization(episode_index=1, episode_id="b", disposition=Disposition.KEEP),
        EpisodeCharacterization(episode_index=2, episode_id="c", disposition=Disposition.DOWNWEIGHT),
        EpisodeCharacterization(episode_index=3, episode_id="d", disposition=Disposition.ANNOTATE),
        EpisodeCharacterization(episode_index=4, episode_id="e", disposition=Disposition.REVIEW),
    ]
    report = CurationReport(
        original_n_episodes=5, retained_n_episodes=3, dispositions=dispositions
    )
    # KEEP / DOWNWEIGHT / ANNOTATE are "in the training set" → retained.
    assert report.retained_indices == [1, 2, 3]
    assert report.dropped_indices == [0, 4]


def test_legacy_indices_derive_dispositions():
    """Passing only retained/dropped indices fills in a minimal dispositions list."""
    report = CurationReport(
        original_n_episodes=4,
        retained_n_episodes=2,
        retained_indices=[1, 3],
        dropped_indices=[0, 2],
    )
    assert len(report.dispositions) == 4
    by_idx = {d.episode_index: d.disposition for d in report.dispositions}
    assert by_idx == {
        0: Disposition.DROP,
        1: Disposition.KEEP,
        2: Disposition.DROP,
        3: Disposition.KEEP,
    }


def test_both_supplied_are_left_untouched():
    dispositions = [
        EpisodeCharacterization(episode_index=0, episode_id="a", disposition=Disposition.KEEP),
        EpisodeCharacterization(episode_index=1, episode_id="b", disposition=Disposition.DROP),
    ]
    report = CurationReport(
        original_n_episodes=2,
        retained_n_episodes=1,
        retained_indices=[0],
        dropped_indices=[1],
        dispositions=dispositions,
    )
    assert report.retained_indices == [0]
    assert report.dropped_indices == [1]
    assert [d.disposition for d in report.dispositions] == [Disposition.KEEP, Disposition.DROP]


def test_disposition_counts_and_by_disposition():
    dispositions = [
        EpisodeCharacterization(episode_index=0, episode_id="a", disposition=Disposition.KEEP),
        EpisodeCharacterization(episode_index=1, episode_id="b", disposition=Disposition.KEEP),
        EpisodeCharacterization(episode_index=2, episode_id="c", disposition=Disposition.DROP),
        EpisodeCharacterization(episode_index=3, episode_id="d", disposition=Disposition.REVIEW),
    ]
    report = CurationReport(
        original_n_episodes=4, retained_n_episodes=2, dispositions=dispositions
    )
    assert report.disposition_counts() == {"KEEP": 2, "DROP": 1, "REVIEW": 1}
    reviewed = report.by_disposition(Disposition.REVIEW)
    assert [r.episode_id for r in reviewed] == ["d"]


def test_summary_lists_non_keepdrop_dispositions():
    dispositions = [
        EpisodeCharacterization(episode_index=0, episode_id="a", disposition=Disposition.KEEP),
        EpisodeCharacterization(episode_index=1, episode_id="b", disposition=Disposition.REVIEW),
    ]
    report = CurationReport(
        original_n_episodes=2, retained_n_episodes=1, dispositions=dispositions
    )
    s = report.summary()
    assert "Calibra Curation Report" in s
    assert "review=1" in s


# ── EpisodeCurator now emits dispositions ────────────────────────────────────


def test_curator_emits_aligned_dispositions():
    batch = _batch_with_one_short(n_episodes=6, short_steps=10)
    report = Pipeline().run(batch)

    curator = EpisodeCurator(min_length=50)
    _, curation_report = curator.curate(batch, report)

    # one characterization per original episode, in order
    assert [d.episode_index for d in curation_report.dispositions] == list(range(6))
    assert [d.episode_id for d in curation_report.dispositions] == [
        ep.metadata.episode_id for ep in batch.episodes
    ]

    ep0 = curation_report.dispositions[0]
    assert ep0.disposition is Disposition.DROP
    assert ep0.n_steps == 10
    assert "length" in ep0.integrity_flags
    assert ep0.reasons and "min_length" in ep0.reasons[0]

    for d in curation_report.dispositions[1:]:
        assert d.disposition is Disposition.KEEP
        assert d.integrity_flags == []
        assert d.reasons == []


def test_curator_dispositions_consistent_with_legacy_indices():
    batch = _batch_with_one_short(n_episodes=8, short_steps=10)
    report = Pipeline().run(batch)

    curator = EpisodeCurator(min_length=50)
    _, curation_report = curator.curate(batch, report)

    keep_from_disp = sorted(
        d.episode_index
        for d in curation_report.dispositions
        if d.disposition is Disposition.KEEP
    )
    assert keep_from_disp == sorted(curation_report.retained_indices)
    assert curation_report.disposition_counts() == {"KEEP": 7, "DROP": 1}


def test_curator_no_thresholds_all_keep():
    batch = _batch_with_one_short()
    report = Pipeline().run(batch)

    _, curation_report = EpisodeCurator().curate(batch, report)

    assert all(d.disposition is Disposition.KEEP for d in curation_report.dispositions)
    assert curation_report.disposition_counts() == {"KEEP": batch.n_episodes}


def test_curator_dispositions_carry_assessment_axes():
    """dispositions now carry anomaly_score / quality_risk / coverage_value."""
    batch = _batch_with_one_short(n_episodes=6, short_steps=10)
    report = Pipeline().run(batch)

    _, curation_report = EpisodeCurator(min_length=50).curate(batch, report)

    for d in curation_report.dispositions:
        assert d.quality_risk is not None
        assert 0.0 <= d.quality_risk <= 1.0
        assert d.anomaly_score is not None
        assert 0.0 <= d.anomaly_score <= 1.0
        assert d.coverage_value is None or 0.0 <= d.coverage_value <= 1.0

    # ep_0 is a length outlier → its anomaly_score should lead the batch
    ep0 = curation_report.dispositions[0]
    assert ep0.anomaly_score == max(d.anomaly_score for d in curation_report.dispositions)


def test_curator_empty_batch():
    batch = EpisodeBatch(
        episodes=[], dataset_name="empty", format="hdf5", source_path="/tmp/empty.h5"
    )
    report = Pipeline().run(batch)
    _, curation_report = EpisodeCurator(min_length=50).curate(batch, report)
    assert curation_report.dispositions == []
    assert curation_report.disposition_counts() == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
