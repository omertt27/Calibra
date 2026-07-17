"""
Integration tests for the LeRobot adapter against real HuggingFace Hub datasets.

These tests require network access and the lerobot optional dependencies:
    pip install 'calibra-robotics[lerobot]'

Run with:
    pytest tests/test_lerobot_adapter.py -v

Skip in offline CI by passing: -m "not integration"
All tests are marked @pytest.mark.integration.

v2 reference dataset : lerobot/pusht            (206 episodes, Parquet v2 format)
v3 reference dataset : lerobot/aloha_mobile_shrimp (18 episodes, small; override via
                       CALIBRA_TEST_V3_DATASET env var if a different v3 dataset is preferred)
"""

from __future__ import annotations

import os

import pytest

from calibra.ingestion.adapters.lerobot import LeRobotReader
from calibra.schema.episode import EpisodeBatch

# Skip the entire module when optional deps are missing.
datasets = pytest.importorskip("datasets", reason="pip install 'calibra-robotics[lerobot]'")
pyarrow = pytest.importorskip("pyarrow", reason="pip install 'calibra-robotics[lerobot]'")

pytestmark = pytest.mark.integration

# ── fixtures ──────────────────────────────────────────────────────────────────

V2_DATASET = "lerobot/pusht"
# 18 episodes — tiny enough for CI. Override with CALIBRA_TEST_V3_DATASET if needed.
V3_DATASET = os.environ.get("CALIBRA_TEST_V3_DATASET", "lerobot/aloha_mobile_shrimp")


@pytest.fixture(scope="module")
def pusht_batch() -> EpisodeBatch:
    """Read lerobot/pusht from the Hub (cached after first run)."""
    return LeRobotReader().read(V2_DATASET)


@pytest.fixture(scope="module")
def kitten_batch() -> EpisodeBatch:
    """Read the v3 reference dataset from the Hub (18 episodes by default)."""
    return LeRobotReader().read(V3_DATASET)


# ── v2: lerobot/pusht ─────────────────────────────────────────────────────────


class TestLeRobotV2PushT:
    def test_returns_episode_batch(self, pusht_batch):
        assert isinstance(pusht_batch, EpisodeBatch)

    def test_format_name(self, pusht_batch):
        assert pusht_batch.format == "lerobot"

    def test_episode_count(self, pusht_batch):
        # pusht ships with 206 demonstrations
        assert pusht_batch.n_episodes == 206

    def test_episodes_have_actions(self, pusht_batch):
        for ep in pusht_batch.episodes:
            assert ep.actions.ndim == 2
            assert ep.actions.shape[1] > 0

    def test_timestamps_monotonically_increasing(self, pusht_batch):
        for ep in pusht_batch.episodes:
            diffs = ep.timestamps[1:] - ep.timestamps[:-1]
            assert (diffs > 0).all(), f"non-monotonic timestamps in {ep.metadata.episode_id}"

    def test_no_image_columns_in_observations(self, pusht_batch):
        for ep in pusht_batch.episodes:
            for key in ep.observations:
                assert "image" not in key.lower(), f"image column leaked: {key}"

    def test_can_read_hub_id(self):
        assert LeRobotReader.can_read(V2_DATASET)

    def test_can_read_hf_uri(self):
        assert LeRobotReader.can_read(f"hf://{V2_DATASET}")

    def test_dataset_name_set(self, pusht_batch):
        assert pusht_batch.dataset_name != ""

    def test_all_episodes_have_nonzero_steps(self, pusht_batch):
        for ep in pusht_batch.episodes:
            assert len(ep.timestamps) > 0

    def test_action_dim_consistent_across_episodes(self, pusht_batch):
        dims = {ep.actions.shape[1] for ep in pusht_batch.episodes}
        assert len(dims) == 1, f"inconsistent action dims across episodes: {dims}"

    def test_hf_uri_matches_hub_id_result(self):
        batch_id = LeRobotReader().read(V2_DATASET)
        batch_uri = LeRobotReader().read(f"hf://{V2_DATASET}")
        assert batch_id.n_episodes == batch_uri.n_episodes
        assert batch_id.n_samples == batch_uri.n_samples


# ── v3: lerobot/aloha_mobile_shrimp (or CALIBRA_TEST_V3_DATASET) ─────────────


class TestLeRobotV3Kitten:
    def test_returns_episode_batch(self, kitten_batch):
        assert isinstance(kitten_batch, EpisodeBatch)

    def test_format_name(self, kitten_batch):
        assert kitten_batch.format == "lerobot"

    def test_has_episodes(self, kitten_batch):
        assert kitten_batch.n_episodes > 0

    def test_episodes_have_actions(self, kitten_batch):
        for ep in kitten_batch.episodes:
            assert ep.actions.ndim == 2
            assert ep.actions.shape[1] > 0

    def test_no_image_columns_in_observations(self, kitten_batch):
        for ep in kitten_batch.episodes:
            for key in ep.observations:
                assert "image" not in key.lower(), f"image column leaked: {key}"

    def test_timestamps_monotonically_increasing(self, kitten_batch):
        for ep in kitten_batch.episodes:
            diffs = ep.timestamps[1:] - ep.timestamps[:-1]
            assert (diffs > 0).all(), f"non-monotonic timestamps in {ep.metadata.episode_id}"

    def test_action_dim_consistent(self, kitten_batch):
        dims = {ep.actions.shape[1] for ep in kitten_batch.episodes}
        assert len(dims) == 1


# ── lazy streaming (v2/v3) ────────────────────────────────────────────────────


class TestLeRobotLazyStreaming:
    """iter_episodes_lazy requires a local v2/v3 dataset; download first."""

    def test_iter_lazy_v2_local(self, tmp_path):
        """Download pusht to a local cache dir, then verify lazy iteration."""
        hf_datasets = pytest.importorskip("datasets")

        ds = hf_datasets.load_dataset(V2_DATASET, split="train")
        cache = tmp_path / "pusht_local"
        ds.save_to_disk(str(cache))

        reader = LeRobotReader()
        # saved_to_disk produces v1 layout — lazy iter requires v2/v3
        with pytest.raises(ValueError, match="v2"):
            list(reader.iter_episodes_lazy(str(cache)))

    def test_iter_lazy_requires_local_v2(self):
        reader = LeRobotReader()
        with pytest.raises(ValueError, match="v2"):
            list(reader.iter_episodes_lazy("/nonexistent/path"))


# ── registry integration ──────────────────────────────────────────────────────


class TestLeRobotRegistryIntegration:
    def test_auto_detect_hub_id(self):
        from calibra.ingestion import registry

        reader_cls = registry.detect_reader(V2_DATASET)
        assert reader_cls is LeRobotReader

    def test_auto_detect_hf_uri(self):
        from calibra.ingestion import registry

        reader_cls = registry.detect_reader(f"hf://{V2_DATASET}")
        assert reader_cls is LeRobotReader
