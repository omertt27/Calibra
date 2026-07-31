"""Tests for opt-in LeRobot v1 image decoding (calibra/ingestion/adapters/lerobot.py)."""

from __future__ import annotations

import io

import numpy as np
import pytest

from calibra.analyzers.blur import BlurAnalyzer
from calibra.analyzers.camera_freeze import CameraFreezeAnalyzer
from calibra.analyzers.duplicate_frame import DuplicateFrameAnalyzer
from calibra.ingestion.adapters.lerobot import LeRobotReader, _decode_image_column
from calibra.schema.report import RiskLevel

datasets = pytest.importorskip("datasets")

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_v1_dataset(tmp_path, n_episodes: int = 6, n_steps: int = 10, freeze_ep: int = 0):
    """
    Build a small local v1 (HuggingFace `datasets`-saved) LeRobot dataset with
    a real Image-feature column, save it to disk, and return the path.

    `freeze_ep` gets a run of identical frames so CameraFreezeAnalyzer has
    something real to flag once images are decoded.
    """
    from datasets import Dataset, Features, Sequence, Value
    from datasets import Image as HFImage

    rng = np.random.default_rng(0)
    episode_index, frame_index, timestamp, action, images = [], [], [], [], []
    for ep in range(n_episodes):
        frames = rng.integers(0, 255, (n_steps, 16, 16, 3), dtype=np.uint8)
        if ep == freeze_ep:
            frames[:8] = frames[0]
        for step in range(n_steps):
            episode_index.append(ep)
            frame_index.append(step)
            timestamp.append(step * 0.05)
            action.append(rng.uniform(-1, 1, 6).astype(np.float32).tolist())
            images.append(frames[step])

    ds = Dataset.from_dict(
        {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "action": action,
            "observation.images.top": images,
        },
        features=Features(
            {
                "episode_index": Value("int64"),
                "frame_index": Value("int64"),
                "timestamp": Value("float32"),
                "action": Sequence(Value("float32"), length=6),
                "observation.images.top": HFImage(),
            }
        ),
    )
    path = tmp_path / "v1_dataset"
    ds.save_to_disk(str(path))
    return str(path)


# ── _decode_image_column ────────────────────────────────────────────────────


class TestDecodeImageColumn:
    def test_decodes_valid_png_bytes(self):
        from PIL import Image as PILImage

        img = PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cells = [{"bytes": buf.getvalue(), "path": None}]
        result = _decode_image_column(cells)
        assert result is not None
        assert result.shape == (1, 4, 4, 3)
        assert result.dtype == np.uint8

    def test_non_image_cells_return_none(self):
        assert _decode_image_column([{"not": "an image"}]) is None
        assert _decode_image_column([42, 43]) is None


# ── LeRobotReader(decode_images=...) ────────────────────────────────────────


class TestLeRobotV1ImageDecoding:
    def test_default_excludes_images(self, tmp_path):
        path = _make_v1_dataset(tmp_path)
        batch = LeRobotReader().read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        assert not any("image" in k.lower() or "camera" in k.lower() for k in obs_keys)
        assert "images" not in batch.capabilities

    def test_decode_images_true_includes_images(self, tmp_path):
        path = _make_v1_dataset(tmp_path)
        batch = LeRobotReader(decode_images=True).read(path)
        obs_keys = set(batch.episodes[0].observations.keys())
        image_keys = [k for k in obs_keys if "camera" in k.lower() or "image" in k.lower()]
        assert image_keys, f"expected an image observation key, got {obs_keys}"
        arr = batch.episodes[0].observations[image_keys[0]]
        assert arr.ndim == 4  # (T, H, W, C)
        assert arr.dtype == np.uint8
        assert "images" in batch.capabilities

    def test_end_to_end_image_analyzers_find_real_signal(self, tmp_path):
        path = _make_v1_dataset(tmp_path, freeze_ep=0)
        batch = LeRobotReader(decode_images=True).read(path)

        dup_result = DuplicateFrameAnalyzer().analyze(batch)
        freeze_result = CameraFreezeAnalyzer(min_freeze_run=5).analyze(batch)
        blur_result = BlurAnalyzer().analyze(batch)

        for result in (dup_result, freeze_result, blur_result):
            assert result.flags, f"{result.analyzer_name} produced no flags"
            assert result.flags[0].level != RiskLevel.INFO, (
                f"{result.analyzer_name} returned INFO — found no image data to work with"
            )
        assert freeze_result.flags[0].level in (RiskLevel.WARNING, RiskLevel.CRITICAL)
