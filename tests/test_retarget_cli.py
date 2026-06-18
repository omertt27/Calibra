"""
Tests for calibra retarget — the `calibra retarget` CLI command.

Tests the `run_retarget` function (internal entry point) using a temporary
directory of synthetic HDF5-like input, mocked via an in-memory EpisodeBatch
so no real dataset is required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

scipy = pytest.importorskip("scipy", reason="scipy required for kinematics")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_eef_batch(n_episodes: int = 3, n_steps: int = 20):
    """Return a synthetic EpisodeBatch with eef_pos + eef_quat observations."""
    from scipy.spatial.transform import Rotation
    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    rng = np.random.default_rng(42)
    episodes = []
    for i in range(n_episodes):
        pos  = rng.random((n_steps, 3)).astype(np.float64) * 0.5
        quat = Rotation.random(n_steps, random_state=i).as_quat().astype(np.float64)
        episodes.append(Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
            timestamps=np.arange(n_steps, dtype=float) * 0.02,
            observations={"eef_pos": pos, "eef_quat": quat,
                          "proprio": rng.random((n_steps, 7)).astype(np.float32)},
            actions=rng.random((n_steps, 6)).astype(np.float32),
        ))

    return EpisodeBatch(
        episodes=episodes,
        dataset_name="test_eef",
        format="hdf5",
        source_path="/tmp/test_eef.h5",
    )


def _make_batch_no_eef(n_episodes: int = 3, n_steps: int = 20):
    """Return a batch with NO EEF observation keys."""
    from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

    rng = np.random.default_rng(0)
    episodes = [
        Episode(
            metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
            timestamps=np.arange(n_steps, dtype=float) * 0.02,
            observations={"proprio": rng.random((n_steps, 7)).astype(np.float32)},
            actions=rng.random((n_steps, 6)).astype(np.float32),
        )
        for i in range(n_episodes)
    ]
    return EpisodeBatch(
        episodes=episodes, dataset_name="no_eef",
        format="hdf5", source_path="/tmp/no_eef.h5",
    )


def _run(argv: list[str], batch=None) -> int:
    """
    Run `run_retarget` with mocked dataset loader.
    Returns sys.exit code or 0 on success.
    """
    from calibra.retarget import run_retarget

    if batch is None:
        batch = _make_eef_batch()

    with patch("calibra.retarget.run_retarget.__module__"):
        pass  # no-op to import check

    with patch("calibra.ingestion.registry.load", return_value=batch):
        try:
            run_retarget(argv)
            return 0
        except SystemExit as e:
            return int(e.code) if e.code is not None else 0


# ── output file tests ─────────────────────────────────────────────────────────

class TestRetargetCLIOutputFiles:
    def test_creates_npz_per_episode(self):
        batch = _make_eef_batch(n_episodes=4, n_steps=15)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget(["/dummy/path", "--out", tmp])

            files = list(Path(tmp).glob("*.npz"))
            assert len(files) == 4

    def test_npz_contains_relative_actions(self):
        batch = _make_eef_batch(n_episodes=2, n_steps=10)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget(["/dummy/path", "--out", tmp])

            for f in Path(tmp).glob("*.npz"):
                data = np.load(f, allow_pickle=True)
                assert "relative_actions" in data
                arr = data["relative_actions"]
                # T=10, no pad → shape should be (9, 6)
                assert arr.shape == (9, 6)
                assert arr.dtype == np.float32

    def test_action_shape_is_T_minus_1_by_6(self):
        n_steps = 20
        batch = _make_eef_batch(n_episodes=1, n_steps=n_steps)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget(["/dummy/path", "--out", tmp])

            f = next(Path(tmp).glob("*.npz"))
            data = np.load(f)
            assert data["relative_actions"].shape == (n_steps - 1, 6)

    def test_pad_flag_extends_to_T_rows(self):
        n_steps = 20
        batch = _make_eef_batch(n_episodes=1, n_steps=n_steps)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget(["/dummy/path", "--out", tmp, "--pad"])

            f = next(Path(tmp).glob("*.npz"))
            data = np.load(f)
            assert data["relative_actions"].shape == (n_steps, 6)

    def test_pad_last_row_is_zero(self):
        batch = _make_eef_batch(n_episodes=1, n_steps=15)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget(["/dummy/path", "--out", tmp, "--pad"])

            f = next(Path(tmp).glob("*.npz"))
            data = np.load(f)
            np.testing.assert_allclose(data["relative_actions"][-1], 0.0)

    def test_out_dir_created_if_missing(self):
        batch = _make_eef_batch(n_episodes=1, n_steps=10)
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "nested" / "out"
            assert not new_dir.exists()
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget(["/dummy/path", "--out", str(new_dir)])
            assert new_dir.exists()
            assert list(new_dir.glob("*.npz"))


# ── explicit key override ─────────────────────────────────────────────────────

class TestRetargetExplicitKeys:
    def test_custom_obs_keys(self):
        """--obs-key-pos and --obs-key-quat override auto-detection."""
        from scipy.spatial.transform import Rotation
        from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata

        rng = np.random.default_rng(7)
        n = 15
        ep = Episode(
            metadata=EpisodeMetadata(episode_id="ep_0"),
            timestamps=np.arange(n, dtype=float) * 0.02,
            observations={
                "my_pos":  rng.random((n, 3)).astype(np.float64),
                "my_quat": Rotation.random(n, random_state=7).as_quat().astype(np.float64),
            },
            actions=rng.random((n, 6)).astype(np.float32),
        )
        batch = EpisodeBatch(episodes=[ep], dataset_name="custom_keys",
                             format="hdf5", source_path="/tmp/x.h5")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget([
                    "/dummy/path", "--out", tmp,
                    "--obs-key-pos", "my_pos",
                    "--obs-key-quat", "my_quat",
                ])
            files = list(Path(tmp).glob("*.npz"))
            assert len(files) == 1


# ── error / exit-code tests ───────────────────────────────────────────────────

class TestRetargetCLIErrors:
    def test_exits_2_when_no_eef_keys(self):
        """Exit code 2 when no EEF keys are found in any episode."""
        batch = _make_batch_no_eef()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                with pytest.raises(SystemExit) as exc_info:
                    run_retarget(["/dummy/path", "--out", tmp])
                assert exc_info.value.code == 2

    def test_exits_1_on_load_failure(self):
        """Exit code 1 when dataset loading raises an exception."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", side_effect=RuntimeError("no data")):
                from calibra.retarget import run_retarget
                with pytest.raises(SystemExit) as exc_info:
                    run_retarget(["/dummy/path", "--out", tmp])
                assert exc_info.value.code == 1


# ── JSON output ───────────────────────────────────────────────────────────────

class TestRetargetJSONOutput:
    def test_json_flag_prints_valid_json(self, capsys):
        batch = _make_eef_batch(n_episodes=2, n_steps=10)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("calibra.ingestion.registry.load", return_value=batch):
                from calibra.retarget import run_retarget
                run_retarget(["/dummy/path", "--out", tmp, "--json"])

            captured = capsys.readouterr()
            # stdout contains both the summary banner and the JSON block.
            # Find the JSON part (starts after the banner).
            json_start = captured.out.find("{")
            assert json_start != -1, "No JSON found in stdout"
            payload = json.loads(captured.out[json_start:])
            assert payload["n_converted"] == 2
            assert payload["n_skipped"] == 0
            assert "converted_ids" in payload
