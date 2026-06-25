"""Unit tests for the visual HTML report generator."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from calibra.pipeline import Pipeline
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.report_html import generate_html_report


def _make_batch(n_eps: int = 3, n_steps: int = 20) -> EpisodeBatch:
    rng = np.random.default_rng(42)
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        acts = rng.uniform(-1, 1, (n_steps, 2)).astype(np.float32)
        obs = {
            "proprio": rng.uniform(-1, 1, (n_steps, 4)).astype(np.float32),
            "force_torque": rng.normal(0, 1, (n_steps, 6)).astype(np.float32),
            "contact_sensor": rng.uniform(0, 1, n_steps).astype(np.float32),
        }
        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=ts,
                observations=obs,
                actions=acts,
            )
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="html_test_ds",
        format="hdf5",
        source_path="/tmp/html_test.h5",
    )


class TestReportHTML:
    def test_generate_html_report_creates_file(self):
        batch = _make_batch()
        report = Pipeline().run(batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "report.html"

            # Run generator
            generate_html_report(report, str(out_file), outliers={0: ["ldlj outlier"]})

            # Check file exists and has content
            assert out_file.exists()
            content = out_file.read_text(encoding="utf-8")

            # Assert Tailwind, Chart.js, and dataset name are embedded
            assert "<!DOCTYPE html>" in content
            assert "tailwindcss" in content
            assert "Chart.js" in content
            assert "html_test_ds" in content
            assert "ldlj outlier" in content
            assert "ssl_trajectory_outliers" in content
            assert "contact_dropout" in content
