"""Unit tests for the ForceTorqueContactAnalyzer."""

from __future__ import annotations

import numpy as np

from calibra.analyzers.force_torque import ForceTorqueContactAnalyzer
from calibra.schema.episode import Episode, EpisodeBatch, EpisodeMetadata
from calibra.schema.report import RiskLevel


def _make_ft_batch(
    n_eps: int = 3,
    n_steps: int = 50,
    include_ft: bool = True,
    include_contact: bool = True,
    inject_spikes: bool = False,
    stuck_contact: bool = False,
) -> EpisodeBatch:
    rng = np.random.default_rng(10)
    episodes = []
    for i in range(n_eps):
        ts = np.arange(n_steps, dtype=np.float64) * 0.1
        obs = {}
        if include_ft:
            # Generate nominal force/torque data
            ft = rng.normal(0.0, 1.0, (n_steps, 6)).astype(np.float32)
            if inject_spikes and i == 0:
                # Inject a huge force spike/shock
                ft[25] = np.array([500.0, 500.0, 500.0, 0.0, 0.0, 0.0])
            obs["force_torque"] = ft
        if include_contact:
            # Contact state (boolean/float thresholded at 0.5)
            if stuck_contact:
                # Contact dropout (nearly 0 contacts)
                contact = np.zeros(n_steps, dtype=np.float32)
            else:
                contact = rng.uniform(0.0, 1.0, n_steps).astype(np.float32)
            obs["contact_sensor"] = contact

        episodes.append(
            Episode(
                metadata=EpisodeMetadata(episode_id=f"ep_{i}"),
                timestamps=ts,
                observations=obs,
                actions=rng.normal(0.0, 0.1, (n_steps, 2)).astype(np.float32),
            )
        )
    return EpisodeBatch(
        episodes=episodes,
        dataset_name="ft_test",
        format="hdf5",
        source_path="/tmp/ft_test.h5",
    )


class TestForceTorqueContactAnalyzer:
    def test_skips_when_no_modalities_present(self):
        batch = _make_ft_batch(include_ft=False, include_contact=False)
        analyzer = ForceTorqueContactAnalyzer()
        result = analyzer.analyze(batch)
        assert result.analyzer_name == "force_torque"
        assert result.flags == []
        assert result.raw_metrics == {}

    def test_analyzes_nominal_force_torque_and_contacts(self):
        batch = _make_ft_batch(include_ft=True, include_contact=True)
        analyzer = ForceTorqueContactAnalyzer()
        result = analyzer.analyze(batch)

        assert result.analyzer_name == "force_torque"
        assert "force_keys_found" in result.raw_metrics
        assert "contact_keys_found" in result.raw_metrics
        assert "force_torque" in result.raw_metrics["force_keys_found"]
        assert "contact_sensor" in result.raw_metrics["contact_keys_found"]

    def test_detects_force_spikes(self):
        batch = _make_ft_batch(include_ft=True, include_contact=True, inject_spikes=True)
        analyzer = ForceTorqueContactAnalyzer()
        result = analyzer.analyze(batch)

        # Should flag force spike warnings
        spike_flags = [f for f in result.flags if f.metric == "force_spike_rate"]
        assert len(spike_flags) > 0
        assert spike_flags[0].level == RiskLevel.WARNING

    def test_detects_contact_dropout(self):
        batch = _make_ft_batch(include_ft=True, include_contact=True, stuck_contact=True)
        analyzer = ForceTorqueContactAnalyzer()
        result = analyzer.analyze(batch)

        # Should flag contact dropout critical
        dropout_flags = [f for f in result.flags if f.metric == "contact_dropout"]
        assert len(dropout_flags) > 0
        assert dropout_flags[0].level == RiskLevel.CRITICAL
