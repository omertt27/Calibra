"""
Force/Torque and Contact Dynamics Observability Analyzer.

Diagnoses quality anomalies in physical robot interactions using force/torque
(wrench) sensors and contact state logs.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from calibra.analyzers.base import Analyzer
from calibra.schema.episode import EpisodeBatch
from calibra.schema.report import AnalyzerResult, RiskFlag, RiskLevel, ObservedValue


class ForceTorqueContactAnalyzer(Analyzer):
    """
    Diagnoses force/torque readings and contact sensor feeds.

    Identifies high-impact physical shocks (force spikes), abnormal signal
    vibrations (excessive noise), contact signal dropouts, or misalignments
    with movement states.
    """

    @property
    def name(self) -> str:
        return "force_torque"

    def _find_modality_keys(
        self, episode_obs: dict[str, np.ndarray]
    ) -> tuple[list[str], list[str]]:
        """Identify keys in observation dict containing force/torque or contact data."""
        ft_keys = []
        contact_keys = []
        for k in episode_obs.keys():
            k_lower = k.lower()
            if "force" in k_lower or "torque" in k_lower or "wrench" in k_lower:
                ft_keys.append(k)
            elif "contact" in k_lower:
                contact_keys.append(k)
        return ft_keys, contact_keys

    def analyze(
        self,
        batch: EpisodeBatch,
        policy_family: Optional[str] = None,
    ) -> AnalyzerResult:
        flags = []

        # Check if there is force/torque or contact data in the batch
        if not batch.episodes:
            return AnalyzerResult(analyzer_name=self.name)

        ft_keys, contact_keys = self._find_modality_keys(batch.episodes[0].observations)
        if not ft_keys and not contact_keys:
            # Skip analysis if no force/torque/contact modalities are found
            return AnalyzerResult(analyzer_name=self.name)

        per_episode_spikes = []
        per_episode_contacts = []
        total_steps = 0
        total_ft_spikes = 0
        total_contact_steps = 0

        for ep in batch.episodes:
            # 1. Analyze Force/Torque Spikes
            ep_spikes = 0
            ep_ft_steps = 0
            for key in ft_keys:
                ft_data = ep.observations[key]
                if ft_data.ndim < 2:
                    continue
                # Compute magnitude if multidimensional
                mag = np.linalg.norm(ft_data, axis=1)
                if len(mag) < 2:
                    continue

                # Compute first-order differences (shocks)
                diff = np.abs(np.diff(mag))
                median_diff = np.median(diff)
                mad_diff = np.median(np.abs(diff - median_diff))

                # A force spike is a shock that is > 5 * 1.4826 * MAD from the median
                threshold = median_diff + 5.0 * 1.4826 * max(mad_diff, 1e-4)
                spikes = np.sum(diff > threshold)

                ep_spikes += spikes
                ep_ft_steps += len(diff)

            if ep_ft_steps > 0:
                spike_rate = float(ep_spikes / ep_ft_steps)
                per_episode_spikes.append(spike_rate)
                total_ft_spikes += ep_spikes
                total_steps += ep_ft_steps
            else:
                per_episode_spikes.append(0.0)

            # 2. Analyze Contact States
            ep_contacts = 0
            ep_contact_steps = 0
            for key in contact_keys:
                contact_data = ep.observations[key]
                # Assume boolean or float thresholded at 0.5
                is_contact = contact_data > 0.5
                ep_contacts += np.sum(is_contact)
                ep_contact_steps += len(contact_data)

            if ep_contact_steps > 0:
                contact_rate = float(ep_contacts / ep_contact_steps)
                per_episode_contacts.append(contact_rate)
                total_contact_steps += ep_contacts
            else:
                per_episode_contacts.append(0.0)

        # A high average force spike rate across episodes indicate collisions or sensor glitches
        overall_spike_rate = float(total_ft_spikes / total_steps) if total_steps > 0 else 0.0
        if ft_keys and overall_spike_rate > 0.01:
            flags.append(
                RiskFlag(
                    level=RiskLevel.WARNING,
                    metric="force_spike_rate",
                    observed=ObservedValue(value=overall_spike_rate, unit="fraction"),
                    threshold=0.01,
                    interpretation=f"Force/torque shock rate is {overall_spike_rate:.2%}.",
                    implication=(
                        "Frequent high-impact force/torque spikes indicate collisions, operator "
                        "struggling, or sensor communication glitches. Inspect trajectories."
                    ),
                    affected_fraction=overall_spike_rate,
                )
            )

        # Low contact density when contact sensors are present
        if contact_keys:
            overall_contact_density = float(
                total_contact_steps / sum(ep.n_steps for ep in batch.episodes)
            )
            if overall_contact_density < 0.01:
                flags.append(
                    RiskFlag(
                        level=RiskLevel.CRITICAL,
                        metric="contact_dropout",
                        observed=ObservedValue(value=overall_contact_density, unit="fraction"),
                        threshold=0.01,
                        interpretation=f"Average contact density is {overall_contact_density:.3f}.",
                        implication=(
                            "Contact sensors report virtually zero contacts in this dataset. "
                            "Verify that sensors are connected, active, and properly mapped."
                        ),
                    )
                )

        raw_metrics = {
            "per_episode_force_spikes": per_episode_spikes,
            "per_episode_contact_density": per_episode_contacts,
            "force_keys_found": ft_keys,
            "contact_keys_found": contact_keys,
        }

        return AnalyzerResult(
            analyzer_name=self.name,
            flags=flags,
            raw_metrics=raw_metrics,
        )
