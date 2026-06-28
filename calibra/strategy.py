"""
calibra.strategy — Regime-adaptive dataset selection strategy.

Maps measurable dataset structure (noise level, dropout, outlier density)
to the optimal CoresetSelector configuration before any training begins.

Scientific basis
----------------
Ablation experiments across ALOHA (clean dual-arm) and DROID-100
(heterogeneous multi-robot) show that the winning selection component
is not fixed — it depends on the dataset's noise-diversity regime:

  Low noise  (ALOHA, spike < 0.02):
      Quality filter has few episodes to remove; diversity selection
      dominates. Combining both yields the best result (+22.6% vs random).

  Moderate noise (DROID, spike 0.02–0.10):
      Quality filter alone collapses coverage of rare behavioral modes.
      Diversity selection is the primary mechanism (+16.9% vs random).
      Full pipeline matches diversity-only because quality contamination
      is low but spatially concentrated in rare morphologies.

  High noise (BridgeData-type, spike > 0.10):
      Corruption is pervasive enough that quality filtering is essential
      before diversity selection. Both components contribute strongly.

Thresholds are calibrated on 3 datasets and marked for revision as more
ablation results accumulate. The `RegimeDiagnosis.evidence` dict exposes
the raw metric values so callers can inspect or override.

Usage
-----
    from calibra.pipeline import Pipeline
    from calibra.strategy import diagnose_regime

    report = Pipeline().run(batch)
    diagnosis = diagnose_regime(report)

    print(diagnosis.regime)           # SelectionRegime.LOW_NOISE
    print(diagnosis.explanation)      # human-readable
    print(diagnosis.recommended_config)  # dict ready for CoresetSelector(**cfg)

    from calibra.pruning import CoresetSelector
    selector = CoresetSelector(**diagnosis.recommended_config, keep_fraction=0.3)
    result = selector.select(batch, report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from calibra.schema.report import DiagnosticReport


class SelectionRegime(str, Enum):
    """
    Dataset noise-diversity regime, determining which selection component dominates.

    LOW_NOISE
        Clean demonstrations, redundancy is the primary inefficiency.
        Diversity selection is the key lever; quality filter adds a secondary
        boost by removing mild motion artefacts.
        Ablation evidence: ALOHA mobile cabinet (spike=0.007, disc=0.013)
        → quality-only +8.8%, diversity-only +13.0%, full pipeline +22.6%.

    MODERATE_NOISE
        Real-world noise at levels that trigger quality filtering, but
        contamination is concentrated in a small number of episodes that
        may represent rare-but-valid behaviours. Quality-alone risks
        collapsing coverage of minority morphologies or tasks.
        Diversity selection dominates; quality filter helps only when
        combined with diversity.
        Ablation evidence: DROID-100 (spike=0.046, disc=0.071)
        → quality-only -5.8% (hurts), diversity-only +16.9%, full +16.9%.

    HIGH_NOISE
        Pervasive corruption (jerk spikes, dropouts, sensor lag). Quality
        filtering is a prerequisite before diversity selection; without it
        the coreset will include corrupted demonstrations that poison the
        downstream model regardless of coverage.
        Expected: BridgeData-style real-world datasets (spike > 0.10).
        Predicted: both quality and diversity contribute strongly.
    """

    LOW_NOISE = "low_noise"
    MODERATE_NOISE = "moderate_noise"
    HIGH_NOISE = "high_noise"


# ── recommended CoresetSelector configs per regime ────────────────────────────
#
# These translate the regime into a concrete parameter set. All configs use
# strategy="diversity" (greedy max-coverage) as Stage 2 — the regimes differ
# in how aggressively Stage 1 (quality filtering) is applied and how much
# weight is given to diversity features vs quality features in the distance
# metric used by Stage 2.

_REGIME_CONFIGS: dict[SelectionRegime, dict] = {
    SelectionRegime.LOW_NOISE: {
        "strategy": "diversity",
        "diversity_weight": 0.85,
        # Lenient quality thresholds — dataset is clean; avoid over-filtering
        # the few episodes that might have higher jerk due to complex motions.
        "max_spike_rate": 0.15,
        "max_vel_disc_rate": 0.30,
        "max_dropout_fraction": 0.10,
        "min_ldlj": -35.0,
    },
    SelectionRegime.MODERATE_NOISE: {
        "strategy": "diversity",
        "diversity_weight": 0.90,
        # Relax quality thresholds further to avoid accidentally removing
        # the only representative of a rare robot morphology or task mode.
        # Diversity selection will naturally deprioritise noisy episodes
        # if they are not in isolated regions of behaviour space.
        "max_spike_rate": 0.25,
        "max_vel_disc_rate": 0.40,
        "max_dropout_fraction": 0.15,
        "min_ldlj": -40.0,
    },
    SelectionRegime.HIGH_NOISE: {
        "strategy": "diversity",
        "diversity_weight": 0.70,
        # Tighter quality thresholds — corruption is pervasive enough that
        # strict filtering is necessary before diversity selection.
        "max_spike_rate": 0.08,
        "max_vel_disc_rate": 0.15,
        "max_dropout_fraction": 0.08,
        "min_ldlj": -25.0,
    },
}

_REGIME_LABELS = {
    SelectionRegime.LOW_NOISE: "LOW NOISE",
    SelectionRegime.MODERATE_NOISE: "MODERATE NOISE",
    SelectionRegime.HIGH_NOISE: "HIGH NOISE",
}


# ── threshold constants ───────────────────────────────────────────────────────
#
# Calibrated against ablation results on ALOHA (LOW_NOISE) and DROID-100
# (MODERATE_NOISE). Marked for revision as BridgeData and other datasets
# are added to the ablation matrix.

_NOISE_SPIKE_LOW = 0.025       # spike_fraction below this → LOW_NOISE
_NOISE_SPIKE_HIGH = 0.090      # spike_fraction above this → HIGH_NOISE
_NOISE_DISC_LOW = 0.040        # vel_disc_rate below this → LOW_NOISE
_NOISE_DISC_HIGH = 0.130       # vel_disc_rate above this → HIGH_NOISE
_DROPOUT_HIGH = 0.05           # dropout above this raises noise score


# ── metric extraction ─────────────────────────────────────────────────────────


def _extract_metrics(report: DiagnosticReport) -> dict:
    """Pull regime-relevant scalars out of a DiagnosticReport."""
    m: dict = {
        "spike_fraction": 0.0,
        "vel_disc_rate": 0.0,
        "dropout_fraction": 0.0,
        "outlier_transition_fraction": 0.0,
        "state_redundancy": None,
        "state_entropy": None,
        "n_episodes": report.n_episodes,
    }

    for ar in report.analyzer_results:
        name = ar.analyzer_name
        raw = ar.raw_metrics

        if name == "control_smoothness":
            js = raw.get("jerk_spikes", {})
            vd = raw.get("vel_discontinuities", {})
            m["spike_fraction"] = float(js.get("mean_spike_fraction", 0.0))
            m["vel_disc_rate"] = float(vd.get("mean_disc_fraction", 0.0))

        elif name == "temporal_stability":
            dr = raw.get("dropout", {})
            m["dropout_fraction"] = float(dr.get("mean_dropout_fraction", 0.0))

        elif name == "latent_dynamics":
            m["outlier_transition_fraction"] = float(
                raw.get("outlier_transition_fraction", 0.0)
            )
            m["state_redundancy"] = raw.get("state_redundancy")
            m["state_entropy"] = raw.get("state_space_entropy_2d")

    return m


def _noise_score(m: dict) -> float:
    """
    Composite noise score in [0, 1].

    Weighted combination of jerk spike rate (primary), velocity discontinuity
    rate (secondary), and frame dropout fraction (tertiary).
    """
    score = (
        0.50 * min(m["spike_fraction"] / _NOISE_SPIKE_HIGH, 1.0)
        + 0.35 * min(m["vel_disc_rate"] / _NOISE_DISC_HIGH, 1.0)
        + 0.15 * min(m["dropout_fraction"] / max(_DROPOUT_HIGH, 1e-9), 1.0)
    )
    return round(float(score), 4)


# ── regime classifier ─────────────────────────────────────────────────────────


@dataclass
class RegimeDiagnosis:
    """
    Output of diagnose_regime().

    Attributes
    ----------
    regime
        The detected SelectionRegime.
    noise_score
        Composite noise score in [0, 1].  0 = perfectly clean, 1 = heavily corrupted.
    recommended_config
        Dict of keyword arguments ready to unpack into CoresetSelector(...).
        Does not include keep_fraction — caller sets that.
    evidence
        Raw metric values that drove the regime decision (for inspection / override).
    explanation
        Human-readable explanation of the diagnosis.
    n_datasets_calibrated
        Number of datasets used to calibrate the thresholds.  Increases as more
        ablation results are added to the matrix.
    """

    regime: SelectionRegime
    noise_score: float
    recommended_config: dict
    evidence: dict
    explanation: str
    n_datasets_calibrated: int = 3   # ALOHA, DROID-100, PushT (update as matrix grows)

    def summary(self) -> str:
        label = _REGIME_LABELS[self.regime]
        lines = [
            "=" * 60,
            f"  CALIBRA REGIME DIAGNOSIS",
            "=" * 60,
            f"  Regime      : {label}",
            f"  Noise score : {self.noise_score:.3f}  (0=clean, 1=corrupted)",
            f"  Calibrated  : {self.n_datasets_calibrated} datasets",
            "-" * 60,
            f"  Evidence:",
            f"    spike_fraction   = {self.evidence['spike_fraction']:.4f}",
            f"    vel_disc_rate    = {self.evidence['vel_disc_rate']:.4f}",
            f"    dropout_fraction = {self.evidence['dropout_fraction']:.4f}",
            f"    outlier_frac     = {self.evidence['outlier_transition_fraction']:.4f}",
            "-" * 60,
            f"  {self.explanation}",
            "-" * 60,
            f"  Recommended CoresetSelector config:",
        ]
        for k, v in self.recommended_config.items():
            lines.append(f"    {k:<28} = {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


def diagnose_regime(
    report: DiagnosticReport,
    custom_thresholds: Optional[dict] = None,
) -> RegimeDiagnosis:
    """
    Classify the dataset noise-diversity regime from a DiagnosticReport.

    Parameters
    ----------
    report
        Output of Pipeline().run(batch). Must include control_smoothness
        and temporal_stability analyzer results for a meaningful diagnosis.
        Missing analyzers cause their metrics to default to 0.

    custom_thresholds
        Optional overrides for the spike/disc thresholds. Keys:
        ``spike_low``, ``spike_high``, ``disc_low``, ``disc_high``.

    Returns
    -------
    RegimeDiagnosis
        Includes the regime enum, noise score, recommended CoresetSelector
        config, raw evidence dict, and a human-readable explanation.
    """
    t = {
        "spike_low": _NOISE_SPIKE_LOW,
        "spike_high": _NOISE_SPIKE_HIGH,
        "disc_low": _NOISE_DISC_LOW,
        "disc_high": _NOISE_DISC_HIGH,
    }
    if custom_thresholds:
        t.update(custom_thresholds)

    metrics = _extract_metrics(report)
    ns = _noise_score(metrics)

    spike = metrics["spike_fraction"]
    disc = metrics["vel_disc_rate"]

    # Primary classification on noise score
    if spike < t["spike_low"] and disc < t["disc_low"]:
        regime = SelectionRegime.LOW_NOISE
        explanation = (
            f"Dataset is predominantly clean (spike={spike:.3f}, disc={disc:.3f}). "
            f"Quality filter has few episodes to remove; both diversity selection and "
            f"quality filtering contribute, with diversity as the primary driver. "
            f"Full pipeline recommended with lenient quality thresholds."
        )

    elif spike >= t["spike_high"] or disc >= t["disc_high"]:
        regime = SelectionRegime.HIGH_NOISE
        explanation = (
            f"Pervasive corruption detected (spike={spike:.3f}, disc={disc:.3f}). "
            f"Quality filtering is a prerequisite — corrupted demonstrations will "
            f"degrade the model regardless of coverage. Apply tight quality thresholds "
            f"before diversity selection."
        )

    else:
        regime = SelectionRegime.MODERATE_NOISE
        explanation = (
            f"Moderate real-world noise (spike={spike:.3f}, disc={disc:.3f}). "
            f"Quality-alone risks collapsing coverage of rare behavioral modes "
            f"(validated: quality-only hurt on DROID-100 at spike=0.046). "
            f"Use relaxed quality thresholds combined with strong diversity selection "
            f"so that rare-but-valid episodes are not filtered out."
        )

    return RegimeDiagnosis(
        regime=regime,
        noise_score=ns,
        recommended_config=dict(_REGIME_CONFIGS[regime]),
        evidence=metrics,
        explanation=explanation,
    )


# ── convenience: diagnose + select in one call ────────────────────────────────


def select_with_regime(
    batch: "EpisodeBatch",
    report: DiagnosticReport,
    keep_fraction: float = 0.30,
    verbose: bool = True,
) -> tuple["PruningResult", RegimeDiagnosis]:
    """
    Diagnose the regime and run CoresetSelector with the recommended config.

    Parameters
    ----------
    batch, report
        EpisodeBatch and corresponding DiagnosticReport.
    keep_fraction
        Target fraction of episodes to retain.
    verbose
        Print the regime diagnosis summary before selecting.

    Returns
    -------
    (PruningResult, RegimeDiagnosis)
    """
    from calibra.pruning import CoresetSelector

    diagnosis = diagnose_regime(report)
    if verbose:
        print(diagnosis.summary())

    selector = CoresetSelector(
        keep_fraction=keep_fraction,
        **diagnosis.recommended_config,
    )
    result = selector.select(batch, report)
    return result, diagnosis
