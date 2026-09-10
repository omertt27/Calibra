"""
Public JSON report schema — the long-term stable contract for CalibraReport.

Consumed by: leaderboard site, dataset page generators, badge/verification
systems, historical tracking diffs, and external APIs.

Rule: DiagnosticReport (internal) can change freely.
      CalibraReport (this file) requires a schema_version bump.

Three-layer metric structure:
  raw value  (physical units, e.g. dropout_rate = 0.003 fraction)
  → normalized score  (0-100, computed by scoring rubric)
  → weighted dimension/overall score

Separating raw from score lets the rubric be revised without re-running
expensive audits — scores can be recomputed from preserved raw values.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel

# ── dataset ───────────────────────────────────────────────────────────────────


class RobotInfo(BaseModel):
    platform: Optional[str] = None  # "pusht", "aloha", "so100"
    embodiment: Optional[str] = None  # "planar-manipulator", "bimanual-arm"
    action_dimensions: Optional[int] = None
    dof: Optional[int] = None


class DatasetInfo(BaseModel):
    provider: str  # "huggingface", "local"
    repository_id: str  # "lerobot/pusht" or absolute path
    revision: Optional[str] = None  # HF dataset revision / git SHA
    dataset_format: str  # "lerobot-v2", "hdf5", "rlds", "mcap"
    license: Optional[str] = None
    homepage: Optional[str] = None
    episodes_total: int
    episodes_audited: int
    frames_total: int
    robot: Optional[RobotInfo] = None


# ── audit configuration ───────────────────────────────────────────────────────


class SamplingConfig(BaseModel):
    mode: Literal["full", "random", "stratified"] = "full"
    seed: Optional[int] = None
    fraction: float = 1.0


class EnvironmentInfo(BaseModel):
    python: str  # "3.12.4"
    platform: str  # "linux-x86_64"


class AuditConfig(BaseModel):
    profile: Optional[str] = None  # named profile, e.g. "pusht"
    configuration_hash: str  # sha256[:16] of profile+rubric+sampling
    scoring_rubric: str  # "robot-dataset-quality-v1.0"
    sampling: SamplingConfig
    environment: EnvironmentInfo


# ── report identity ───────────────────────────────────────────────────────────


class ReportMeta(BaseModel):
    id: str  # "sha256:<hex>" — hash of canonical body
    generated_at: datetime
    calibra_version: str
    status: Literal["complete", "partial", "failed"]


# ── metric values ─────────────────────────────────────────────────────────────


class MetricValue(BaseModel):
    value: Optional[float] = None  # raw physical value (e.g. 0.003 fraction)
    unit: str = ""  # physical unit ("fraction", "ms", "bits/dim")
    score: Optional[float] = None  # normalized 0-100; null if not applicable
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    ci_method: str = "bootstrap"
    methodology: str = ""  # "temporal.dropout_rate.v1"


class DimensionResult(BaseModel):
    score: float  # 0-100, weighted aggregate of metric scores
    weight: float  # this dimension's fraction of the overall score
    metrics: dict[str, MetricValue] = {}


# ── overall result ────────────────────────────────────────────────────────────


class OverallResult(BaseModel):
    score: float  # 0-100
    grade: str  # "A"–"F"
    confidence: float  # 0-1, CI-derived
    certification: Literal["pass", "provisional", "fail"]
    critical_failures: list[str] = []  # metric names of CRITICAL flags


# ── policy recommendations ────────────────────────────────────────────────────


class PolicyRecommendation(BaseModel):
    status: Literal["recommended", "review", "not_recommended"]
    reason: Optional[str] = None


class Recommendations(BaseModel):
    behavior_cloning: PolicyRecommendation = PolicyRecommendation(status="review")
    act: PolicyRecommendation = PolicyRecommendation(status="review")
    diffusion_policy: PolicyRecommendation = PolicyRecommendation(status="review")
    gr00t: PolicyRecommendation = PolicyRecommendation(status="review")


# ── findings ──────────────────────────────────────────────────────────────────


class Finding(BaseModel):
    severity: Literal["critical", "warning", "info", "ok"]
    code: str  # "TEMPORAL_JITTER_HIGH"
    metric: str  # raw metric name from analyzer
    message: str  # human-readable interpretation
    implication: str = ""  # downstream training risk
    affected_fraction: Optional[float] = None
    observed_value: Optional[float] = None
    observed_unit: str = ""
    threshold: Optional[float] = None
    benign_baseline_rate: Optional[float] = None  # fraction flagged on known-clean datasets; null = no baseline
    baseline_source: Optional[str] = None  # e.g. "lerobot/pusht (n=206, v0.10.0)"


# ── results container ─────────────────────────────────────────────────────────


class AuditResults(BaseModel):
    overall: OverallResult
    dimensions: dict[str, DimensionResult]
    findings: list[Finding]
    recommendations: Recommendations


# ── episode verdicts ──────────────────────────────────────────────────────────


class EpisodeVerdicts(BaseModel):
    """
    Per-episode selection output from a calibra prune run.

    This is the machine-readable contract for downstream systems:
    training pipelines, CI/CD checks, dataset version managers.

    reason_codes maps episode_id → list of failure codes, e.g.:
      {"42": ["jerk_spike", "timestamp_dropout"], "7": ["diversity_pruned"]}

    Standard reason codes
    ---------------------
    Stage 1 (quality filter):
      short_episode         — fewer than min_length steps
      jerk_spike            — spike_rate exceeded threshold
      velocity_discontinuity — vel_disc_rate exceeded threshold
      timestamp_dropout     — frame dropout fraction exceeded threshold
      low_smoothness        — LDLJ below minimum threshold

    Stage 2 (diversity / novelty selection):
      diversity_pruned      — redundant under greedy max-coverage
      novelty_pruned        — low transition novelty score
      influence_pruned      — low influence score
      energy_pruned         — low dynamics energy / surprisal
      world_model_pruned    — low world-model surprise score
    """

    keep_episode_ids: list[str] = []
    reject_episode_ids: list[str] = []
    reason_codes: dict[str, list[str]] = {}  # episode_id → [reason, ...]
    quality_scores: dict[str, float] = {}  # episode_id → composite quality score
    n_original: int = 0
    n_kept: int = 0
    keep_fraction_actual: float = 0.0
    method: str = ""


# ── anomaly summary ───────────────────────────────────────────────────────────


class DetectorCalibrationSummary(BaseModel):
    """
    Per-detector calibration context for an anomaly audit.

    Separates what the detector observed from what clean datasets typically show.
    A detected anomaly is not the same as confirmed corruption — this context
    lets users judge whether the observed rate is genuinely elevated.
    """

    detector: str
    n_flagged: int  # episodes flagged by this detector in the current audit
    fraction_flagged: float  # n_flagged / total episodes
    benign_firing_rate: Optional[float] = None  # from calibration registry; null = no baseline
    corrupted_episode_detection_rate: Optional[float] = None  # from benchmark; null = not measured
    # Episode-level: an episode is counted detected if the detector fired anywhere in it.
    # Does not distinguish whether the detector fired at the corrupted region or elsewhere.
    baseline_dataset: Optional[str] = None  # which dataset the baseline came from
    sample_count: Optional[int] = None  # n_episodes in the baseline measurement
    concentration: str = "unknown"  # "spread" | "clustered" | "endpoint" | "unknown"


class AnomalySummary(BaseModel):
    """
    Episode-level anomaly detection summary.

    Designed to answer: "When Calibra flags my data, how surprised should I be?"

    Architecture note: detected anomaly ≠ confirmed corruption ≠ DROP decision.
    The detector identifies statistical outliers within the dataset's own
    distribution. The decision layer (EpisodeCharacterization / CurationReport)
    determines what to do with them.
    """

    total_flags: int  # total (episode, detector) flag pairs
    affected_episodes: int  # episodes with at least one flag
    affected_episode_rate: float  # affected_episodes / n_total_episodes
    n_total_episodes: int
    detectors: list[DetectorCalibrationSummary] = []
    # Concentration: fraction of flags in the top-5 most-flagged episodes.
    # High concentration → investigate those episodes for a shared cause.
    top5_episode_flag_fraction: Optional[float] = None
    position_note: str = ""  # e.g. "concentrated near dataset end"


# ── top-level contract ────────────────────────────────────────────────────────


class EpisodeHash(BaseModel):
    episode_id: str
    hash: str


class CalibraReport(BaseModel):
    schema_version: str = "1.2.0"
    report: ReportMeta
    dataset: DatasetInfo
    audit: AuditConfig
    results: AuditResults
    episode_verdicts: Optional[EpisodeVerdicts] = None
    # Episode-level anomaly detection summary with calibration context.
    # Null when produced without anomaly detection (e.g. raw `calibra certify`
    # without the decision layer). Added in schema 1.2.0.
    anomaly_summary: Optional[AnomalySummary] = None
    # Incremental analysis: per-episode SHA-256[:16] of timestamps+actions.
    # A list of records (not a dict keyed by episode_id) so the field has a
    # stable Arrow/Parquet schema regardless of which episodes are hashed —
    # a dict shape made every observed episode_id its own struct field, and
    # collapsed to a zero-field struct when no report had any hashes at all,
    # which PyArrow refuses to write to Parquet.
    # Empty when the report was produced without --cache-dir.
    episode_hashes: list[EpisodeHash] = []

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def write(self, path: str) -> None:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load(path: str) -> "CalibraReport":
        from pathlib import Path

        return CalibraReport.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def compute_id(body: dict[str, Any]) -> str:
        """SHA-256 of the canonical report body (report.id field excluded)."""
        body_no_id = {k: v for k, v in body.items() if k != "id"}
        canonical = json.dumps(body_no_id, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"sha256:{digest}"
