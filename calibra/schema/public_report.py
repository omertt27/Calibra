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
    platform: Optional[str] = None          # "pusht", "aloha", "so100"
    embodiment: Optional[str] = None        # "planar-manipulator", "bimanual-arm"
    action_dimensions: Optional[int] = None
    dof: Optional[int] = None


class DatasetInfo(BaseModel):
    provider: str                           # "huggingface", "local"
    repository_id: str                      # "lerobot/pusht" or absolute path
    revision: Optional[str] = None          # HF dataset revision / git SHA
    dataset_format: str                     # "lerobot-v2", "hdf5", "rlds", "mcap"
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
    python: str     # "3.12.4"
    platform: str   # "linux-x86_64"


class AuditConfig(BaseModel):
    profile: Optional[str] = None          # named profile, e.g. "pusht"
    configuration_hash: str                # sha256[:16] of profile+rubric+sampling
    scoring_rubric: str                    # "robot-dataset-quality-v1.0"
    sampling: SamplingConfig
    environment: EnvironmentInfo


# ── report identity ───────────────────────────────────────────────────────────

class ReportMeta(BaseModel):
    id: str                                 # "sha256:<hex>" — hash of canonical body
    generated_at: datetime
    calibra_version: str
    status: Literal["complete", "partial", "failed"]


# ── metric values ─────────────────────────────────────────────────────────────

class MetricValue(BaseModel):
    value: Optional[float] = None   # raw physical value (e.g. 0.003 fraction)
    unit: str = ""                  # physical unit ("fraction", "ms", "bits/dim")
    score: Optional[float] = None   # normalized 0-100; null if not applicable
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    ci_method: str = "bootstrap"
    methodology: str = ""           # "temporal.dropout_rate.v1"


class DimensionResult(BaseModel):
    score: float            # 0-100, weighted aggregate of metric scores
    weight: float           # this dimension's fraction of the overall score
    metrics: dict[str, MetricValue] = {}


# ── overall result ────────────────────────────────────────────────────────────

class OverallResult(BaseModel):
    score: float                                        # 0-100
    grade: str                                          # "A"–"F"
    confidence: float                                   # 0-1, CI-derived
    certification: Literal["pass", "provisional", "fail"]
    critical_failures: list[str] = []                  # metric names of CRITICAL flags


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
    code: str                               # "TEMPORAL_JITTER_HIGH"
    metric: str                             # raw metric name from analyzer
    message: str                            # human-readable interpretation
    implication: str = ""                   # downstream training risk
    affected_fraction: Optional[float] = None
    observed_value: Optional[float] = None
    observed_unit: str = ""
    threshold: Optional[float] = None


# ── results container ─────────────────────────────────────────────────────────

class AuditResults(BaseModel):
    overall: OverallResult
    dimensions: dict[str, DimensionResult]
    findings: list[Finding]
    recommendations: Recommendations


# ── top-level contract ────────────────────────────────────────────────────────

class CalibraReport(BaseModel):
    schema_version: str = "1.0.0"
    report: ReportMeta
    dataset: DatasetInfo
    audit: AuditConfig
    results: AuditResults

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
