"""
calibra.outcome_db — Empirical outcome database for closing the prediction loop.

Records observed training outcomes alongside the diagnostic fingerprint Calibra
computed before training. Enables two capabilities:

  1. Empirical blending: when predict() encounters a dataset similar to a past
     outcome, it blends the heuristic score with the empirical observation.
     The closer the fingerprint, the more weight the empirical value gets.

  2. Weight calibration: calibra calibrate re-fits the penalty weights from
     accumulated outcomes using a non-negative least-squares fit.

Storage: JSON Lines at ~/.calibra/outcomes.jsonl (one record per run).

Usage
-----
    from calibra.outcome_db import OutcomeDatabase

    db = OutcomeDatabase()
    db.record(fingerprint, predicted_score=72.0, actual_rate=0.81, policy_family="act")

    similar = db.find_similar(fingerprint, policy_family="act", k=5)
    blended = db.blend_prediction(heuristic_score=72.0, similar=similar)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Keys used to build the fingerprint vector — must match predict.py metric keys
_FINGERPRINT_KEYS = [
    "ldlj",
    "spike_rate",
    "vel_disc_rate",
    "dropout_rate",
    "jitter_cv",
    "action_entropy",
    "contact_phase_fraction",
]

# Per-metric normalization ranges (maps raw value to [0, 1] for distance calcs)
_NORM_RANGES: dict[str, tuple[float, float]] = {
    "ldlj": (-30.0, 0.0),  # more negative = worse
    "spike_rate": (0.0, 0.20),
    "vel_disc_rate": (0.0, 0.40),
    "dropout_rate": (0.0, 0.20),
    "jitter_cv": (0.0, 0.50),
    "action_entropy": (0.0, 6.0),  # higher = better
    "contact_phase_fraction": (0.0, 0.50),  # higher = better
}


def _normalize(fingerprint: dict[str, float]) -> np.ndarray:
    vec = []
    for key in _FINGERPRINT_KEYS:
        val = fingerprint.get(key)
        if val is None:
            vec.append(0.5)  # unknown → mid-range
            continue
        lo, hi = _NORM_RANGES[key]
        span = hi - lo
        vec.append(float(np.clip((val - lo) / span, 0.0, 1.0)) if span > 0 else 0.5)
    return np.array(vec, dtype=float)


class OutcomeRecord:
    __slots__ = (
        "record_id",
        "timestamp",
        "fingerprint",
        "predicted_score",
        "actual_success_rate",
        "policy_family",
        "n_episodes",
        "dataset_name",
        "notes",
    )

    def __init__(
        self,
        record_id: str,
        timestamp: float,
        fingerprint: dict[str, float],
        predicted_score: float,
        actual_success_rate: float,
        policy_family: str,
        n_episodes: int,
        dataset_name: str,
        notes: str,
    ) -> None:
        self.record_id = record_id
        self.timestamp = timestamp
        self.fingerprint = fingerprint
        self.predicted_score = predicted_score
        self.actual_success_rate = actual_success_rate
        self.policy_family = policy_family
        self.n_episodes = n_episodes
        self.dataset_name = dataset_name
        self.notes = notes

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
            "predicted_score": self.predicted_score,
            "actual_success_rate": self.actual_success_rate,
            "policy_family": self.policy_family,
            "n_episodes": self.n_episodes,
            "dataset_name": self.dataset_name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutcomeRecord":
        return cls(
            record_id=d["record_id"],
            timestamp=d["timestamp"],
            fingerprint=d["fingerprint"],
            predicted_score=d["predicted_score"],
            actual_success_rate=d["actual_success_rate"],
            policy_family=d.get("policy_family", "generic"),
            n_episodes=d.get("n_episodes", 0),
            dataset_name=d.get("dataset_name", "unknown"),
            notes=d.get("notes", ""),
        )

    @property
    def normalized(self) -> np.ndarray:
        return _normalize(self.fingerprint)


_DEFAULT_DB_PATH = Path.home() / ".calibra" / "outcomes.jsonl"


class OutcomeDatabase:
    """
    Persistent store of (fingerprint → observed_success_rate) pairs.

    Thread-safety: append-only writes are atomic on POSIX; reads load all
    records at init time and are not updated during the session.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _DEFAULT_DB_PATH
        self._records: list[OutcomeRecord] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(OutcomeRecord.from_dict(json.loads(line)))
                except Exception:
                    pass

    def _append(self, rec: OutcomeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    # ── public API ────────────────────────────────────────────────────────────

    def record(
        self,
        fingerprint: dict[str, float],
        *,
        predicted_score: float,
        actual_success_rate: float,
        policy_family: str = "generic",
        n_episodes: int = 0,
        dataset_name: str = "unknown",
        notes: str = "",
    ) -> OutcomeRecord:
        """
        Add a new observed outcome to the database.

        Parameters
        ----------
        fingerprint         : metric values from predict._extract_metrics()
        predicted_score     : Calibra's predicted score (0–100) before training
        actual_success_rate : observed policy success rate (0.0–1.0)
        """
        rec = OutcomeRecord(
            record_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            fingerprint=fingerprint,
            predicted_score=predicted_score,
            actual_success_rate=actual_success_rate,
            policy_family=policy_family,
            n_episodes=n_episodes,
            dataset_name=dataset_name,
            notes=notes,
        )
        self._records.append(rec)
        self._append(rec)
        self._maybe_sync_to_cloud(rec)
        return rec

    # ── cloud sync ────────────────────────────────────────────────────────────

    def _get_installation_id(self) -> str:
        """Return a stable anonymous installation UUID, generating one on first call.

        Persisted in ~/.calibra/config.json so the same ID is reused across
        invocations. Allows server-side deduplication without identifying the user.
        """
        config_path = Path.home() / ".calibra" / "config.json"
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.exists():
                with open(config_path) as f:
                    data = json.load(f)
                if "installation_id" in data:
                    return data["installation_id"]
            else:
                data = {}
            installation_id = str(uuid.uuid4())
            data["installation_id"] = installation_id
            with open(config_path, "w") as f:
                json.dump(data, f)
            return installation_id
        except Exception:
            # If we can't persist, return a session-only ID rather than crashing.
            return str(uuid.uuid4())

    def _print_telemetry_notice_once(self) -> None:
        """Print a one-time notice to stderr about anonymized telemetry sync."""
        config_path = Path.home() / ".calibra" / "config.json"
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            data: dict = {}
            if config_path.exists():
                with open(config_path) as f:
                    data = json.load(f)
            if data.get("telemetry_notice_shown"):
                return
            import sys

            print(
                "\n[calibra] Anonymized training outcomes are synced to calibra.ai to "
                "improve global\n"
                "          prediction accuracy. No paths, filenames, or identifiable data "
                "are sent.\n"
                "          Set CALIBRA_NO_CLOUD_SYNC=1 to opt out.\n",
                file=sys.stderr,
            )
            data["telemetry_notice_shown"] = True
            with open(config_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _maybe_sync_to_cloud(self, record: OutcomeRecord) -> None:
        """POST an outcome fingerprint to the cloud endpoint.

        Authenticated users (after `calibra login`): syncs to the authenticated
        endpoint — full record, tied to your account, improves your team's model.

        Unauthenticated users: syncs anonymized metric values only to the public
        aggregation endpoint, which improves the global heuristic for everyone.
        Opt out by setting CALIBRA_NO_CLOUD_SYNC=1.

        Failures are always silent (3-second timeout, never blocks the CLI).
        """
        no_sync = os.environ.get("CALIBRA_NO_CLOUD_SYNC", "")
        if no_sync.lower() in ("1", "true"):
            return

        try:
            from calibra import __version__ as _version
        except ImportError:
            _version = "unknown"

        # Prefer authenticated sync when a token is available
        try:
            from calibra.cloud.auth import get_token
            token = get_token()
        except Exception:
            token = None

        if token:
            self._sync_authenticated(record, token, _version)
        else:
            self._sync_anonymous(record, _version)

    def _sync_authenticated(self, record: OutcomeRecord, token: str, version: str) -> None:
        endpoint = os.environ.get(
            "CALIBRA_CLOUD_URL", "https://app.calibra.io"
        ) + "/api/outcomes/authenticated"
        try:
            payload = json.dumps(
                {
                    "fingerprint": record.fingerprint,
                    "predicted_score": record.predicted_score,
                    "actual_success_rate": record.actual_success_rate,
                    "policy_family": record.policy_family,
                    "n_episodes": record.n_episodes,
                    "dataset_name": record.dataset_name,
                    "calibra_version": version,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
            logger.debug("Authenticated cloud sync succeeded")
        except Exception as exc:
            logger.debug("Authenticated cloud sync failed (non-fatal): %s", exc)

    def _sync_anonymous(self, record: OutcomeRecord, version: str) -> None:
        self._print_telemetry_notice_once()
        endpoint = os.environ.get(
            "CALIBRA_CLOUD_ENDPOINT", "https://outcomes.calibra.ai/v1/record"
        )
        try:
            payload = json.dumps(
                {
                    "installation_id": self._get_installation_id(),
                    "fingerprint": record.fingerprint,
                    "predicted_score": record.predicted_score,
                    "actual_success_rate": record.actual_success_rate,
                    "policy_family": record.policy_family,
                    "calibra_version": version,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
            logger.debug("Anonymous cloud sync succeeded to %s", endpoint)
        except Exception as exc:
            logger.debug("Anonymous cloud sync failed (non-fatal): %s", exc)

    def find_similar(
        self,
        fingerprint: dict[str, float],
        policy_family: Optional[str] = None,
        k: int = 5,
        max_distance: float = 0.4,
    ) -> list[tuple[OutcomeRecord, float]]:
        """
        Return up to k records whose fingerprint is within max_distance.

        Returns list of (record, distance) sorted by ascending distance.
        Policy family is used as a soft filter: if family matches, distance
        is multiplied by 0.7 (preferred); mismatches are not excluded.
        """
        if not self._records:
            return []

        query = _normalize(fingerprint)
        scored: list[tuple[OutcomeRecord, float]] = []
        for rec in self._records:
            dist = float(np.linalg.norm(query - rec.normalized))
            if policy_family and rec.policy_family == policy_family:
                dist *= 0.7
            if dist <= max_distance:
                scored.append((rec, dist))

        scored.sort(key=lambda x: x[1])
        return scored[:k]

    def blend_prediction(
        self,
        heuristic_score: float,
        similar: list[tuple[OutcomeRecord, float]],
    ) -> tuple[float, float]:
        """
        Blend heuristic score with empirically observed outcomes.

        The empirical weight scales with the number of close matches and
        inversely with their average distance. With 0 matches the result
        is the pure heuristic. With 3+ close matches (distance < 0.15)
        the empirical component gets up to 60% weight.

        Returns (blended_score, empirical_weight) where empirical_weight
        is in [0, 1] and indicates how much the empirical data contributed.
        """
        if not similar:
            return heuristic_score, 0.0

        # Inverse-distance weights
        weights = np.array([1.0 / (d + 1e-6) for _, d in similar])
        weights /= weights.sum()

        emp_score = sum(
            w * rec.actual_success_rate * 100.0 for (rec, _), w in zip(similar, weights)
        )

        # Empirical weight: grows with match count and match closeness
        min_dist = similar[0][1]
        match_strength = min(1.0, len(similar) / 5.0)
        closeness = max(0.0, 1.0 - min_dist / 0.4)
        empirical_weight = 0.6 * match_strength * closeness

        blended = (1.0 - empirical_weight) * heuristic_score + empirical_weight * emp_score
        return round(float(np.clip(blended, 0.0, 100.0)), 1), round(empirical_weight, 3)

    def calibrate_weights(self) -> Optional[dict[str, float]]:
        """
        Fit new penalty weights from accumulated outcomes using non-negative
        least-squares regression.

        Returns a dict of {metric: suggested_penalty_at_warning} if enough
        data is available (≥ 10 records), else None.
        """
        if len(self._records) < 10:
            return None

        # Build design matrix: rows = records, cols = normalized metric deviations
        X_rows = []
        y = []
        for rec in self._records:
            fp = rec.fingerprint
            # Feature: how far each metric is from its warning threshold (positive = over threshold)
            from calibra.predict import _THRESHOLDS

            row = []
            for key in _FINGERPRINT_KEYS:
                val = fp.get(key)
                if val is None:
                    row.append(0.0)
                    continue
                thresh = _THRESHOLDS.get(key, {})
                warn = thresh.get("warn", 0.0)
                # Deviation from threshold, normalized
                lo, hi = _NORM_RANGES[key]
                span = hi - lo
                dev = (val - warn) / span if span > 0 else 0.0
                row.append(float(np.clip(dev, -1.0, 1.0)))
            X_rows.append(row)
            y.append(1.0 - rec.actual_success_rate)  # target = failure rate

        X = np.array(X_rows)
        y = np.array(y)

        try:
            from numpy.linalg import lstsq

            coeffs, _, _, _ = lstsq(X, y, rcond=None)
            # Map to penalty scale (×100 to match the 0–100 score system)
            result = {}
            for key, coeff in zip(_FINGERPRINT_KEYS, coeffs):
                result[key] = round(float(max(0.0, coeff * 100.0)), 2)
            return result
        except Exception:
            return None

    def summary(self) -> str:
        n = len(self._records)
        if n == 0:
            return "Outcome database: 0 records. Run `calibra predict --record-outcome` after training."
        errors = [abs(r.predicted_score / 100.0 - r.actual_success_rate) for r in self._records]
        mae = float(np.mean(errors)) * 100.0
        lines = [
            f"Outcome database: {n} record(s) at {self.path}",
            f"  Mean absolute error (predicted vs actual): {mae:.1f}%",
        ]
        if n >= 10:
            lines.append("  Enough data for weight calibration — run `calibra calibrate`.")
        else:
            lines.append(f"  Need {10 - n} more record(s) for weight calibration.")
        return "\n".join(lines)

    def list_records(self) -> list[dict]:
        return [r.to_dict() for r in self._records]


# ── global weight download ────────────────────────────────────────────────────


def download_global_weights(db: OutcomeDatabase) -> Optional[dict[str, float]]:
    """
    Download community-calibrated penalty weights from Calibra Cloud and save
    them to ~/.calibra/weights.json so future `calibra predict` runs use them.

    If the local database has ≥10 records, the downloaded weights are blended
    70% community / 30% local before saving.
    """
    import sys

    base = os.environ.get("CALIBRA_CLOUD_URL", "https://app.calibra.io")
    endpoint = f"{base}/v1/weights/latest"

    try:
        req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"error: could not reach Calibra Cloud ({endpoint}): {exc}", file=sys.stderr)
        return None

    global_weights: dict[str, float] = data.get("weights", {})
    if not global_weights:
        print(
            "No community weights available yet — the model improves as more users record outcomes.",
            file=sys.stderr,
        )
        return None

    version = data.get("version", "unknown")
    n_records = "?"
    if version.startswith("global-v"):
        try:
            n_records = int(version[len("global-v"):])
        except ValueError:
            pass

    print(f"Community weights downloaded (version={version}, fitted on {n_records} outcomes).")

    local_weights = db.calibrate_weights()
    if local_weights:
        n_local = len(db._records)
        print(f"Blending: 70% community + 30% local ({n_local} local records).")
        merged: dict[str, float] = {}
        for key in set(global_weights) | set(local_weights):
            g = global_weights.get(key, 0.0)
            loc = local_weights.get(key, 0.0)
            merged[key] = round(0.7 * g + 0.3 * loc, 2)
        source = "community+local"
    else:
        merged = {k: round(v, 2) for k, v in global_weights.items()}
        source = "community"

    weights_path = Path.home() / ".calibra" / "weights.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w") as f:
        json.dump({"weights": merged, "source": source, "version": version}, f, indent=2)

    print(f"\nWeights saved to {weights_path} — active on next `calibra predict` run.\n")
    print("Applied weights (warning-level penalty per metric):")
    for metric, weight in sorted(merged.items()):
        print(f"  {metric:<30} {weight:.2f}")

    return merged
