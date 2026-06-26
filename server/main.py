"""
Calibra Cloud Outcomes Server
------------------------------
Receives anonymized training outcome fingerprints from `calibra predict --record-outcome`,
aggregates them into a global calibration dataset, and serves:

  POST /v1/record           — store an anonymized outcome
  POST /v1/badge            — register a dataset certification result
  GET  /v1/stats            — public aggregate stats (total records, MAE, improving curve)
  GET  /v1/weights/latest   — latest calibrated penalty weights (downloaded by `calibra calibrate`)
  GET  /badge/{dataset_id}  — shields.io redirect for HuggingFace dataset quality badge

Storage: SQLite (single file, zero external deps). Swap for Postgres by changing DB_URL.

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8080

Deploy:
    docker build -t calibra-server .
    docker run -p 8080:8080 -v /data:/data calibra-server
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

DB_PATH = Path(os.environ.get("CALIBRA_DB_PATH", "/data/outcomes.db"))

# ── schema ─────────────────────────────────────────────────────────────────────

CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    id               TEXT PRIMARY KEY,
    ts               REAL NOT NULL,
    installation_id  TEXT NOT NULL,
    policy_family    TEXT NOT NULL,
    calibra_version  TEXT NOT NULL,
    predicted_score  REAL NOT NULL,
    actual_rate      REAL NOT NULL,
    -- metric fingerprint columns
    ldlj             REAL,
    spike_rate       REAL,
    vel_disc_rate    REAL,
    dropout_rate     REAL,
    jitter_cv        REAL,
    action_entropy   REAL,
    contact_phase_fraction REAL,
    jepa_surprise    REAL
)
"""

CREATE_BADGES = """
CREATE TABLE IF NOT EXISTS badges (
    dataset_id   TEXT PRIMARY KEY,
    status       TEXT NOT NULL,      -- CERTIFIED | PROVISIONALLY_CERTIFIED | NOT_CERTIFIED
    updated_ts   REAL NOT NULL,
    calibra_version TEXT NOT NULL
)
"""

CREATE_WEIGHTS = """
CREATE TABLE IF NOT EXISTS calibrated_weights (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    version   TEXT NOT NULL,
    weights   TEXT NOT NULL          -- JSON blob
)
"""

_FINGERPRINT_KEYS = [
    "ldlj", "spike_rate", "vel_disc_rate", "dropout_rate",
    "jitter_cv", "action_entropy", "contact_phase_fraction", "jepa_surprise",
]

# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(CREATE_OUTCOMES)
        conn.execute(CREATE_BADGES)
        conn.execute(CREATE_WEIGHTS)
        conn.commit()


# ── request / response models ──────────────────────────────────────────────────

class OutcomeRecord(BaseModel):
    installation_id: str
    fingerprint: dict[str, Optional[float]]
    predicted_score: float = Field(ge=0, le=100)
    actual_success_rate: float = Field(ge=0.0, le=1.0)
    policy_family: str = "generic"
    calibra_version: str = "unknown"


class BadgeRegistration(BaseModel):
    dataset_id: str
    status: str                         # CERTIFIED | PROVISIONALLY_CERTIFIED | NOT_CERTIFIED
    calibra_version: str = "unknown"


class WeightSubmission(BaseModel):
    weights: dict[str, float]
    calibra_version: str = "unknown"


# ── app ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield


app = FastAPI(
    title="Calibra Outcomes Server",
    description="Global aggregation of anonymized Calibra training outcome fingerprints",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── POST /v1/record ────────────────────────────────────────────────────────────

@app.post("/v1/record", status_code=201)
async def record_outcome(body: OutcomeRecord):
    """Store an anonymized outcome fingerprint from a Calibra installation."""
    record_id = str(uuid.uuid4())[:8]
    fp = body.fingerprint

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO outcomes
              (id, ts, installation_id, policy_family, calibra_version,
               predicted_score, actual_rate,
               ldlj, spike_rate, vel_disc_rate, dropout_rate,
               jitter_cv, action_entropy, contact_phase_fraction, jepa_surprise)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record_id,
                time.time(),
                body.installation_id,
                body.policy_family,
                body.calibra_version,
                body.predicted_score,
                body.actual_success_rate,
                fp.get("ldlj"),
                fp.get("spike_rate"),
                fp.get("vel_disc_rate"),
                fp.get("dropout_rate"),
                fp.get("jitter_cv"),
                fp.get("action_entropy"),
                fp.get("contact_phase_fraction"),
                fp.get("jepa_surprise"),
            ),
        )
        conn.commit()

    _maybe_recalibrate()
    return {"ok": True, "id": record_id}


# ── POST /v1/badge ─────────────────────────────────────────────────────────────

@app.post("/v1/badge", status_code=201)
async def register_badge(body: BadgeRegistration):
    """Register a dataset certification result so the badge endpoint can serve it."""
    valid = {"CERTIFIED", "PROVISIONALLY_CERTIFIED", "NOT_CERTIFIED"}
    if body.status not in valid:
        raise HTTPException(400, f"status must be one of {valid}")

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO badges (dataset_id, status, updated_ts, calibra_version)
            VALUES (?,?,?,?)
            ON CONFLICT(dataset_id) DO UPDATE SET
              status=excluded.status,
              updated_ts=excluded.updated_ts,
              calibra_version=excluded.calibra_version
            """,
            (body.dataset_id, body.status, time.time(), body.calibra_version),
        )
        conn.commit()

    return {"ok": True}


# ── GET /v1/stats ──────────────────────────────────────────────────────────────

@app.get("/v1/stats")
async def get_stats():
    """Public aggregate statistics — used by the calibra.ai counter widget."""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                         AS total,
                COUNT(DISTINCT installation_id)                  AS installations,
                AVG(ABS(predicted_score / 100.0 - actual_rate)) AS mae,
                MIN(ts)                                          AS first_ts,
                MAX(ts)                                          AS last_ts
            FROM outcomes
            """
        ).fetchone()

        by_policy = conn.execute(
            "SELECT policy_family, COUNT(*) AS n FROM outcomes GROUP BY policy_family"
        ).fetchall()

        certified_count = conn.execute(
            "SELECT COUNT(*) FROM badges WHERE status='CERTIFIED'"
        ).fetchone()[0]

    return {
        "total_outcomes": row["total"],
        "installations": row["installations"],
        "mean_absolute_error": round(row["mae"] * 100, 2) if row["mae"] else None,
        "first_record": row["first_ts"],
        "last_record": row["last_ts"],
        "by_policy": {r["policy_family"]: r["n"] for r in by_policy},
        "certified_datasets": certified_count,
    }


# ── GET /v1/weights/latest ─────────────────────────────────────────────────────

@app.get("/v1/weights/latest")
async def get_weights():
    """
    Return the latest globally calibrated penalty weights.

    `calibra calibrate --global` downloads these and blends them with
    the lab's local weights proportionally to record counts.
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT weights, version, ts FROM calibrated_weights ORDER BY ts DESC LIMIT 1"
        ).fetchone()

    if not row:
        raise HTTPException(404, "No calibrated weights yet — need more outcome records.")

    return {
        "version": row["version"],
        "ts": row["ts"],
        "weights": json.loads(row["weights"]),
    }


# ── GET /badge/{dataset_id} ────────────────────────────────────────────────────

_BADGE_COLORS = {
    "CERTIFIED": "brightgreen",
    "PROVISIONALLY_CERTIFIED": "yellow",
    "NOT_CERTIFIED": "red",
}

_BADGE_LABELS = {
    "CERTIFIED": "Calibra%20Certified",
    "PROVISIONALLY_CERTIFIED": "Calibra%20Provisional",
    "NOT_CERTIFIED": "Calibra%20Failed",
}


@app.get("/badge/{dataset_id:path}")
async def dataset_badge(dataset_id: str):
    """
    Redirect to a shields.io SVG badge for the given HuggingFace dataset ID.

    Embed in a HuggingFace dataset README:
        ![Calibra](https://outcomes.calibra.ai/badge/lerobot/my_dataset)
    """
    # Strip trailing .svg if present
    clean_id = dataset_id.removesuffix(".svg")

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM badges WHERE dataset_id=?", (clean_id,)
        ).fetchone()

    if not row:
        url = "https://img.shields.io/badge/Calibra-unverified-lightgrey?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+"
        return RedirectResponse(url, status_code=302)

    status = row["status"]
    color = _BADGE_COLORS.get(status, "lightgrey")
    label = _BADGE_LABELS.get(status, "Calibra")
    url = f"https://img.shields.io/badge/{label}-{color}?style=flat-square"
    return RedirectResponse(url, status_code=302)


# ── weight recalibration ───────────────────────────────────────────────────────

def _maybe_recalibrate() -> None:
    """Re-fit global weights when we cross a new 50-record milestone."""
    with _get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]

    if n < 20 or n % 50 != 0:
        return

    _recalibrate(n)


def _recalibrate(n: int) -> None:
    """NNLS fit of penalty weights from all accumulated outcomes."""
    try:
        import numpy as np
        from numpy.linalg import lstsq
    except ImportError:
        return

    _NORM_RANGES: dict[str, tuple[float, float]] = {
        "ldlj": (-30.0, 0.0),
        "spike_rate": (0.0, 0.20),
        "vel_disc_rate": (0.0, 0.40),
        "dropout_rate": (0.0, 0.20),
        "jitter_cv": (0.0, 0.50),
        "action_entropy": (0.0, 6.0),
        "contact_phase_fraction": (0.0, 0.50),
        "jepa_surprise": (0.0, 1.0),
    }

    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(_FINGERPRINT_KEYS)}, actual_rate FROM outcomes"
        ).fetchall()

    if len(rows) < 20:
        return

    X, y = [], []
    for row in rows:
        feat = []
        for key in _FINGERPRINT_KEYS:
            val = row[key]
            if val is None:
                feat.append(0.0)
                continue
            lo, hi = _NORM_RANGES[key]
            span = hi - lo
            feat.append(float(np.clip((val - lo) / span, 0.0, 1.0)) if span > 0 else 0.5)
        X.append(feat)
        y.append(1.0 - row["actual_rate"])

    X_arr = np.array(X)
    y_arr = np.array(y)
    coeffs, _, _, _ = lstsq(X_arr, y_arr, rcond=None)

    weights = {
        key: round(float(max(0.0, c * 100.0)), 3)
        for key, c in zip(_FINGERPRINT_KEYS, coeffs)
    }
    version = f"global-v{n}"

    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO calibrated_weights (ts, version, weights) VALUES (?,?,?)",
            (time.time(), version, json.dumps(weights)),
        )
        conn.commit()
