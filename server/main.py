"""
Calibra Cloud Outcomes Server
------------------------------
Receives anonymized training outcome fingerprints from `calibra predict --record-outcome`,
aggregates them into a global calibration dataset, and serves:

  POST /v1/record           — store an anonymized outcome
  POST /v1/badge            — register a dataset certification result
  GET  /v1/stats            — public aggregate stats (total records, MAE, improving curve)
  GET  /v1/weights/latest   — latest calibrated penalty weights (downloaded by `calibra calibrate`)
  GET  /v1/percentiles      — community metric percentiles (used by `calibra compare --community`)
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

import hashlib
import secrets

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

DB_PATH = Path(os.environ.get("CALIBRA_DB_PATH", "/data/outcomes.db"))

PLAN_LIMITS: dict[str, Optional[int]] = {"free": 5, "pro": None}  # None = unlimited

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
    domain           TEXT NOT NULL DEFAULT 'robotics',
    fingerprint_json TEXT,
    -- robotics metric fingerprint columns (domain='robotics' only; other domains
    -- store their fingerprint in fingerprint_json instead, since their metric
    -- sets differ from robotics' fixed columns)
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

# Columns added after the initial release — kept out of CREATE_OUTCOMES' happy path
# so an already-deployed sqlite file (created before these existed) can be migrated
# in place via ALTER TABLE rather than requiring a destructive rebuild.
_OUTCOMES_MIGRATION_COLUMNS: list[tuple[str, str]] = [
    ("domain", "TEXT NOT NULL DEFAULT 'robotics'"),
    ("fingerprint_json", "TEXT"),
]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)").fetchall()}
    for col_name, col_def in _OUTCOMES_MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE outcomes ADD COLUMN {col_name} {col_def}")

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

CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    token         TEXT PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    plan          TEXT NOT NULL DEFAULT 'free',
    project_count INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
)
"""

CREATE_REPORTS = """
CREATE TABLE IF NOT EXISTS reports (
    id           TEXT PRIMARY KEY,
    user_token   TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    report_json  TEXT NOT NULL,
    created_at   REAL NOT NULL,
    FOREIGN KEY (user_token) REFERENCES users(token)
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
        conn.execute(CREATE_USERS)
        conn.execute(CREATE_REPORTS)
        _migrate_schema(conn)
        conn.commit()


# ── auth helpers ───────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _get_user_by_token(token: str) -> Optional[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE token=?", (token,)).fetchone()


def _auth_required(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> sqlite3.Row:
    if not creds:
        raise HTTPException(401, "Authorization header required")
    user = _get_user_by_token(creds.credentials)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user


# ── request / response models ──────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    password: str


class ReportUpload(BaseModel):
    dataset_name: str
    report: dict


class OutcomeRecord(BaseModel):
    installation_id: str
    fingerprint: dict[str, Optional[float]]
    predicted_score: float = Field(ge=0, le=100)
    actual_success_rate: float = Field(ge=0.0, le=1.0)
    policy_family: str = "generic"
    calibra_version: str = "unknown"
    domain: str = "robotics"


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


# ── POST /api/users/signup ─────────────────────────────────────────────────────

@app.post("/api/users/signup", status_code=201)
async def signup(body: SignupRequest):
    """Create a new account and return an API token."""
    token = secrets.token_hex(32)
    pw_hash = _hash_password(body.password)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (token, email, password_hash, created_at) VALUES (?,?,?,?)",
                (token, body.email.lower().strip(), pw_hash, time.time()),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Email already registered")
    return {"token": token, "email": body.email}


# ── GET /api/me ────────────────────────────────────────────────────────────────

@app.get("/api/me")
async def get_me(user: sqlite3.Row = Depends(_auth_required)):
    """Return the authenticated user's profile."""
    return {
        "email": user["email"],
        "plan": user["plan"],
        "project_count": user["project_count"],
    }


# ── POST /api/reports ──────────────────────────────────────────────────────────

@app.post("/api/reports", status_code=201)
async def push_report(
    body: ReportUpload,
    user: sqlite3.Row = Depends(_auth_required),
):
    """Store a DiagnosticReport and return a shareable URL."""
    plan = user["plan"]
    limit = PLAN_LIMITS.get(plan)
    if limit is not None and user["project_count"] >= limit:
        raise HTTPException(402, f"Free plan limit ({limit} reports) reached. Upgrade to Pro.")

    report_id = str(uuid.uuid4())[:12]
    base_url = os.environ.get("CALIBRA_BASE_URL", "https://app.calibra.io")
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO reports (id, user_token, dataset_name, report_json, created_at) VALUES (?,?,?,?,?)",
            (report_id, user["token"], body.dataset_name, json.dumps(body.report), time.time()),
        )
        conn.execute(
            "UPDATE users SET project_count = project_count + 1 WHERE token=?",
            (user["token"],),
        )
        conn.commit()
    return {"id": report_id, "url": f"{base_url}/reports/{report_id}"}


# ── GET /api/reports ───────────────────────────────────────────────────────────

@app.get("/api/reports")
async def list_reports(user: sqlite3.Row = Depends(_auth_required)):
    """List the authenticated user's stored diagnostic reports."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, dataset_name, created_at FROM reports WHERE user_token=? ORDER BY created_at DESC LIMIT 100",
            (user["token"],),
        ).fetchall()
    return [
        {"id": r["id"], "dataset_name": r["dataset_name"], "created_at": r["created_at"]}
        for r in rows
    ]


# ── GET /api/reports/{id} ──────────────────────────────────────────────────────

@app.get("/api/reports/{report_id}")
async def get_report(report_id: str, user: sqlite3.Row = Depends(_auth_required)):
    """Fetch a single stored report by ID."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, dataset_name, report_json, created_at FROM reports WHERE id=? AND user_token=?",
            (report_id, user["token"]),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Report not found")
    return {
        "id": row["id"],
        "dataset_name": row["dataset_name"],
        "created_at": row["created_at"],
        "report": json.loads(row["report_json"]),
    }


# ── POST /api/outcomes/authenticated ──────────────────────────────────────────

@app.post("/api/outcomes/authenticated", status_code=201)
async def record_outcome_authenticated(
    body: OutcomeRecord,
    user: sqlite3.Row = Depends(_auth_required),
):
    """Authenticated variant of /v1/record — links outcome to a user account."""
    record_id = str(uuid.uuid4())[:8]
    fp = body.fingerprint
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO outcomes
              (id, ts, installation_id, policy_family, calibra_version,
               predicted_score, actual_rate, domain, fingerprint_json,
               ldlj, spike_rate, vel_disc_rate, dropout_rate,
               jitter_cv, action_entropy, contact_phase_fraction, jepa_surprise)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record_id, time.time(), user["email"],
                body.policy_family, body.calibra_version,
                body.predicted_score, body.actual_success_rate,
                body.domain, json.dumps(fp),
                fp.get("ldlj"), fp.get("spike_rate"), fp.get("vel_disc_rate"),
                fp.get("dropout_rate"), fp.get("jitter_cv"), fp.get("action_entropy"),
                fp.get("contact_phase_fraction"), fp.get("jepa_surprise"),
            ),
        )
        conn.commit()
    _maybe_recalibrate()
    return {"ok": True, "id": record_id}


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
               predicted_score, actual_rate, domain, fingerprint_json,
               ldlj, spike_rate, vel_disc_rate, dropout_rate,
               jitter_cv, action_entropy, contact_phase_fraction, jepa_surprise)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record_id,
                time.time(),
                body.installation_id,
                body.policy_family,
                body.calibra_version,
                body.predicted_score,
                body.actual_success_rate,
                body.domain,
                json.dumps(fp),
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


# ── GET /v1/percentiles ────────────────────────────────────────────────────────

@app.get("/v1/percentiles")
async def get_community_percentiles(policy_family: Optional[str] = None):
    """
    Return p25/p50/p75/p90 for each metric across all recorded outcomes.

    Used by `calibra compare --community` to show how a dataset ranks
    against the broader Calibra user base. Public — no auth required.

    Robotics-only: scoped to domain='robotics' since these columns aren't
    populated for other domains (e.g. llm_sft, which stores its fingerprint in
    fingerprint_json instead — see /api/community/similar for that path).
    """
    where = " WHERE domain = 'robotics' AND policy_family = ?" if policy_family else " WHERE domain = 'robotics'"
    params = (policy_family,) if policy_family else ()
    col_list = ", ".join(_FINGERPRINT_KEYS)

    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT {col_list} FROM outcomes{where}", params
        ).fetchall()

    n = len(rows)
    if n < 5:
        return {"n": n, "policy_family": policy_family or "all", "percentiles": {}}

    result: dict[str, dict] = {}
    for i, col in enumerate(_FINGERPRINT_KEYS):
        vals = sorted(r[i] for r in rows if r[i] is not None)
        if not vals:
            continue
        k = len(vals)
        result[col] = {
            "p25": vals[int(k * 0.25)],
            "p50": vals[int(k * 0.50)],
            "p75": vals[int(k * 0.75)],
            "p90": vals[min(int(k * 0.90), k - 1)],
            "n": k,
        }

    return {"n": n, "policy_family": policy_family or "all", "percentiles": result}


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


# ── POST /api/community/similar ───────────────────────────────────────────────

class SimilarRequest(BaseModel):
    fingerprint: dict[str, Optional[float]]
    policy_family: str = "generic"
    k: int = Field(default=5, ge=1, le=20)
    domain: str = "robotics"


# Per-domain normalization ranges, mirroring calibra.outcome_db._DOMAIN_SCHEMAS on the
# client. Kept as a local duplicate (not imported from the `calibra` package) to match
# this server's existing style of not depending on the client library.
_DOMAIN_NORM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "robotics": {
        "ldlj": (-30.0, 0.0),
        "spike_rate": (0.0, 0.20),
        "vel_disc_rate": (0.0, 0.40),
        "dropout_rate": (0.0, 0.20),
        "jitter_cv": (0.0, 0.50),
        "action_entropy": (0.0, 6.0),
        "contact_phase_fraction": (0.0, 0.50),
        "jepa_surprise": (0.0, 1.0),
    },
    "llm_sft": {
        "mean_coherence": (0.0, 1.0),
        "repetition_rate": (0.0, 0.50),
        "template_ratio": (0.0, 0.50),
        "mean_response_length": (0.0, 200.0),
        "diversity_nn_dist": (0.0, 0.60),
    },
}


@app.post("/api/community/similar")
async def community_similar(
    body: SimilarRequest,
    user: sqlite3.Row = Depends(_auth_required),
):
    """
    Find the k most similar past outcomes in the community database for a
    given metric fingerprint. Pro feature: enables cross-lab blended predictions.

    `domain` selects the fingerprint schema ("robotics" or "llm_sft") — only
    outcomes recorded in the same domain are compared, since their metric sets
    (and therefore distances) aren't comparable across domains.
    """
    norm_ranges = _DOMAIN_NORM_RANGES.get(body.domain)
    if norm_ranges is None:
        raise HTTPException(400, f"unknown domain: {body.domain!r}")

    try:
        import numpy as np
    except ImportError:
        raise HTTPException(500, "numpy not available on server")

    fp = body.fingerprint
    keys = list(norm_ranges.keys())

    def _norm_vec(fingerprint: dict) -> "np.ndarray":
        vec = []
        for k in keys:
            val = fingerprint.get(k)
            if val is None:
                vec.append(0.5)
                continue
            lo, hi = norm_ranges[k]
            span = hi - lo
            vec.append(float(np.clip((val - lo) / span, 0.0, 1.0)) if span > 0 else 0.5)
        return np.array(vec, dtype=float)

    query_vec = _norm_vec(fp)

    where_clauses = ["domain = ?"]
    params: list = [body.domain]
    if body.policy_family != "generic":
        where_clauses.append("policy_family = ?")
        params.append(body.policy_family)
    where = " WHERE " + " AND ".join(where_clauses)

    if body.domain == "robotics":
        col_list = ", ".join(keys)
        with _get_conn() as conn:
            rows = conn.execute(
                f"SELECT id, policy_family, actual_rate, {col_list} FROM outcomes{where}",
                params,
            ).fetchall()
        row_fingerprints = [({k: row[k] for k in keys}, row) for row in rows]
    else:
        with _get_conn() as conn:
            rows = conn.execute(
                f"SELECT id, policy_family, actual_rate, fingerprint_json FROM outcomes{where}",
                params,
            ).fetchall()
        row_fingerprints = [
            (json.loads(row["fingerprint_json"]) if row["fingerprint_json"] else {}, row)
            for row in rows
        ]

    if not rows:
        return {"similar": [], "n_community": 0}

    scored = []
    for rec_fp, row in row_fingerprints:
        rec_vec = _norm_vec(rec_fp)
        dist = float(np.linalg.norm(query_vec - rec_vec))
        if body.policy_family != "generic" and row["policy_family"] == body.policy_family:
            dist *= 0.7
        if dist <= 0.4:
            scored.append({"id": row["id"], "policy_family": row["policy_family"], "actual_rate": row["actual_rate"], "distance": round(dist, 3)})

    scored.sort(key=lambda x: x["distance"])
    top_k = scored[: body.k]

    blended_score: Optional[float] = None
    if top_k:
        weights = np.array([1.0 / (s["distance"] + 1e-6) for s in top_k])
        weights /= weights.sum()
        blended_score = round(float(sum(w * s["actual_rate"] * 100.0 for w, s in zip(weights, top_k))), 1)

    return {
        "similar": top_k,
        "n_community": len(rows),
        "blended_predicted_success": blended_score,
    }


# ── GET /api/registry/public ───────────────────────────────────────────────────

@app.get("/api/registry/public")
async def public_registry():
    """Return a paginated list of all Calibra-certified datasets."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT dataset_id, status, updated_ts, calibra_version FROM badges "
            "WHERE status='CERTIFIED' ORDER BY updated_ts DESC LIMIT 200"
        ).fetchall()
    return {
        "datasets": [
            {
                "dataset_id": r["dataset_id"],
                "status": r["status"],
                "certified_at": r["updated_ts"],
                "calibra_version": r["calibra_version"],
            }
            for r in rows
        ],
        "total": len(rows),
    }


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
