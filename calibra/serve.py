"""
calibra serve — FastAPI server exposing Calibra's analysis pipeline as a REST API.

Serves the web dashboard as static files and provides a machine-friendly JSON
API for all Calibra analysis commands.

Usage:
    calibra serve                    # start on default port 7842
    calibra serve --port 8000        # custom port
    calibra serve --host 0.0.0.0     # expose on all interfaces
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional

_START_TIME = time.monotonic()
_WEB_DIST = Path(__file__).parent.parent / "web" / "dist"


# ── helpers ───────────────────────────────────────────────────────────────────


def _raw_metrics(report, analyzer: str) -> dict:
    for r in report.analyzer_results:
        if r.analyzer_name == analyzer:
            return r.raw_metrics
    return {}


def _overall_status(report) -> str:
    from calibra.schema.report import RiskLevel

    if report.flags_at_level(RiskLevel.CRITICAL):
        return "NOT_CERTIFIED"
    if report.flags_at_level(RiskLevel.WARNING):
        return "PROVISIONALLY_CERTIFIED"
    return "CERTIFIED"


def _normalize_flags(report) -> list[dict]:
    """Map DiagnosticReport flags to the frontend-friendly flat shape."""
    result = []
    for flag in report.flags:
        obs = flag.observed
        obs_str = str(obs)
        if flag.threshold is not None:
            unit = obs.unit or ""
            thresh_str = f"<{flag.threshold:.4g}{(' ' + unit) if unit else ''}"
        else:
            thresh_str = None
        result.append(
            {
                "level": flag.level.value.lower(),
                "metric": flag.metric,
                "observed": obs_str,
                "threshold": thresh_str,
                "msg": flag.implication,
            }
        )
    return result


def _load_report(path: str, policy: Optional[str] = None, fmt: Optional[str] = None):
    """Synchronously load a dataset and run the full diagnostic pipeline."""
    from calibra.pipeline import Pipeline

    reader = None
    if fmt:
        from calibra.__main__ import _get_reader

        reader = _get_reader(fmt)
    return Pipeline().analyze_path(path, policy_family=policy, reader=reader)


# ── application factory ───────────────────────────────────────────────────────


def _make_app():
    """Build and return the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ImportError as exc:
        raise SystemExit(
            "FastAPI/uvicorn not installed. Run:  pip install 'calibra-robotics[serve]'\n"
            f"  {exc}"
        ) from exc

    from calibra import __version__

    app = FastAPI(
        title="Calibra API",
        description="Dataset reliability diagnostics for robotics IL",
        version=__version__,
    )

    # Enable CORS for all origins (local web dev: React runs on a different port)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── request bodies ────────────────────────────────────────────────────────

    class AnalyzeRequest(BaseModel):
        path: str
        policy: Optional[str] = None
        format: Optional[str] = None

    class CompareRequest(BaseModel):
        path: str
        reference: str
        format: Optional[str] = None

    class CertifyRequest(BaseModel):
        path: str
        reference: Optional[str] = None
        policy: Optional[str] = None
        strict: bool = False

    class PredictRequest(BaseModel):
        path: str
        policy: Optional[str] = None
        reference: Optional[str] = None

    class ScoreRequest(BaseModel):
        path: str
        reference: Optional[str] = None

    class RecordOutcomeRequest(BaseModel):
        path: str
        actual_rate: float
        policy: Optional[str] = None

    # ── executor wrapper ──────────────────────────────────────────────────────

    async def _in_thread(func):
        """Run a zero-argument synchronous callable in a thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    # ── GET /health ───────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": round(time.monotonic() - _START_TIME, 3),
        }

    # ── POST /analyze ─────────────────────────────────────────────────────────

    @app.post("/analyze")
    async def analyze(req: AnalyzeRequest):
        """
        Run the full Calibra diagnostic pipeline and return a normalized
        frontend-friendly report.
        """

        def _run():
            report = _load_report(req.path, req.policy, req.format)

            t = _raw_metrics(report, "temporal_stability")
            s = _raw_metrics(report, "control_smoothness")
            c = _raw_metrics(report, "coverage_entropy")

            jerk_spike_raw = s.get("jerk_spikes", {}).get("mean_spike_fraction")
            vel_disc_raw = s.get("vel_discontinuities", {}).get("mean_disc_fraction")
            ldlj = s.get("ldlj", {}).get("mean_ldlj")
            dropout_raw = t.get("dropout", {}).get("mean_dropout_fraction")
            jitter_cv = t.get("jitter", {}).get("mean_cv")
            action_entropy = c.get("action_entropy", {}).get("entropy_bits_per_dim")

            # Express rate metrics as percentages (matching the example schema)
            jerk_spike_pct = (
                round(jerk_spike_raw * 100, 2) if jerk_spike_raw is not None else None
            )
            vel_disc_pct = (
                round(vel_disc_raw * 100, 2) if vel_disc_raw is not None else None
            )
            dropout_pct = round(dropout_raw * 100, 2) if dropout_raw is not None else None

            # Episode-level outlier detection
            outlier_indices: list[int] = []
            try:
                from calibra.anomalies import find_outliers

                outlier_indices = [a.episode_idx for a in find_outliers(report)]
            except Exception:
                pass

            # Composite score
            score_val = None
            try:
                from calibra.score import compute_score

                score_val = compute_score(report)["total_score"]
            except Exception:
                pass

            # Predicted success rate
            predicted_success_rate = None
            try:
                from calibra.predict import predict_outcome

                predicted_success_rate = predict_outcome(
                    report, policy_family=req.policy
                )["predicted_success_rate"]
            except Exception:
                pass

            return {
                "name": report.dataset_name,
                "path": report.source_path,
                "episodes": report.n_episodes,
                "steps": report.n_samples,
                "jerk_spike_rate": jerk_spike_pct,
                "vel_discontinuity": vel_disc_pct,
                "ldlj": round(ldlj, 4) if ldlj is not None else None,
                "dropout_rate": dropout_pct,
                "jitter_cv": round(jitter_cv, 5) if jitter_cv is not None else None,
                "action_entropy": (
                    round(action_entropy, 4) if action_entropy is not None else None
                ),
                "overall_status": _overall_status(report),
                "flags": _normalize_flags(report),
                "outliers": outlier_indices,
                "score": score_val,
                "predicted_success_rate": predicted_success_rate,
            }

        try:
            return await _in_thread(_run)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail={"error": str(exc), "detail": repr(exc)}
            )

    # ── POST /compare ─────────────────────────────────────────────────────────

    @app.post("/compare")
    async def compare(req: CompareRequest):
        """
        Compare a dataset against a named reference profile and return
        structured metric deltas plus recommended actions.
        """

        def _run():
            from calibra.pipeline import Pipeline
            from calibra.analyzers.coverage import CoverageEntropyAnalyzer
            from calibra.analyzers.smoothness import ControlSmoothnessAnalyzer
            from calibra.analyzers.task_structure import TaskStructureAnalyzer
            from calibra.analyzers.temporal import TemporalAnalyzer
            from calibra.compare import (
                _recommended_actions,
                _ref_is_sim,
                load_reference,
                metrics_from_reference,
                metrics_from_report,
            )
            from calibra.anomalies import find_outliers

            reader = None
            if req.format:
                from calibra.__main__ import _get_reader

                reader = _get_reader(req.format)

            ref_data = load_reference(req.reference)
            pipeline = Pipeline(
                analyzers=[
                    TemporalAnalyzer(),
                    ControlSmoothnessAnalyzer(),
                    CoverageEntropyAnalyzer(),
                    TaskStructureAnalyzer(),
                ]
            )
            report = pipeline.analyze_path(req.path, reader=reader)

            your_metrics = metrics_from_report(report)
            ref_metrics = metrics_from_reference(ref_data)
            outlier_episodes = find_outliers(report)

            _METRIC_LABELS = [
                ("vel_disc_rate", "Velocity Discontinuity Rate"),
                ("spike_rate", "Jerk Spike Rate"),
                ("ldlj", "LDLJ"),
                ("jitter_cv", "Timestamp Jitter CV"),
                ("dropout_rate", "Timestamp Dropout Rate"),
                ("action_entropy", "Action Entropy (bits/dim)"),
            ]
            metrics_list = []
            for key, label in _METRIC_LABELS:
                y = your_metrics.get(key)
                r = ref_metrics.get(key)
                delta = (y - r) if (y is not None and r is not None) else None
                metrics_list.append(
                    {
                        "name": key,
                        "label": label,
                        "yours": round(y, 6) if y is not None else None,
                        "reference": round(r, 6) if r is not None else None,
                        "delta": round(delta, 6) if delta is not None else None,
                    }
                )

            recommended = _recommended_actions(
                req.path,
                your_metrics,
                ref_metrics,
                ref_is_sim=_ref_is_sim(ref_metrics),
                outlier_episodes=outlier_episodes,
            )

            return {
                "dataset": report.dataset_name,
                "reference": req.reference,
                "metrics": metrics_list,
                "recommended_actions": recommended,
            }

        try:
            return await _in_thread(_run)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail={"error": str(exc), "detail": repr(exc)}
            )

    # ── POST /certify ─────────────────────────────────────────────────────────

    @app.post("/certify")
    async def certify(req: CertifyRequest):
        """
        Grade a dataset on the CERTIFIED / PROVISIONALLY_CERTIFIED / NOT_CERTIFIED
        scale and return itemised warnings and failures.
        """

        def _run():
            from calibra.pipeline import Pipeline
            from calibra.certify import _grade, _is_scripted_report
            from calibra.schema.report import RiskLevel

            report = Pipeline().analyze_path(req.path, policy_family=req.policy)

            grade, exit_code = _grade(report)
            if req.strict and exit_code == 1:
                exit_code = 2

            is_scripted = _is_scripted_report(report)
            _SCRIPTED_EXEMPT = frozenset({"spike_rate", "jerk_spikes"})

            failures = [
                {"metric": f.metric, "msg": f.interpretation}
                for f in report.flags_at_level(RiskLevel.CRITICAL)
                if not (is_scripted and f.metric in _SCRIPTED_EXEMPT)
            ]
            warnings_list = [
                {"metric": f.metric, "msg": f.interpretation}
                for f in report.flags_at_level(RiskLevel.WARNING)
                if not (is_scripted and f.metric in _SCRIPTED_EXEMPT)
            ]

            # Normalise grade to underscore form (e.g. "PROVISIONALLY CERTIFIED" →
            # "PROVISIONALLY_CERTIFIED") so JSON consumers don't have to handle spaces.
            status = grade.replace(" ", "_")

            return {
                "status": status,
                "warnings": warnings_list,
                "failures": failures,
            }

        try:
            return await _in_thread(_run)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail={"error": str(exc), "detail": repr(exc)}
            )

    # ── POST /predict ─────────────────────────────────────────────────────────

    @app.post("/predict")
    async def predict(req: PredictRequest):
        """
        Predict the expected training outcome (success-rate range and tier)
        for a dataset.
        """

        def _run():
            from calibra.pipeline import Pipeline
            from calibra.predict import predict_outcome

            report = Pipeline().analyze_path(req.path, policy_family=req.policy)
            result = predict_outcome(report, policy_family=req.policy)

            return {
                "predicted_score": result["predicted_score"],
                "tier": result["tier"],
                "range": result["predicted_range"],
                "deductions": result["deductions"],
            }

        try:
            return await _in_thread(_run)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail={"error": str(exc), "detail": repr(exc)}
            )

    # ── POST /score ───────────────────────────────────────────────────────────

    @app.post("/score")
    async def score(req: ScoreRequest):
        """Compute the composite Calibra Score (0–100) for a dataset."""

        def _run():
            from calibra.pipeline import Pipeline
            from calibra.score import compute_score

            report = Pipeline().analyze_path(req.path)
            result = compute_score(report)

            return {
                "score": result["total_score"],
                "category": result["category"],
                "dimensions": result["dimensions"],
            }

        try:
            return await _in_thread(_run)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail={"error": str(exc), "detail": repr(exc)}
            )

    # ── GET /outcomes ─────────────────────────────────────────────────────────

    @app.get("/outcomes")
    async def get_outcomes():
        """Return all recorded training outcomes from the local outcome database."""
        try:
            from calibra.outcome_db import OutcomeDatabase

            db = OutcomeDatabase()
            records = db.list_records()
            return {"records": records, "count": len(records)}
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail={"error": str(exc), "detail": repr(exc)}
            )

    # ── POST /outcomes ────────────────────────────────────────────────────────

    @app.post("/outcomes")
    async def record_outcome(req: RecordOutcomeRequest):
        """
        Record an observed training success rate for a dataset so future
        predictions on similar datasets benefit from empirical blending.
        """
        if not (0.0 <= req.actual_rate <= 1.0):
            raise HTTPException(
                status_code=400,
                detail={"error": "actual_rate must be between 0.0 and 1.0"},
            )

        def _run():
            from calibra.pipeline import Pipeline
            from calibra.predict import predict_outcome
            from calibra.outcome_db import OutcomeDatabase

            report = Pipeline().analyze_path(req.path, policy_family=req.policy)
            pred = predict_outcome(report, policy_family=req.policy)

            db = OutcomeDatabase()
            rec = db.record(
                fingerprint=pred["metric_values"],
                predicted_score=pred["heuristic_score"],
                actual_success_rate=req.actual_rate,
                policy_family=pred["policy_family"],
                n_episodes=pred["n_episodes"],
                dataset_name=pred["dataset_name"],
            )
            return {"recorded": True, "record_id": rec.record_id}

        try:
            return await _in_thread(_run)
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail={"error": str(exc), "detail": repr(exc)}
            )

    # ── static files (web dashboard SPA) ─────────────────────────────────────

    if _WEB_DIST.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(_WEB_DIST), html=True),
            name="web",
        )

    return app


# ── banner ────────────────────────────────────────────────────────────────────


def _print_banner(host: str, port: int) -> None:
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    base_url = f"http://{display_host}:{port}"
    width = 60
    div = "─" * width

    print(f"\n{div}")
    print("  Calibra API Server")
    print(div)
    print(f"  URL   : {base_url}")
    print(f"  Docs  : {base_url}/docs  (Swagger UI)")
    print()
    print("  Endpoints:")
    print("    GET  /health")
    print("    POST /analyze")
    print("    POST /compare")
    print("    POST /certify")
    print("    POST /predict")
    print("    POST /score")
    print("    GET  /outcomes")
    print("    POST /outcomes")
    if _WEB_DIST.exists():
        print()
        print(f"  Dashboard : {base_url}/")
    else:
        print()
        print("  Dashboard : not built  (run `npm run build` in web/)")
    print(f"{div}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────


def run_serve(argv: list[str]) -> None:
    """Called by calibra.__main__ when the user runs `calibra serve ...`."""
    p = argparse.ArgumentParser(
        prog="calibra serve",
        description="Start the Calibra REST API server.",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=7842,
        help="Port to listen on (default: 7842)",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = p.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn not installed. Run:  pip install 'calibra-robotics[serve]'"
        )

    _print_banner(args.host, args.port)

    app = _make_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
