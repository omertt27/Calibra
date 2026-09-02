"""
calibra.metrics_ingest — Read measured training metrics from a finished run.

The design-partner protocol (see calibra.experiment_log) needs GPU-hours,
wall-clock, eval success rate and loss recorded consistently for every
full / random / Calibra training run. Typing those in by hand invites
transcription errors — and the credibility of the full-vs-random-vs-Calibra
comparison rests on those numbers being right.

This module reads them straight out of whatever the training job already
emitted:

  * A flat JSON metrics/summary file — any keys; an alias table maps common
    names, and `parse_field_map` / `--map` extends it for anything unusual.
  * A Weights & Biases *offline* run summary (``wandb-summary.json``), read
    from disk — no network, no ``wandb`` import.

Nothing here touches the network. Like calibra.experiment_log, measured
partner results never leave the machine as a side effect of being read.

GPU-hours is only ever filled when it is literally present in the source.
It is deliberately *not* derived from wall-clock × GPU count here: a derived
figure must not be mistaken for a measured one by
calibra.benchmark's measured/simulated classifier. Pass ``--gpu-hours``
explicitly if you want to record a derived value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

# Canonical record field -> source keys that mean the same thing (lowercased,
# "/"-joined for nested keys). First match wins; `--map` overrides.
_ALIASES: dict[str, list[str]] = {
    "gpu_hours": ["gpu_hours", "gpu_hrs", "gpu_time_h", "gpu_time_hours", "gpuhours"],
    "wall_clock_seconds": [
        "wall_clock_seconds",
        "wall_clock_s",
        "wall_clock",
        "runtime_s",
        "runtime_seconds",
        "elapsed_sec",
        "elapsed_seconds",
        "train_runtime",
        "_runtime",
        "_wandb/runtime",
    ],
    "energy_kwh": ["energy_kwh", "energy_kw_h", "total_energy_kwh", "energy"],
    "eval_success_rate": [
        "eval_success_rate",
        "eval_success",
        "eval/success_rate",
        "eval/success",
        "success_rate",
        "final_success_rate",
        "sr",
    ],
    "training_loss": ["training_loss", "train_loss", "train/loss", "final_loss", "loss"],
}

_CANONICAL = frozenset(_ALIASES)

_DIR_CANDIDATES = (
    "wandb-summary.json",
    "files/wandb-summary.json",
    "latest-run/files/wandb-summary.json",
    "wandb/latest-run/files/wandb-summary.json",
    "metrics.json",
    "summary.json",
    "results.json",
)


@dataclass
class MetricsBundle:
    """Measured values pulled from one finished run, ready to hand to
    ExperimentLog.record(). Any field left None was not found in the source."""

    gpu_hours: Optional[float] = None
    wall_clock_seconds: Optional[float] = None
    energy_kwh: Optional[float] = None
    training_loss: Optional[float] = None
    eval_success_rate: Optional[float] = None
    # canonical field -> the source key (or `--map` path) it came from
    matched: dict[str, str] = field(default_factory=dict)
    # provenance string, e.g. "wandb:/path/to/wandb-summary.json"
    source: str = ""
    raw: dict = field(default_factory=dict)


def parse_field_map(pairs: list[str]) -> dict[str, str]:
    """Turn ``["eval_success_rate=results.eval.success", ...]`` into a dict.

    LHS must be one of the canonical ExperimentRecord metric fields; RHS is a
    "."- or "/"-separated key path into the metrics file.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--map expects FIELD=PATH, got {pair!r}")
        raw_field, _, path = pair.partition("=")
        f = raw_field.strip()
        if f not in _CANONICAL:
            raise ValueError(
                f"--map: unknown field {f!r}; valid fields: {', '.join(sorted(_CANONICAL))}"
            )
        if not path.strip():
            raise ValueError(f"--map {f}=: missing key path")
        out[f] = path.strip()
    return out


def load_metrics(
    path: Union[str, Path],
    fmt: str = "auto",
    extra_map: Optional[dict[str, str]] = None,
) -> MetricsBundle:
    """Read a metrics file (or a run directory containing one) into a MetricsBundle.

    `fmt` is "auto" | "json" | "wandb" — it only steers the provenance label
    and directory search; the parse itself is format-independent (flatten the
    JSON, match keys against the alias table).
    """
    p = Path(path)
    if p.is_dir():
        p = _find_in_dir(p)
    if not p.exists():
        raise FileNotFoundError(f"{path}: no such file")

    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p.name}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{p.name}: expected a JSON object, got {type(raw).__name__}")

    flat = _flatten(raw)
    values, matched = _extract(flat)

    for canon, keypath in (extra_map or {}).items():
        v = _lookup(flat, keypath)
        if v is None:
            raise ValueError(f"--map {canon}={keypath}: no numeric value at that path in {p.name}")
        values[canon] = v
        matched[canon] = keypath

    if "eval_success_rate" in values:
        values["eval_success_rate"], note = _normalize_rate(values["eval_success_rate"])
        if note:
            matched["eval_success_rate"] = matched.get("eval_success_rate", "?") + note

    label = "wandb" if (fmt == "wandb" or p.name == "wandb-summary.json") else "json"
    return MetricsBundle(
        gpu_hours=values.get("gpu_hours"),
        wall_clock_seconds=values.get("wall_clock_seconds"),
        energy_kwh=values.get("energy_kwh"),
        training_loss=values.get("training_loss"),
        eval_success_rate=values.get("eval_success_rate"),
        matched=matched,
        source=f"{label}:{p}",
        raw=raw,
    )


# ── internals ────────────────────────────────────────────────────────────────


def _find_in_dir(d: Path) -> Path:
    for candidate in _DIR_CANDIDATES:
        c = d / candidate
        if c.exists():
            return c
    raise FileNotFoundError(
        f"{d}: none of {', '.join(_DIR_CANDIDATES)} found — "
        f"point --from-metrics at the metrics file directly"
    )


def _flatten(obj, parent: str = "") -> dict:
    """Flatten nested dicts to {'a/b/c': leaf}, lowercasing the full key path.

    Non-dict, non-scalar leaves (lists, nested arrays) are dropped — the alias
    table only ever wants scalars.
    """
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent}/{k}" if parent else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, bool):
        pass  # a bool is an int in Python; never a metric value we want
    elif isinstance(obj, (int, float)):
        out[parent.lower()] = obj
    return out


def _extract(flat: dict) -> tuple[dict, dict]:
    values: dict[str, float] = {}
    matched: dict[str, str] = {}
    for canon, keys in _ALIASES.items():
        for k in keys:
            if k in flat:
                values[canon] = float(flat[k])
                matched[canon] = k
                break
    return values, matched


def _lookup(flat: dict, keypath: str) -> Optional[float]:
    for cand in (keypath.lower().replace(".", "/"), keypath.lower()):
        if cand in flat:
            return float(flat[cand])
    return None


def _normalize_rate(v: float) -> tuple[float, str]:
    """Eval success rate: fraction in [0, 1] is used as-is; a value in (1, 100]
    is treated as a percentage and divided by 100. Anything above 100 is left
    alone so ExperimentRecord's own [0, 1] validation rejects it loudly."""
    if 1.0 < v <= 100.0:
        return v / 100.0, " (read as percent)"
    return v, ""
