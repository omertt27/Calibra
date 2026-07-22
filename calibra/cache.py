"""
calibra.cache — file-based audit result cache for incremental analysis.

Production pipelines audit datasets repeatedly. When the dataset hasn't changed,
re-running the full pipeline wastes CPU. This module provides a report-level
cache keyed by a deterministic batch fingerprint: the cache hits instantly on
unchanged data and misses when any episode is added, removed, or modified.

Layout
------
    <cache_dir>/
        reports/
            <fingerprint>.json   # serialized DiagnosticReport
        index.json               # {fingerprint: {source_path, n_episodes, created_at}}

Usage
-----
    from calibra.cache import AuditCache
    from calibra.pipeline import Pipeline

    cache = AuditCache(".calibra/cache")
    report = Pipeline().run(batch, policy_family="act", cache=cache)
    # Second call on unchanged data and same policy returns instantly.

    # Check cache state
    print(cache.stats())

    # Clear all cached reports
    cache.clear()

Module-level helpers (used without an AuditCache instance)
----------------------------------------------------------
    from calibra.cache import episode_content_hash, batch_episode_hashes

    hashes = batch_episode_hashes(batch)       # {episode_id: hash[:16]}
    fp     = batch_fingerprint(batch, "act")   # 24-char fingerprint
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_CACHE_DIR = ".calibra/cache"


# ── module-level hash helpers (importable without AuditCache) ─────────────────


def episode_content_hash(episode) -> str:
    """
    16-char SHA-256 content hash for a single episode.

    Based on timestamps + actions — the two fields that change when an episode
    is re-recorded, re-processed, or corrupted. Observations are excluded for
    speed; they correlate strongly with the action trajectory for kinematic data.
    """
    data = np.concatenate([
        episode.timestamps.astype(np.float32).flatten(),
        episode.actions.astype(np.float32).flatten(),
    ])
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


def batch_episode_hashes(batch) -> dict[str, str]:
    """Return {episode_id: content_hash[:16]} for every episode in a batch."""
    return {ep.metadata.episode_id: episode_content_hash(ep) for ep in batch.episodes}


def batch_fingerprint(batch, policy_family: Optional[str] = None) -> str:
    """
    Deterministic 24-char fingerprint for an (EpisodeBatch, policy) pair.

    Changes when any episode is added, removed, or modified, or when the target
    policy family changes (which affects which analyzers run).
    """
    pairs = sorted(
        (ep.metadata.episode_id, episode_content_hash(ep))
        for ep in batch.episodes
    )
    payload = json.dumps({"episodes": pairs, "policy": policy_family or ""})
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


# ── cache class ───────────────────────────────────────────────────────────────


class AuditCache:
    """
    File-based cache for Calibra ``DiagnosticReport`` results.

    Each cached entry maps a batch fingerprint to a serialized DiagnosticReport.
    The fingerprint encodes the full episode manifest (sorted episode_id + content
    hash pairs) and the policy family, so it automatically invalidates when the
    dataset or configuration changes.

    Parameters
    ----------
    cache_dir : directory for cached reports (default: ``".calibra/cache"``).
                Created automatically if it does not exist.
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self._reports_dir = self.cache_dir / "reports"
        self._index_path = self.cache_dir / "index.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def fingerprint(self, batch, policy_family: Optional[str] = None) -> str:
        """Compute the batch fingerprint for cache lookup."""
        return batch_fingerprint(batch, policy_family)

    def episode_hashes(self, batch) -> dict[str, str]:
        """Return {episode_id: content_hash} for all episodes."""
        return batch_episode_hashes(batch)

    def get(self, fingerprint: str):
        """
        Return the cached ``DiagnosticReport`` for *fingerprint*, or ``None`` on miss.

        A ``None`` return means the cache does not have a valid entry — the caller
        should run the full pipeline and then call ``put()``.
        """
        path = self._reports_dir / f"{fingerprint}.json"
        if not path.exists():
            return None
        try:
            from calibra.schema.report import DiagnosticReport
            return DiagnosticReport.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, fingerprint: str, report, batch=None) -> None:
        """
        Store *report* under *fingerprint*.

        Safe to call concurrently — writes atomically via a temp file on
        platforms that support ``Path.replace()``.
        """
        path = self._reports_dir / f"{fingerprint}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
        self._update_index(fingerprint, report)

    def has(self, fingerprint: str) -> bool:
        """Return True if the cache contains an entry for *fingerprint*."""
        return (self._reports_dir / f"{fingerprint}.json").exists()

    def clear(self) -> int:
        """Delete all cached reports. Returns the number of files deleted."""
        count = 0
        for f in self._reports_dir.glob("*.json"):
            f.unlink()
            count += 1
        if self._index_path.exists():
            self._index_path.unlink()
        return count

    def stats(self) -> str:
        """Return a human-readable summary of cache contents."""
        index = self._load_index()
        n = len(index)
        size_bytes = sum(
            f.stat().st_size for f in self._reports_dir.glob("*.json")
            if not f.name.endswith(".tmp")
        )
        lines = [
            "━" * 55,
            "  CALIBRA AUDIT CACHE",
            "━" * 55,
            f"  Directory : {self.cache_dir}",
            f"  Entries   : {n}",
            f"  Size      : {size_bytes / 1024:.1f} KB",
        ]
        for fp, meta in list(index.items())[:8]:
            lines.append(
                f"    {fp[:14]}…  {meta.get('source_path', '?')[:30]:<30}  "
                f"{meta.get('n_episodes', '?'):>5} eps  "
                f"{meta.get('created_at', '?')[:10]}"
            )
        if n > 8:
            lines.append(f"    … and {n - 8} more")
        lines.append("━" * 55)
        return "\n".join(lines)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _update_index(self, fingerprint: str, report) -> None:
        index = self._load_index()
        index[fingerprint] = {
            "source_path": report.source_path,
            "n_episodes": report.n_episodes,
            "n_samples": report.n_samples,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}
