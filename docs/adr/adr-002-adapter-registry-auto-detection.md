# ADR-002: Decorator-Based Adapter Registry with Probe-Based Auto-Detection

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Calibra ships multiple format adapters (LeRobot, RLDS, HDF5, MCAP, Isaac Lab).
A user calling `calibra analyze /path/to/dataset` should not have to specify
the format if it can be inferred from the path. At the same time, format
dependencies are optional extras — an adapter may not be importable if the
user hasn't installed its dependencies.

The registry must:
1. Allow adapters to register themselves without any central manifest.
2. Auto-detect formats at runtime by probing the given path.
3. Fail gracefully when an adapter's dependencies are missing.
4. Support explicit reader injection for tests and ambiguous paths.

---

## Decision

Adapters self-register at import time using a `@register` class decorator
defined in `calibra/ingestion/registry.py`. The decorator appends the adapter
class to a module-level list `_READERS`.

`detect_reader(path)` iterates `_READERS` in registration order, calling
`cls.can_read(path)` on each until one returns `True`. The first match wins.
If no adapter claims the path, a `ValueError` is raised with a list of
registered readers and installation hints.

The public entry point `load(path, reader=None)` accepts an optional `reader`
argument. When supplied, the reader is used directly, bypassing detection.
This allows tests to pass a synthetic reader without registering it, and
allows callers to resolve ambiguity when the path heuristic is insufficient.

Registration order determines probe priority. Adapters that require more
specific path conditions (e.g. the presence of `meta/info.json`) should be
registered before more permissive adapters.

---

## Alternatives Considered

### Central manifest (e.g. pyproject.toml entry points)
Register adapters via setuptools entry points in `pyproject.toml`.
Rejected: requires an installed package for discovery; complicates editable
installs and testing; makes the registration location non-obvious to contributors.

### Format flag required on every CLI call
Force users to always supply `--format lerobot`.
Rejected: degrades UX for the common case; auto-detection is a documented
feature goal.

### Try-all-adapters-and-return-first-success
Load each adapter and return the first one that doesn't raise an exception.
Rejected: side effects of partial reads can corrupt state; also slow when
multiple adapters partially match.

---

## Constraints

- `can_read(path)` must be fast (filesystem stat/glob, JSON key check) and
  must not perform any data loading.
- `can_read(path)` must not raise exceptions; it returns `bool`.
- Adapters whose format dependencies are not installed must still register —
  `can_read` may return `True`, but `read` will raise an `ImportError` with
  an installation hint. This preserves the diagnostic message
  ("No reader found" vs. "Install calibra[lerobot]") for the user.

---

## Consequences

**Positive:**
- Adding a new adapter is a one-file change; no manifest to update.
- The probe order is explicit and documented by import order.
- Tests can inject synthetic readers without touching the global registry.
- Users get actionable error messages when format dependencies are missing.

**Negative:**
- Registration order is determined by Python import order, which is implicit.
  If two adapters have overlapping `can_read` heuristics, the first one
  registered wins, which may be surprising. This is documented in
  `calibra/ingestion/registry.py`.
- The global `_READERS` list is module-level mutable state; concurrent
  registration from threads is not safe, but this is not a concern in the
  current single-threaded CLI context.

---

## References

- `calibra/ingestion/registry.py` — `register`, `detect_reader`, `load`
- `calibra/ingestion/adapters/` — all registered adapters
