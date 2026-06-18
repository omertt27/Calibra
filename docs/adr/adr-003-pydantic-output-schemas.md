# ADR-003: Pydantic v2 for All Output-Facing Schemas

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Calibra produces structured diagnostic output (flags, hints, reports,
comparison results, curation audits) that must be:
- Serializable to JSON for programmatic consumption.
- Validatable at construction time to catch analyzer bugs early.
- Stable as an API surface that downstream tools can depend on.

The internal ingestion types (`Episode`, `EpisodeBatch`) do not need
serialization; they are ephemeral objects that exist only during a pipeline
run.

---

## Decision

All output-facing schema types use Pydantic v2 `BaseModel`:

- `ObservedValue`, `RiskFlag`, `CompatibilityHint`, `AnalyzerResult`,
  `DiagnosticReport` — in `calibra/schema/report.py`
- `DriftFlag`, `ComparisonReport`, `EpisodeFlag`, `CurationReport` —
  in `calibra/schema/comparison.py`

The internal ingestion types (`Episode`, `EpisodeBatch`, `EpisodeMetadata`)
use Python `dataclass` instead of Pydantic. They are not serialized to users
and do not need Pydantic's validation overhead for the hot path through
analyzers.

The split rule is:
> **If it crosses the boundary between Calibra and a user/downstream tool,
> it is Pydantic. If it stays inside the pipeline, it is a dataclass.**

---

## Alternatives Considered

### Plain dataclasses everywhere
Use `@dataclass` for all types, with manual `to_dict()` / `from_dict()` methods.
Rejected: serialization boilerplate is high; field validation must be written
manually; the resulting JSON schema is undocumented.

### TypedDict + marshmallow
TypedDict for type hints, marshmallow for (de)serialization.
Rejected: two separate type systems for the same thing; marshmallow is an
additional dependency not already in the codebase.

### attrs
`attrs` with `cattrs` for conversion.
Rejected: less ecosystem momentum than Pydantic; `cattrs` is an additional
dependency; Pydantic v2 is fast enough to not be a concern at report
construction time.

### Pydantic for all types (including EpisodeBatch)
Use Pydantic models even for internal ingestion types.
Rejected: Pydantic validation on every episode construction adds measurable
overhead on large datasets (thousands of episodes × many steps). The hot path
must remain free of validation overhead. Internal types are not user-facing
and do not need the validation guarantee.

---

## Constraints

- `numpy` arrays may not appear as fields in Pydantic models, because Pydantic
  cannot serialize them to JSON. Numeric data that must appear in reports is
  stored as `Optional[float]` scalars. The `raw_metrics: dict[str, Any]` bag
  on `AnalyzerResult` holds arbitrary numeric structures for downstream
  consumers that need the full data.
- `RiskLevel` is defined as a `str` enum to ensure JSON serialization produces
  the string value rather than an integer.

---

## Consequences

**Positive:**
- `report.model_dump()` and `report.model_dump_json()` work without custom
  serialization code.
- Field mismatches (e.g. an analyzer returning a malformed `AnalyzerResult`)
  raise a `ValidationError` at construction, not silently at serialization time.
- Downstream tools can generate type stubs from the JSON schema.

**Negative:**
- Two type system conventions (`dataclass` vs. `BaseModel`) exist in the
  codebase. Contributors must know the split rule. It is documented in this
  ADR and in module docstrings.
- `raw_metrics: dict[str, Any]` is untyped and cannot be validated. This is
  intentional — it is an escape hatch, not the primary output channel.

---

## References

- `calibra/schema/report.py` — all Phase 1 output models
- `calibra/schema/comparison.py` — all Phase 2 output models
- `calibra/schema/episode.py` — internal dataclass types
- `pyproject.toml` — `pydantic>=2.0` core dependency
