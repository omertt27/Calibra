# Architectural Decision Records

This directory records the significant architectural decisions made in Calibra.
Each ADR captures the context, decision, alternatives considered, and consequences
at the time the decision was made.

ADRs are append-only. A superseded decision should have its status updated and
a new ADR written to replace it — the original is never deleted.

## Index

| ID | Title | Status |
|----|-------|--------|
| [ADR-001](adr-001-episodebatch-interchange-format.md) | EpisodeBatch as the universal ingestion-to-analysis interchange format | Accepted |
| [ADR-002](adr-002-adapter-registry-auto-detection.md) | Decorator-based adapter registry with probe-based auto-detection | Accepted |
| [ADR-003](adr-003-pydantic-output-schemas.md) | Pydantic v2 for all output-facing schemas | Accepted |
| [ADR-004](adr-004-stateless-analyzer-pattern.md) | Stateless Analyzer contract | Accepted |
| [ADR-005](adr-005-observation-key-normalization.md) | Configuration-driven observation key normalization layer | Accepted |
| [ADR-006](adr-006-duckdb-lerobot-v2-fast-path.md) | DuckDB fast path for LeRobot v2 Parquet scanning | Accepted |
| [ADR-007](adr-007-optional-format-dependencies.md) | Optional format dependencies with lazy imports | Accepted |
| [ADR-008](adr-008-claims-registry.md) | Evidence-backed falsifiable claims registry | Accepted |
| [ADR-009](adr-009-two-phase-pipeline.md) | Two-phase pipeline: Phase 1 analysis then Phase 2 comparison | Accepted |
| [ADR-010](adr-010-claims-references-parity.md) | Claims/references parity rule | Accepted |
