# ADR-009: Two-Phase Pipeline — Phase 1 Analysis then Phase 2 Comparison

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

A complete Calibra workflow involves:
1. Profiling a dataset (running analyzers to produce metrics and flags).
2. Comparing the result against a reference profile (another profiled dataset
   stored as a JSON reference).

These two operations have different performance characteristics and caching
needs. Phase 1 (running analyzers over a full dataset) is expensive: it
iterates every episode and every step. Phase 2 (comparing two sets of
pre-computed metrics) is cheap: it operates on scalar summaries.

A user might want to compare one dataset against multiple references, or
compare multiple datasets against one reference. Re-running Phase 1 for
every comparison is wasteful.

Additionally, `calibra compare` must be usable without a live dataset — e.g.
to compare two previously saved reference profiles. This is only possible if
the comparison layer accepts pre-computed reports rather than raw datasets.

---

## Decision

The pipeline is explicitly divided into two phases with distinct inputs and
outputs:

**Phase 1 — `Pipeline.run(batch) → DiagnosticReport`**
- Input: `EpisodeBatch` (live data from the ingestion layer)
- Output: `DiagnosticReport` (Pydantic model, JSON-serializable)
- Implemented in: `calibra/pipeline.py`, `calibra/analyzers/`

**Phase 2 — `DatasetComparator.compare(baseline_report, candidate_report) → ComparisonReport`**
- Input: two `DiagnosticReport` objects (can be loaded from cached JSON)
- Output: `ComparisonReport` (Pydantic model, JSON-serializable)
- Implemented in: `calibra/comparison/comparator.py`

`DiagnosticReport` is the stable boundary between the two phases. It contains
both the scalar metric summaries (used by Phase 2 for comparison) and the
per-episode raw values (stored in `raw_metrics` under the `per_episode_<key>`
convention, used by Phase 2 for permutation testing).

The CLI `calibra compare` materializes Phase 1 by calling `Pipeline.run()`
internally when given a live path, then passes the result directly to
Phase 2 without requiring the user to save an intermediate file. Saving and
reloading a `DiagnosticReport` for deferred comparison is supported but
not required.

---

## Alternatives Considered

### Single pass: analyzers produce comparison results directly
Run comparison during Phase 1 by passing the reference profile into each
analyzer.
Rejected: couples analyzers to the reference format; makes single-dataset
profiling (without any reference) a special case; prevents comparing the
same analysis run against multiple references.

### Streaming comparison (process episodes from both datasets simultaneously)
Load both datasets and compare episode-by-episode without materializing
a full DiagnosticReport.
Rejected: cross-dataset permutation tests require having all per-episode
values available for both datasets before computing p-values. True streaming
comparison is not statistically possible for the tests Calibra uses.

### Store raw episode data in DiagnosticReport
Include the full `EpisodeBatch` (or all raw arrays) in the DiagnosticReport
so Phase 2 can recompute any metric.
Rejected: DiagnosticReport is a reporting artifact meant to be compact and
shareable; embedding GB of raw arrays defeats this purpose. The per-episode
scalar summaries stored in `raw_metrics["per_episode_<key>"]` are sufficient
for permutation testing.

---

## Constraints

- The `per_episode_<key>` convention in `raw_metrics` is the contract between
  Phase 1 analyzers and Phase 2 permutation testing. An analyzer that wants
  its metric to be permutation-testable must store a list of per-episode
  scalar values under this key. This is documented in `comparator.py`.
- Phase 2 falls back to a CI non-overlap heuristic for metrics where
  per-episode data is unavailable (e.g. batch-level entropy, PCA variance).
  This fallback is less statistically rigorous and is labelled as such in the
  `ComparisonReport`.

---

## Consequences

**Positive:**
- Running Phase 1 once and comparing against N references costs O(1×Phase 1 +
  N×Phase 2) rather than O(N×Phase 1 + N×Phase 2).
- `DiagnosticReport` is a stable artifact that can be stored in CI, shared
  between team members, or used as a reference profile.
- The comparison layer can be tested against synthetic `DiagnosticReport`
  objects without loading any dataset.

**Negative:**
- The `per_episode_<key>` convention in `raw_metrics` is informal
  (untyped `dict[str, Any]`). If an analyzer names its per-episode data
  differently, permutation testing silently falls back to the CI heuristic.
  This is a known documentation debt.

---

## References

- `calibra/pipeline.py` — Phase 1 implementation
- `calibra/comparison/comparator.py` — Phase 2 implementation,
  `per_episode_<key>` convention documentation
- `calibra/schema/report.py` — `DiagnosticReport` (Phase 1/2 boundary)
- `calibra/schema/comparison.py` — `ComparisonReport`
- `calibra/compare.py` — CLI wiring of Phase 1 → Phase 2
