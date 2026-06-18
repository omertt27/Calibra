# ADR-004: Stateless Analyzer Contract

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Analyzers are the diagnostic units of Calibra's Phase 1 pipeline. Each
analyzer receives an `EpisodeBatch` and returns an `AnalyzerResult`. As
the number of analyzers grows, and as the pipeline becomes composable
(users can pass custom analyzer lists to `Pipeline`), correctness depends
on analyzers being independent of each other.

Policy-family-specific analysis (e.g. GR00T compatibility checks) is a
recurring need. There are two ways to handle it: branch inside existing
analyzers on `policy_family`, or attach specialized analyzers conditionally.

---

## Decision

Analyzers must be stateless. Specifically:

1. An `Analyzer` subclass must not mutate the `EpisodeBatch` it receives.
2. An `Analyzer` subclass must not retain state between calls to `analyze()`.
   Two calls to `analyzer.analyze(batch)` with the same input must return
   equivalent results.
3. Analyzers must not communicate with each other. If analyzer B needs a
   result from analyzer A, that result must come from the `EpisodeBatch`
   (i.e., it must be precomputed and stored in the batch), not from
   inter-analyzer state.

Policy-family-specific analyzers are implemented as separate `Analyzer`
subclasses (e.g. `GR00TCompatibilityAnalyzer`) and appended to the pipeline's
analyzer list conditionally by `Pipeline.run()` when the relevant policy
family is detected. They are not injected as branches inside other analyzers.

---

## Alternatives Considered

### Stateful analyzers that accumulate results across batches
Allow analyzers to maintain running statistics across multiple `analyze()`
calls, accumulating results before producing a final report.
Rejected: complicates the pipeline contract (callers must know to call
`flush()` or similar); makes testing harder; prevents parallelization.

### Policy-family branching inside existing analyzers
Add `if policy_family == "gr00t": ...` branches inside `TemporalAnalyzer`,
`ControlSmoothnessAnalyzer`, etc.
Rejected: pollutes general-purpose analyzers with policy-specific knowledge;
every general analyzer must import policy-specific logic; removing a policy
family requires surgery across multiple files. Separate analyzer classes are
self-contained and independently testable.

### Analyzer composition with a shared context object
Pass a shared `AnalysisContext` object to all analyzers so they can read
results from each other.
Rejected: creates hidden ordering dependencies; makes the execution order
semantically significant in non-obvious ways; prevents future parallelization
of independent analyzers.

---

## Constraints

- The `analyze()` method signature is fixed:
  `analyze(self, batch: EpisodeBatch, policy_family: Optional[str] = None) -> AnalyzerResult`
  Policy-family-specific hints are returned inside the `AnalyzerResult.hints`
  list, not via a separate code path.
- The `name` property must return a stable string (not derived from
  `id(self)` or any mutable state) — it is used as a key in
  `DiagnosticReport.timing`.

---

## Consequences

**Positive:**
- Analyzers can be tested in isolation with a single synthetic `EpisodeBatch`.
- Pipeline order does not affect correctness — any permutation of analyzers
  produces the same per-analyzer result.
- Future parallelization (running analyzers concurrently) is safe by
  construction.
- Policy-specific analyzers can be added or removed without touching existing
  analyzers.

**Negative:**
- Some computations that would naturally be shared (e.g. computing a per-episode
  velocity array used by both the smoothness and temporal analyzers) must be
  duplicated per analyzer. The cost is accepted because the datasets are not
  large enough to make this a bottleneck; if it becomes one, precomputation
  can be added to the ingestion layer or as a dedicated preprocessor step
  before the analyzer loop.

---

## References

- `calibra/analyzers/base.py` — `Analyzer` abstract base class
- `calibra/pipeline.py` — `Pipeline.run()`, conditional GR00T attachment
- `calibra/analyzers/gr00t.py` — example policy-specific analyzer
