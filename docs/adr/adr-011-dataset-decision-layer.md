# ADR-011: Dataset Decision Layer — Analysis Outputs Support Both Pruning and Metadata Annotation

**Status:** Accepted  
**Date:** 2026-09-03  
**Deciders:** Calibra core team

---

## Context

Calibra's headline feature is coreset selection: `calibra prune` produces a
smaller training set by removing bad and redundant episodes ("up to 75% less
training data"). Internally the product has been modeled as **Analyze →
Prune**, and the output schema reflects that — `CurationReport`
(`calibra/schema/comparison.py`) records a binary partition:
`retained_indices` vs. `dropped_indices`.

That binary model bakes in an assumption: that an episode Calibra does not
select for the coreset has no training value. Recent work in robot learning
on heterogeneous data — metadata-conditioned policies (ACT, Diffusion Policy,
VLA-style models) trained with per-episode quality / length / success signals
as conditioning inputs — shows this assumption does not always hold. When the
trainer can condition on metadata, mediocre data retained *with* an honest
quality annotation can raise performance rather than lower it, and dataset
diversity matters more than per-episode polish for compositional
generalization.

Calibra already computes everything such a trainer would want to condition on
— `calibra_score`, integrity flags, success/failure, episode length,
coverage/novelty contribution, redundancy, quality risk (`calibra/assessment.py`'s
`EpisodeAssessment`, `calibra/pruning.py`, the integrity analyzers). It
discards most of it at the last step by collapsing to keep/drop.

This is an architectural product decision, not a future feature: it changes
the shape of Calibra's primary output.

---

## Decision

### 1. Reframe the internal product model: `Analyze → Characterize → Decide`

"Prune" is one consumption strategy over the analysis, not Calibra's identity.
Schema, code organization, and documentation reflect a decision layer whose
output is a per-episode disposition, not a keep/drop list.

### 2. Every episode carries a full `characterization`, independent of disposition

The characterization bundle is the union of signals Calibra already computes:
`calibra_score`, integrity flags, `success` / `failure`, episode length,
`anomaly_score`, `quality_risk`, `coverage_value` / novelty contribution,
redundancy. It is computed for all episodes; the decision step consumes it
but does not gate its production.

### 3. Each episode gets a `disposition` from a closed enum

| Disposition  | Meaning |
|--------------|---------|
| `KEEP`       | In the coreset as-is. Default when no signal fires. |
| `DROP`       | Exclude — integrity failure, or pure redundancy with no coverage value. |
| `DOWNWEIGHT` | Include with a reduced sample weight (the weight is an attribute of the disposition). |
| `ANNOTATE`   | Include; emit characterization as trainer-facing conditioning metadata. |
| `REVIEW`     | Human inspection before a decision — unusual but not clearly bad (the existing review-queue posture). |
| `RECOLLECT`  | Reserved. Data-acquisition guidance — not emitted until that work lands (roadmap 2027–28). |

### 4. Two consumption modes over one analysis pass

- **Prune mode** (current headline, unchanged): materialize the `KEEP` set →
  smaller dataset on disk. Serves vanilla BC/ACT/Diffusion pipelines that
  cannot exploit metadata. Positioned as: *"When your trainer can't use
  metadata, Calibra finds the smallest useful subset."*
- **Annotate mode** (new): emit a LeRobot-compatible sidecar carrying each
  episode's `disposition` + `characterization`, plus documented recipes for
  turning those fields into conditioning inputs for ACT / Diffusion Policy /
  VLA training. Does not modify episode data.

### 5. Schema change

`CurationReport` gains a per-episode `disposition` and a structured
`characterization`. `retained_indices` / `dropped_indices` become derived
views (`KEEP` vs. not-`KEEP`), kept for backward compatibility with a
deprecation note.

### 6. The "up to 75% less training data" claim is retained but qualified

In technical material it reads *"on validated non-conditioned policy
benchmarks"* (see [ADR-008](adr-008-claims-registry.md), `docs/claims.md`).
It is a training-efficiency result on measured benchmarks — not an
information-theoretic claim that removed data is worthless.

### 7. Gating

Promoting annotate mode to a co-headline is contingent on a 4-arm benchmark:
full / prune-25% / full + metadata conditioning / prune-25% + metadata
conditioning, across ACT and Diffusion Policy, measuring success rate,
GPU-hours, convergence speed, generalization, and rare-behavior retention.
That experiment is a research-roadmap item tracked separately and is **not**
part of this ADR's acceptance. This ADR accepts only the output *architecture*
(decision layer + disposition enum + dual consumption modes), which is worth
building regardless of which mode wins.

---

## Alternatives Considered

### Keep the binary prune-only model; add annotation as an unrelated command
Ship `calibra annotate` as its own pipeline alongside `calibra prune`.
Rejected: duplicates the analysis pass; two code paths over the same signals
drift apart; no shared audit trail; forces users to pick a lane before seeing
the characterization.

### Pivot fully to "keep everything + metadata", drop pruning
Rejected: premature. No internal benchmark yet supports it; the GPU-cost
value of pruning is real and validated for non-conditioned policies (the
majority of current LeRobot-ecosystem training); abandoning a validated,
differentiated result on an external-literature bet is the wrong risk
trade-off. Item 7's benchmark exists precisely so this call is not made blind.

### Free-form tags instead of a closed disposition enum
Rejected: downstream tooling (exporters, `calibra gate`, CI) needs a small
stable set to switch on. Same reasoning as `RiskLevel` being a closed `str`
enum in [ADR-003](adr-003-pydantic-output-schemas.md).

### A single continuous per-episode weight, no categorical disposition
Rejected: `DROP` vs. `DOWNWEIGHT` vs. `RECOLLECT` are qualitatively different
downstream actions, not points on one scale. A continuous weight is meaningful
*within* `DOWNWEIGHT` and is modeled as an attribute of it.

---

## Constraints

- `Disposition` is a closed `str` enum in `calibra/schema/`. Adding a value is
  an explicit decision (per [ADR-003](adr-003-pydantic-output-schemas.md)).
- `characterization` fields must be JSON-serializable scalars/lists; numpy
  stays in `raw_metrics` (per [ADR-003](adr-003-pydantic-output-schemas.md)).
- The annotate-mode sidecar must round-trip through LeRobot metadata
  conventions without mutating episode data.
- `RECOLLECT` is reserved now and only emitted once acquisition-guidance work
  lands. Every consumer must tolerate an unrecognized disposition without
  erroring.
- Default disposition when no signal fires is `KEEP` — existing `calibra
  prune` behavior is preserved exactly.
- Default CLI output collapses to the familiar keep/drop summary unless
  annotate mode is requested, so the common case is not confronted with a
  six-way breakdown.

---

## Consequences

**Positive:**
- One analysis pass feeds both prune and annotate; a single unified audit
  trail.
- Hedges the metadata-conditioning trend instead of betting against it —
  Calibra stays useful whether the field settles on "smaller sets" or
  "annotated sets."
- The disposition model extends cleanly to the planned roadmap:
  training-aware weighting (2027–28) is `DOWNWEIGHT` with computed weights;
  acquisition guidance (2028–29) is `RECOLLECT` with target descriptions.
- "Prune" becomes a strategy name, not the product ceiling — consistent with
  the observability-layer repositioning.

**Negative:**
- The `CurationReport` schema change is breaking for any consumer reading
  `retained_indices` / `dropped_indices` directly. Mitigated by keeping them
  as derived properties through a deprecation window.
- More output surface to test and document; the "Decide" step introduces
  threshold → disposition policy config that users must understand.
- Risk of overwhelming users with a six-way output. Mitigated by the
  collapse-to-keep/drop default in the constraints above.

---

## References

- `calibra/schema/comparison.py` — `CurationReport`, `EpisodeFlag` (schema change target)
- `calibra/assessment.py` — `EpisodeAssessment` (characterization backbone: anomaly / quality_risk / coverage_value)
- `calibra/pruning.py`, `calibra/curation/` — current prune path
- `calibra/integrity.py` — integrity flags feeding characterization
- [ADR-003](adr-003-pydantic-output-schemas.md) — Pydantic output schemas, closed-enum convention
- [ADR-008](adr-008-claims-registry.md) — claims registry (the "75% less data" claim qualification)
- [ADR-009](adr-009-two-phase-pipeline.md) — two-phase pipeline (the analysis boundary this builds on)
