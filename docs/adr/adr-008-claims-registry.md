# ADR-008: Evidence-Backed Falsifiable Claims Registry

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Calibra's diagnostic thresholds (e.g. "flag velocity discontinuity rate > 5%
as WARNING") are not arbitrary. They are derived from empirical observations
about what metric values are typical for different classes of robotics datasets.
Without an explicit record of this empirical basis, thresholds become
unjustifiable magic numbers — and, critically, there is no process for
updating them when new evidence contradicts them.

There is also a risk of overfitting thresholds to a small number of datasets
that happened to be profiled early. A claim system that tracks which datasets
support a threshold creates accountability for the strength of the evidence.

---

## Decision

Calibra maintains a claims registry: a set of JSON files in `calibra/claims/`
where each file contains one or more falsifiable assertions about the
statistical behavior of a metric under a specified class of robotics dataset.

Each claim has:
- A stable, never-reused `id` (e.g. `VD-001`).
- A `metric` key matching a Calibra metric.
- A `class` scope (`velocity`, `position`, `torque`, `sim`, `hardware`, `any`).
- An `assertion` — one specific, falsifiable sentence.
- An `evidence` list — one entry per profiled dataset, recording the observed
  value and whether it supports the assertion.
- A `falsification` block — the explicit condition and threshold that would
  falsify the claim.
- A `status` — `active_hypothesis`, `validated`, `falsified`, or `updated`.

**Confidence is derived, not declared.** `calibra/claims.py` computes
confidence at read time from the count of supporting evidence entries, using
a fixed scale (0 → NOT VALIDATED, 1 → LOW-MODERATE, 2–4 → MODERATE,
5–9 → HIGH, 10+ → STRONG). This prevents contributors from asserting high
confidence without supporting evidence.

**Falsified claims are never deleted.** They remain in the registry with
`status: "falsified"` so the falsification event is traceable. Revised claims
get new IDs.

The contract is specified in `calibra/claims/SPEC.md`.

---

## Alternatives Considered

### Hardcoded thresholds with comments
Embed thresholds directly in analyzer code with a comment explaining the
rationale.
Rejected: thresholds are invisible to the CLI and to downstream consumers;
there is no process for updating them; the comment rot immediately once the
threshold is modified without revisiting the comment.

### Spreadsheet or Google Docs tracking
Track dataset profiling results and threshold derivations in an external
document.
Rejected: not versioned with the code; broken link is a constant risk;
can't be programmatically queried by `calibra compare`.

### Bayesian prior model per metric per class
Build a statistical model P(metric | class) from observed evidence and use
it to generate thresholds and confidence intervals automatically.
Deferred, not rejected: this is explicitly called out in `claims/SPEC.md` as
a future direction once 20+ datasets per class are available. Implementing
distribution modeling with fewer data points would produce "statistical theater,
not statistical insight." The current system is designed to feed into a
Bayesian model when the evidence base is sufficient.

### Remove thresholds entirely; show only raw values
Present raw metric values without any risk classification.
Rejected: raw values are not actionable for users who don't have deep
familiarity with robotics dataset characteristics. Risk levels provide
triage; claims provide the justification for risk levels.

---

## Constraints

- A claim's `metric` field must match a key in `_METRIC_KEYS` in
  `calibra/claims.py`. This is checked by `scripts/generate_claims_doc.py`.
- The claims/references parity rule must be maintained (see ADR-010).
- Claims about causal links (e.g. "metric X predicts training outcome Y")
  enter as `provisional` status, not `active_hypothesis`, because the
  causal mechanism has not been directly measured. The `caution` field
  documents what evidence would be needed to graduate the claim.

---

## Consequences

**Positive:**
- Every diagnostic threshold has a traceable empirical basis.
- `calibra compare` can surface which claims apply to a given dataset class
  and how strong the evidence is.
- When a dataset falsifies a claim, the update process is defined: set status,
  add counter-evidence, write a revised claim with a new ID.
- The evidence base grows systematically as more datasets are profiled.

**Negative:**
- Contributors must follow the claims protocol when adjusting thresholds —
  they cannot simply change a number. This is intentional friction; it prevents
  evidence-free threshold drift.
- The claims/references parity constraint (see ADR-010) means new claims
  cannot be added without first profiling a corresponding dataset.

---

## References

- `calibra/claims/SPEC.md` — full claims contract
- `calibra/claims/*.json` — claim files
- `calibra/claims.py` — programmatic claims interface and confidence derivation
- `scripts/generate_claims_doc.py` — parity check and documentation generation
