# ADR-010: Claims/References Parity Rule

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Calibra's claims registry (see ADR-008) allows contributors to add
falsifiable assertions about metric behavior. Without a constraint on the
ratio of claims to evidence, it is easy to accumulate many plausible-sounding
claims that are not yet backed by actual dataset profiles. This creates a
false impression of empirical grounding and makes the claims registry a
collection of opinions rather than observations.

The tension is: claims are cheap to write; reference profiles require running
a full Phase 1 pipeline over a real dataset, which takes time and disk space.
Left unconstrained, the number of claims will naturally grow faster than
the evidence base.

---

## Decision

The number of **active** claims (those with `status` equal to
`active_hypothesis`, `validated`, or `updated`) must not exceed the number
of reference profiles in `calibra/references/`.

This rule is enforced programmatically by `scripts/generate_claims_doc.py --check`,
which exits with a non-zero status if the parity constraint is violated. This
check is intended to run in CI on every pull request that modifies files
in `calibra/claims/` or `calibra/references/`.

`provisional` claims (directional hypotheses without direct measurement)
are counted separately and are not subject to the parity constraint. However,
they may not be cited as evidence for diagnostic thresholds until they graduate
to `active_hypothesis` status.

`falsified` claims do not count toward the active claim total.

The practical implication for contributors: **before adding a new active claim,
profile a dataset and add its reference JSON.** Theory should not outpace
evidence.

---

## Alternatives Considered

### No parity constraint; trust contributor judgment
Leave the claims/references balance to individual discretion.
Rejected: the history of standards bodies, scientific literature, and
internal company knowledge bases consistently shows that undisciplined
accumulation of unvalidated assertions creates technical debt that is
hard to repay. An automated check is a low-cost way to enforce discipline.

### Minimum evidence per claim (e.g. every claim must have ≥ 1 evidence entry)
Require at least one evidence entry on every active claim, regardless of
total counts.
Considered: this is stricter than parity and was rejected as too burdensome
for claims that are directionally supported by existing references but haven't
yet been formally annotated. The parity rule is coarser but easier to
satisfy incrementally.

### Parity enforced at claim-addition time by tooling (not CI)
Block claim addition at the CLI/script level rather than in CI.
Rejected: enforcement at the commit/PR boundary is the right place; local
enforcement can be bypassed and doesn't protect the main branch.

---

## Constraints

- The check counts only JSON files in `calibra/references/` with a `.json`
  extension as reference profiles. Files that are not valid reference profiles
  (e.g. schema documents) must not be placed in that directory.
- `provisional` claims must include a `caution` field explaining what evidence
  is needed to graduate them to `active_hypothesis`.

---

## Consequences

**Positive:**
- The claims registry cannot grow faster than the evidence base without a
  deliberate decision to violate the parity rule and fix the check.
- Contributors are incentivized to profile datasets before adding claims,
  which expands the reference library as a side effect.
- The ratio of claims to references is visible at a glance from the CI check
  output.

**Negative:**
- The constraint can feel obstructive when a contributor has strong empirical
  intuition about a metric but hasn't yet profiled a dataset formally.
  The `provisional` status is the release valve: it allows recording the
  directional hypothesis without claiming empirical support.
- The parity check counts file totals, not per-class totals. A registry with
  15 `position`-class claims and 15 `velocity`-class references technically
  passes parity even though the position claims have no supporting evidence.
  This is a known limitation; finer-grained parity (per class) is deferred
  until the evidence base is large enough to make per-class tracking meaningful.

---

## References

- `calibra/claims/SPEC.md` — full claims contribution protocol
- `scripts/generate_claims_doc.py` — parity check implementation
- `calibra/references/` — reference profile JSON files
- `calibra/claims/*.json` — claim files
