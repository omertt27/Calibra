# Claim Contract Specification

This document defines what a claim is, how it behaves, and what is required
to add, update, or invalidate one. It is the contract between the claim registry
and reality.

---

## What a claim is

A claim is a falsifiable assertion about the statistical behavior of a metric
under a specified class of robotics dataset. It is not a heuristic, a rule of
thumb, or a best practice. It is a hypothesis with an explicit falsification
condition and a tracked evidence set.

A claim that cannot be falsified is not a claim. It is an assumption — and
assumptions belong in comments, not in the registry.

---

## Anatomy of a claim

```json
{
  "id":         "VD-001",
  "metric":     "velocity_discontinuity_rate",
  "class":      "position",
  "assertion":  "natural language statement of what the claim asserts",
  "conditions": "parameter values or context under which the assertion holds",
  "confidence": "derived from evidence count — see scale below",
  "status":     "one of: active_hypothesis | validated | falsified | updated",
  "evidence": [
    {
      "dataset": "owner/name",
      "observed": 0.024,
      "supports": true,
      "date": "YYYY-MM-DD",
      "notes": "hardware/sim, control freq, action dim, task type"
    }
  ],
  "falsification": {
    "condition": "human-readable description of what would falsify this claim",
    "threshold": 0.08,
    "pending_tests": [
      {
        "dataset": "name or description",
        "reason": "why this dataset is the highest-value next test"
      }
    ]
  },
  "last_updated": "YYYY-MM-DD"
}
```

### Field semantics

**`id`** — unique, stable, never reused. Format: `<PREFIX>-<NNN>`. Prefixes:
`VD` (velocity discontinuity), `JS` (jerk spike), `LDLJ`, `TEMP` (temporal),
`ENT` (entropy), `TS` (task structure), `META` (cross-metric or structural).
When a claim is superseded, the old id is retired — do not reuse it.

**`metric`** — the Calibra metric key this claim is about. Must match a key
in `_claims._METRIC_KEYS` in `calibra/claims.py`. Claims about derived or
composite metrics use the most specific key possible.

**`class`** — the dataset class the claim applies to. Current valid values:
`velocity`, `position`, `torque`, `sim`, `hardware`, `any`. `any` means the
claim is expected to hold regardless of class. A claim scoped to `position`
does not apply to `velocity` datasets and should not be displayed when
comparing against a velocity-command reference.

**`assertion`** — one sentence. Should be specific enough that a reasonable
reader could determine whether a new data point supports or falsifies it.
Bad: "velocity discontinuity is higher for velocity-command datasets."
Good: "velocity-command datasets have velocity discontinuity rate in the range
10–20% under human teleoperation with a 2D joystick-style interface."

**`conditions`** — the parameter context that must hold for the assertion to
be interpretable. Includes metric parameters (e.g. threshold=0.20), action
space assumptions, or control frequency ranges. If the conditions change,
the claim must be updated or a new claim created.

**`confidence`** — derived automatically from evidence count. Do not set this
manually; it is computed by `calibra/claims.py` at read time based on the
number of supporting evidence entries. The derivation scale is:

| Evidence count (supporting) | Confidence       |
|-----------------------------|-----------------|
| 0                           | NOT VALIDATED   |
| 1                           | LOW-MODERATE    |
| 2–4                         | MODERATE        |
| 5–9                         | HIGH            |
| 10+                         | STRONG          |

Counter-evidence (supporting: false) does not add to the count — it triggers
a status review. A single counter-evidence entry at any n should not lower
confidence automatically; it should open a review.

**`status`** — the current epistemic state of the claim. Valid transitions:

```
active_hypothesis
  ├─→ validated      (n >= 5, consistent direction, no counter-evidence)
  ├─→ falsified      (a falsification condition was met)
  └─→ updated        (scope narrowed or threshold revised due to partial evidence)

validated
  ├─→ falsified      (counter-evidence found even after validation)
  └─→ updated        (scope or threshold revision needed)

falsified
  └─→ superseded     (replaced by a new, narrower claim — record the new id)

updated
  └─→ active_hypothesis  (after revision, the claim restarts its evidence cycle)
```

A falsified claim is never deleted. It stays in the registry with status
`falsified` so the falsification event is traceable. If the failure was
dataset-specific rather than general, scope the revised claim more narrowly
and create a new id.

---

## What counts as evidence

A dataset is evidence for a claim if:

1. It has been profiled through Calibra's full Phase 1 pipeline.
2. The observed metric value is available (not None/n/a).
3. The dataset belongs to the claim's class (or class is `any`).
4. The metric was computed under the same `conditions` as the claim specifies.

A dataset **supports** a claim if the observed value is consistent with the
assertion. For numeric range claims (e.g. "rate in range 10–20%"), supporting
means the value falls within the stated range. For directional claims (e.g.
"rate < 5%"), supporting means the value satisfies the inequality.

A dataset **falsifies** a claim if it meets the explicit falsification
condition in `falsification.threshold` or `falsification.condition`. A value
outside the claimed range on one dataset is not automatically a falsification
— it triggers a review. A pattern across multiple datasets is.

Simulated datasets count as evidence for claims about simulation (class: `sim`)
or claims with class: `any`. They do not count as evidence for claims about
real hardware (class: `hardware`).

---

## What falsifies a claim

Falsification conditions must be specified numerically when possible. "Any
position-command hardware dataset with rate > 0.08" is better than "a dataset
with unexpectedly high rates." The threshold should be derived from the
assertion: if the claim is "rate < 5%", the falsification threshold should be
a value meaningfully outside that range (e.g. > 8%, to allow for measurement
noise), not just 5.01%.

When a claim is falsified, the responsible action is:
1. Set `status` to `"falsified"`.
2. Add the counter-evidence entry to the evidence list with `supports: false`.
3. Write a new claim (new id) with a revised assertion that accounts for
   the failure — either narrower scope, revised threshold, or explicit
   exception for the class that falsified it.
4. Update the `compare.py` interpretation function that depended on the claim.

---

## Class taxonomy

Classes are currently flat (not hierarchical). Two orthogonal axes exist:

**Control mode**: `velocity`, `position`, `torque`, `unknown`
**Dataset origin**: `sim`, `hardware`, `unknown`

The current single `class` field encodes only control mode. Claims about
simulation behavior use class `sim` (encoded as a special case). This will
evolve to a two-field schema (`class` + `origin`) when the number of
hardware-specific claims justifies it. Until then, the notes field in each
evidence entry should record whether the dataset is sim or hardware.

---

## Dependency model (deferred)

Claims will eventually depend on other claims. Example: a claim about LDLJ
comparability depends on a claim about control mode classification being
reliable. When one claim is falsified, dependent claims may become uncertain.

This dependency graph is not yet implemented. For now, dependencies are
recorded informally in the `conditions` field ("assumes control mode
classification is correct"). When the dependency graph becomes necessary
(estimated: 15–20 claims with known interactions), it will be implemented
as an explicit `depends_on` list in the claim schema.

Do not implement this prematurely. The cost of adding a field later is low.
The cost of designing a dependency system without knowing the actual dependency
structure is high.

---

## Contribution protocol

### Adding a new claim

1. Identify the metric, class, and assertion from the profiling output.
2. Check whether an existing claim already covers the same metric + class.
   If it does, add evidence to the existing claim rather than creating a new one.
3. Write the falsification condition before writing the assertion. If you
   cannot write a specific falsification condition, the claim is not ready.
4. Set `status` to `active_hypothesis`. Do not set it to `validated` until
   n >= 5 with consistent direction.
5. Add at least one entry to `pending_tests` — the dataset that would most
   reduce uncertainty.

### Running a new dataset

1. Run `scripts/profile_pusht.py --dataset <id> --out calibra/references/<name>.json`.
2. Open every claim whose `pending_tests` includes this dataset or dataset class.
3. For each claim: compare the observed value against the `falsification.threshold`.
   If within the supported range, add a `supports: true` evidence entry.
   If outside the supported range, add a `supports: false` entry and open a review.
4. Update `last_updated` on any claim you touched.
5. Re-run `calibra compare` against an existing reference to verify that evidence
   lines updated correctly.

---

## Anti-patterns

**Not a claim: a threshold**
"Warning fires when rate > 5%" is a threshold. The claim that supports it
is "position-command datasets have rate < 5%" with evidence. The threshold
is derived from the claim, not the other way around.

**Not a claim: a heuristic**
"Datasets from academic labs tend to be cleaner" is not falsifiable in the
claim sense because "cleaner" is undefined and "tend to" has no threshold.

**Not a claim: a universal**
"All robotics datasets have low entropy" cannot be falsified by a single
counter-example. Claims must be scoped to a class and an observable range.

**Not a claim: a feature request**
"The system should warn when LDLJ is below -15" is a design decision, not
a claim. The claim that would justify this design decision is "LDLJ below
-15 correlates with policy failure on [task class]" — which requires evidence.

---

## What this spec does not cover

- Probabilistic scoring (combining claims into a dataset health score)
- Distribution modeling (fitting P(metric | class) from evidence)
- Unified Bayesian updating

These are deferred until the claim graph is stable and the evidence base is
large enough that distribution inference is meaningful (estimated: 20+
datasets per class). Building these abstractions before that threshold is
reached will produce statistical theater, not statistical insight.
