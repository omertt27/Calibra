# Metadata-Conditioning Benchmark (ADR-011 research gate)

**Status:** matrix frozen (2026-09-03). Partner-executable. Not yet run.
**Owner question:** *Can conditioning a policy on Calibra's per-episode metadata
improve training outcomes enough to justify keeping data that aggressive
pruning would otherwise remove?*

This is **not** the pruning benchmark. The existing 3-condition protocol
(Full / Random / Calibra, retention curve — see the design-partner protocol)
answers "does Calibra's *selection* beat random?". This one answers "does the
*characterization sidecar* carry usable signal?". Run it only after the
pruning benchmark; the two share infrastructure (`calibra experiment record`)
but not conclusions.

---

## 1. Conditions

Calibra's `--annotate` output over a dataset yields, per episode, a
`calibra_disposition`:

- **KEEP** — in the aggressive coreset (~`--keep` fraction).
- **DROP** — integrity failure; excluded in every arm.
- **ANNOTATE** — redundant enough that pruning drops it, *rescuable* if the
  trainer conditions on the metadata. This bucket is the whole experiment.

**Frozen matrix (6 arms):**

| Arm | Training data | Metadata conditioning |
|-----|---------------|-----------------------|
| **A** | Full (all non-DROP episodes) | no |
| **B** | Calibra KEEP only (nominal 25%) | no |
| **C** | Full (all non-DROP) | yes |
| **D** | Calibra **KEEP ∪ ANNOTATE** (the rescued set) | yes |
| **R** | Random subset, same episode count as D | no |
| **R+** | Random subset, same episode count as D | yes |

Optional **D0** = KEEP-only + metadata (same data as B) if you also want that
point — cheap, but not required.

> **KNOWN ISSUE (found 2026-09-03 while wiring `metadata_conditioning_reference.py`):**
> with the current `--annotate` pipeline, the `ANNOTATE` bucket is *every*
> non-KEEP, non-DROP episode — so **KEEP ∪ ANNOTATE == all non-DROP == arm C's
> data**, and arms **C and D become identical** (same episodes, both with
> metadata). Two ways out, pick one before running:
> 1. **Narrow `ANNOTATE`** to only the diversity-pruned episodes close enough
>    to a KEEP episode to be genuinely rescuable (a threshold on distance /
>    `redundancy`), so KEEP ∪ ANNOTATE ⊊ non-DROP. This is a pipeline change
>    (breaks the ADR-011 feature freeze) but is the more meaningful design.
> 2. **Collapse the matrix to A / B / C / R / R+** and read D's question off
>    C: "does metadata help on the full non-DROP set, and is that gain about
>    selection (C vs R+) or just the extra input?"

**Why R / R+ are in, not optional (approved 2026-09-03):** without a random
baseline, "B < A" only shows *less data trains fine*, and "D > B" is
confounded between *rescuing the right episodes* and *just having more
episodes*. R separates the selection effect from the size effect; R+ checks
that metadata isn't just a free extra input helping on random data.

**Why D = KEEP ∪ ANNOTATE, not "Calibra 25%" (approved 2026-09-03):**
conditioning on the KEEP-only coreset never touches the ANNOTATE bucket, so it
would not test the ADR-011 rescue hypothesis at all.

**D's retention is not 25%.** ANNOTATE episodes raise effective retention
above the `--keep` target. Record **both** the nominal prune target and the
actual training retention for every arm, and never label D "Calibra-25%" in
the writeup — call it "Calibra KEEP∪ANNOTATE (nominal 25%, actual N%)".
`calibra experiment record` carries this: `--retention` = nominal,
`--actual-retention` = fraction of the original dataset trained on,
`--arm D --metadata-conditioning`.

Every arm: **DROP episodes excluded**. `DOWNWEIGHT` rows (none emitted by the
current pipeline) would carry a loss weight.

---

## 2. Datasets

Pick **2 required + 1 optional**, deliberately heterogeneous (a homogeneous
dataset cannot show ANNOTATE carries distinct information — on PushT the
KEEP↔ANNOTATE mean `coverage_value` gap is ≈ 0.03).

| Slot | Recommended | Why |
|------|-------------|-----|
| 1 — heterogeneous real-world | **DROID** (full, or a ≥ 5k-episode slice; `lerobot/droid_100` only for a smoke run) | multi-operator, multi-scene → wide quality + coverage spread |
| 2 — multi-task | a **multi-task ALOHA** collection (union of `lerobot/aloha_static_*`, or a dedicated multi-task LeRobot set) | tests whether conditioning lets a policy use weak demos from under-covered tasks |
| 3 — simpler control *(optional)* | `lerobot/xarm_lift_medium` or `lerobot/pusht` | a floor; also the integration-validated path |

---

## 3. Held constant across all arms

Policy architecture, hyperparameters, optimizer, LR schedule, total training
steps (**not** scaled to dataset size — fix the step budget so arms differ
only in data), train/eval split, eval protocol and rollout count, hardware,
random seeds. Vary **only** the training-set membership and the
metadata-conditioning switch.

Run **× ACT × Diffusion Policy × 3–5 seeds**.

---

## 4. Metadata-conditioning recipe

Model-agnostic sidecar → model-specific input. Keep it minimal for v1.

**Both architectures**
1. Join `calibra_annotations.jsonl` to the episode index on `episode_id`.
2. Compute quartile bins of `quality_risk` and of `coverage_value` **over the
   training set of that arm** (so bins are comparable within-arm; see the
   scripted-data caveat — these columns are absolute, not dataset-normalized).
3. Exclude `calibra_disposition == "DROP"`.

**ACT** — add two learned embeddings (one per binned column, 4 levels each) to
the existing conditioning/style token stream. Train with each episode's real
bins; **at inference pass bin 0** (cleanest / highest-coverage).

**Diffusion Policy** — concatenate a small MLP embedding of the one-hot bins
(optionally plus raw `anomaly_score`) to the global conditioning vector fed to
the denoiser. At inference pass the "clean" bins.

Arms A / B / R: the conditioning inputs are absent (or a fixed zero token, if
the architecture needs a fixed input shape).

---

## 5. Metrics

Per arm × architecture × seed:

- **Overall success rate** (primary), over a fixed rollout count.
- **GPU-hours** and **wall-clock** to the fixed step budget.
- **Convergence speed** — steps to reach 90 % of that arm's final success.
- **Seed variance** — std of success across seeds.
- **Generalization** — success on a held-out task / scene / initial-state
  distribution not seen in training.
- **Rare / under-covered-slice success** *(the decisive metric)*:
  1. Partition the eval set into slices — by task (multi-task set), or by
     initial-state cluster (k-means on start states), or by object/scene
     (DROID).
  2. Rank slices by the mean `coverage_value` of the *training* episodes that
     fall in each slice.
  3. Report: (a) **mean success on the bottom-quartile slices**, (b)
     **worst-slice success**, (c) success-rate spread across slices.
  Metadata conditioning is expected to help most on (a)/(b): that is where
  the rescued ANNOTATE data lives.

Log every field through `calibra experiment record` — it carries `--arm`,
`--metadata-conditioning`, `--retention` (nominal) and `--actual-retention`
(fraction of the original dataset trained on) so the results feed the same
flywheel log as the pruning experiments, machine-comparable.

---

## 6. Decision rules

Read against arm A (full, no metadata):

| Observation | Conclusion | Product implication |
|-------------|------------|---------------------|
| **B ≈ A** | pruning alone already recovers full performance | double down on "train on less data"; annotate mode is a secondary feature |
| **B ≪ A**, **C ≈ A** | metadata recovers what pruning lost, on full data | characterization is the product; pruning is optional |
| **D ≈ or > A** at meaningfully **less compute than A** | rescue + condition beats both | **strongest outcome** — Calibra is a hybrid optimization layer |
| **C ≫ D** | keeping more data + metadata beats aggressive prune + metadata | lean toward characterization over aggressive pruning |
| **D ≈ R+** | the *rescue selection* adds nothing beyond having more data | ANNOTATE bucket is not carrying signal — investigate coverage_value |
| rare-slice success: **D > B** but overall ≈ | metadata's value is concentrated in the long tail | position annotate mode for coverage-critical deployments |

---

## 7. Results template

```
dataset | arch | arm | metadata | nominal_retention_pct | actual_retention_pct |
        seed | success | gpu_hours | wall_clock_s | steps_to_90pct | seed_var |
        generalization_success | rare_slice_bottom_q_success |
        worst_slice_success | slice_spread
```

Aggregate to a per-(dataset, arch) table of arm means ± std, plus the
rare-slice columns broken out separately.

---

## 8. Case-study rights

Same as the design-partner protocol: one-paragraph Design Partner Evaluation
Agreement at kickoff — partner retains data ownership; Calibra may publish
aggregated/anonymized results subject to partner approval of identifying
info. Ask "can we publish the numbers" **before** the run.

---

## 9. Partner-facing framing

> "We've already validated the annotation pipeline end-to-end on a real
> LeRobot v2 dataset (`tests/test_annotate_integration.py`). This experiment
> tests whether the metadata improves learning outcomes — not whether the
> export path works."

---

## Execution order

1. **`calibra experiment record` fields** — `arm`, `metadata_conditioning`,
   `actual_retention_pct`. *(done 2026-09-03)*
2. **Release / tag** annotate mode so the partner installs a version, not `main`.
3. **Partner handoff** — agreement + this protocol.
4. **ACT / Diffusion Policy conditioning shim** (§4) — partner's or in-house.
5. **Run the benchmark.**

Calibra is in **feature freeze around ADR-011** until step 5 returns. The next
architecture decision comes from training results, not more code.
