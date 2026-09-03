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
- **ANNOTATE** — redundant enough that pruning drops it. Kept + labelled so a
  metadata-conditioned trainer can still use it.

**Frozen matrix (5 arms):**

| Arm | Training data | Metadata conditioning |
|-----|---------------|-----------------------|
| **A** | Full (all non-DROP episodes) | no |
| **B** | Calibra KEEP only (nominal `--keep`, e.g. 25%) | no |
| **C** | Full (all non-DROP) | yes |
| **R** | Random subset, same episode count as B | no |
| **R+** | Random subset, same episode count as B | yes |

Optional **B+meta** = KEEP-only + metadata (same data as B) — cheap, tests
whether conditioning helps on the aggressive coreset itself. Not required.

**Why not an explicit "Calibra-selected + metadata" arm (decided 2026-09-03):**
under the current `--annotate` pipeline the `ANNOTATE` bucket is *every*
non-KEEP, non-DROP episode, so `KEEP ∪ ANNOTATE == all non-DROP == arm C's
data`. A "keep-what-pruning-drops + condition" arm is therefore **identical to
C**. Rather than break the ADR-011 feature freeze to invent a rescue
threshold before there's any evidence metadata conditioning works, the thesis
is read off **C vs A**: if `B ≪ A` (pruning costs performance) and `C ≈ A`
(metadata on the full set recovers it), the metadata is carrying the signal
that pruning loses. A narrower ANNOTATE bucket is a *follow-up* if C beats A.

**Why R / R+ are in, not optional:** without a random baseline, `B < A` only
shows *less data trains fine*. R (random subset the size of B, no metadata)
isolates whether Calibra's *selection* beats random. R+ checks that metadata
isn't just a free extra input that helps even on a random subset.

**Retention bookkeeping.** A and C train on all non-DROP episodes; B and R/R+
on `--keep` of them. Record **both** the nominal prune target and the actual
training retention (`n_episodes / n_original`) for every arm — they differ
whenever there are DROP episodes. `calibra experiment record` carries this:
`--retention` = nominal, `--actual-retention` = fraction of the original
dataset trained on, plus `--arm` and `--metadata-conditioning`.

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

Arms A / B / R: the conditioning inputs are a fixed zero vector (same shape as
C / R+, so the model code is identical across arms —
`metadata_conditioning_reference.build_conditioning(use_metadata=False)`).

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
  If metadata conditioning helps, it should help most on (a)/(b) — the slices
  whose only training data is the mediocre episodes pruning would drop.

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
| **B ≪ A** and **C ≈ or > A** | metadata on the full set recovers what pruning lost | the characterization is the product — annotate mode becomes primary; a narrower `ANNOTATE` bucket ("prune, then rescue only what's rescuable") is the obvious follow-up |
| **B ≪ A** and **C ≈ B** | metadata does *not* rescue the weak data | the ADR-011 rescue thesis fails — stay a pruning tool |
| **B > R** (at the same size) | Calibra's *selection* beats random | the coreset algorithm is doing real work, not just shrinking the set |
| **B ≈ R** | selection is no better than random at this ratio | the pruning value is "less data trains fine", not smart selection — a credibility problem, surface it |
| **R+ > R** but **C ≈ A** | metadata helps even on random data | conditioning is a generic regularizer, not specific to Calibra's signal — weaker story |
| rare-slice: **C > B** but overall ≈ | metadata's value is concentrated in the long tail | position annotate mode for coverage-critical deployments |

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
