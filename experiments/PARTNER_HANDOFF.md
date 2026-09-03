# Design-Partner Handoff — Metadata-Conditioning Evaluation

Everything a design partner needs to run the ADR-011 research gate. Pair this
with `METADATA_CONDITIONING_BENCHMARK.md` (the protocol) and
`metadata_conditioning_reference.py` (the wiring).

---

## The offer (what to say)

> We'll run Calibra on one of your existing robot datasets, free, during our
> design-partner phase. Calibra produces a per-episode metadata sidecar. We'd
> like to measure — on your training stack — whether conditioning an ACT /
> Diffusion Policy on that metadata lets you keep demonstrations that
> aggressive pruning would otherwise drop, without hurting policy performance.
>
> The annotation pipeline is already validated end-to-end on a real LeRobot v2
> dataset (`tests/test_annotate_integration.py`). This experiment tests
> whether the metadata improves learning outcomes — not whether the export
> path works.

Do **not** pitch "try Calibra." Pitch this specific experiment.

---

## What each side does

**Calibra provides**
- `calibra prune --annotate DIR` run on the partner's dataset → the sidecar
  (`calibra_annotations.jsonl` + manifest + parquet).
- The frozen 5-arm protocol and the reference conditioning code.
- Analysis + writeup of the results.

**Partner provides**
- One existing dataset (ideally in an under-covered cell of the diversity
  matrix — see below).
- Their training + eval stack for ACT and Diffusion Policy.
- Compute for the matrix: 5 arms × 2 architectures × 3–5 seeds × N datasets.
  A reduced first pass (arms A + B + C, one architecture, one dataset, 2
  seeds) is a valid go/no-go before committing the full matrix.
- Sign-off on the evaluation agreement below **before** the run.

---

## Design Partner Evaluation Agreement (lock at kickoff)

> Calibra analyzes the Partner's dataset solely to evaluate Calibra's
> technology. The Partner retains all ownership of its data and trained
> models. Calibra's analysis outputs are shared back with the Partner.
> Calibra may publish aggregated, anonymized results of this evaluation
> (e.g. data-retention %, compute reduction, policy-performance comparison,
> rare-slice performance), subject to the Partner's prior approval of any
> identifying information. Either party may end the evaluation at any time;
> on termination Calibra deletes the Partner's data and derived artifacts.

Ask explicitly at kickoff: **"can we publish the numbers?"** Don't discover
after a six-week run that nothing can be cited.

---

## Dataset choice

Prefer whichever fills the **emptiest cell** of the diversity matrix
(embodiment × task × policy × scale × environment × data quality), not
whichever is easiest to get. Two similar partners running the same
embodiment/task/policy add little; one partner in an uncovered cell is worth
far more to the evidence base.

For this experiment specifically: **heterogeneous beats clean.** A dataset
with real quality and coverage spread (multi-operator, multi-scene,
multi-task) is required — on a near-homogeneous set the `ANNOTATE` bucket
carries almost no distinct information and the experiment can't answer its own
question. DROID and multi-task ALOHA are the reference choices.

---

## Kickoff checklist

- [ ] Evaluation agreement signed; publishing rights confirmed.
- [ ] Dataset selected; confirmed heterogeneous (check the `KEEP` vs
      `ANNOTATE` mean `coverage_value` gap in the manifest — if < ~0.05,
      pick another).
- [ ] `calibra prune --annotate` run; sidecar delivered.
- [ ] Partner confirms ACT + Diffusion Policy both train on their stack with a
      **fixed step budget** (not scaled to dataset size).
- [ ] Conditioning wiring adapted from `metadata_conditioning_reference.py`
      and unit-checked (bins computed on the training set; inference passes
      the clean bin).
- [ ] `run_metadata_benchmark.py` pointed at the partner's `train_and_eval`
      callable.
- [ ] First pass: arms A + B + C, one arch, 2 seeds → go/no-go on "does
      metadata recover what pruning loses?"
- [ ] Full matrix.
- [ ] Results logged via `calibra experiment record` (`--arm`,
      `--metadata-conditioning`, `--retention`, `--actual-retention`).
- [ ] Writeup against the decision-rule table.
