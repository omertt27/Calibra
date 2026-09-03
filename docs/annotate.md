# Annotate mode

`calibra prune` normally produces a smaller dataset by removing episodes.
**Annotate mode** (`--annotate DIR`) keeps every episode instead, and writes a
per-episode sidecar recording what Calibra would have done and why. A training
pipeline that can condition on metadata can then use the weaker episodes
rather than discarding them.

```bash
calibra prune /data/my_demos --keep 0.3 --annotate ./calibra_meta/
```

This writes three files into `./calibra_meta/`:

| File | Contents |
|---|---|
| `calibra_annotations.jsonl` | One row per episode — the sidecar you join to your dataset. |
| `calibra_annotations.manifest.json` | Schema version, source dataset, disposition counts, and a field dictionary. |
| `calibra_curation_report.json` | The raw `CurationReport`, for tooling and debugging. |

Add `--annotate-format parquet` (or `both`) to also write
`calibra_annotations.parquet` — the same rows, columnar, with an explicit
typed schema so all-null columns keep a real type. This needs pyarrow
(`pip install 'calibra-robotics[lerobot]'`).

The default `calibra prune` output (`--out`, `--report`, `--export-dataset`)
is unchanged; `--annotate` is additive.

## The sidecar schema

`calibra_annotations.jsonl`, one JSON object per line. Model-agnostic — no
ACT / Diffusion / VLA fields live here.

| Column | Type | Meaning |
|---|---|---|
| `episode_index` | int | Position in the source dataset (0-based). Join key. |
| `episode_id` | str | Stable episode identifier from the source dataset. Join key. |
| `calibra_disposition` | str | `KEEP` / `DROP` / `DOWNWEIGHT` / `ANNOTATE` / `REVIEW` / `RECOLLECT` — see below. |
| `calibra_score` | float \| null | 0–100 per-episode cleanliness = `100·(1 − quality_risk)`. Quality-risk only — see note below. |
| `quality_risk` | float \| null | 0–1. Higher = more likely a real recording/execution problem. |
| `coverage_value` | float \| null | 0–1. Higher = more unique behavioral coverage this episode adds. `null` without `InfluenceAnalyzer`. |
| `anomaly_score` | float \| null | 0–1. Higher = more statistically unusual on some metric (not itself bad). Weak signal on datasets with fewer than ~100 episodes. |
| `redundancy` | float \| null | 0–1 = `1 − coverage_value`. Complement of `coverage_value`, not an independent feature — **not** pairwise duplicate detection. |
| `success` | bool \| null | Episode success flag from the source dataset metadata, if present. |
| `integrity_flags` | list[str] | Metric names that failed an integrity/quality threshold. |
| `n_steps` | int \| null | Episode length in timesteps (a dataset fact, not a Calibra signal). |
| `weight` | float \| null | Training sample weight. `null` → treat as `1.0`. Set only for `DOWNWEIGHT` rows. |

`schema_version` is `1.1.0`. A field rename or a change in meaning bumps it.

!!! warning "`quality_risk` / `calibra_score` are absolute, not dataset-normalized"
    These are **absolute** quality signals — fixed thresholds on jerk,
    velocity discontinuity, dropout and LDLJ — not scores normalized to the
    dataset they came from. On **scripted or planner-generated** datasets
    (e.g. PushT, ALOHA `*_scripted`), where waypoint transitions produce high
    jerk by construction, every episode can score a mediocre `quality_risk`
    (and a low `calibra_score`) even when the data is perfectly healthy
    *relative to that dataset*. Read these columns as "how clean vs. an
    absolute bar", and compare episodes to each other within the file rather
    than to 0/100. (`calibra prune`'s Stage-1 thresholds auto-relax when they
    detect a scripted motion signature; the `quality_risk` **column** does
    not.)

!!! note "`calibra_score` is not the dataset Calibra Score"
    The per-episode `calibra_score` here is `100·(1 − quality_risk)` — a
    quality-risk-only number. The dataset-level **Calibra Score**
    (`calibra score`) is a quality-gated blend of temporal, smoothness,
    coverage and task-structure dimensions. Averaging the `calibra_score`
    column does **not** reproduce it.

## Dispositions

| Disposition | What a vanilla trainer does | What a metadata-conditioned trainer does |
|---|---|---|
| `KEEP` | Train on it. | Train on it. |
| `DROP` | Exclude it. | Exclude it. |
| `ANNOTATE` | *(would exclude — see below)* | **Keep it**, conditioned on the columns above. |
| `DOWNWEIGHT` | Train on it. | Train on it at `weight`. |
| `REVIEW` | — | Needs a human decision first. |
| `RECOLLECT` | — | Reserved. |

`ANNOTATE` is the **rescue disposition**: an episode redundant enough that
`calibra prune` would normally drop it, kept because its characterization
gives the trainer enough context to use it safely. `KEEP` and `DROP` mean the
same thing in both modes; only the boundary episodes change. So `ANNOTATE` /
`DOWNWEIGHT` are always a minority of rows — if most of your dataset is
`ANNOTATE`, lower `--keep`.

## Using the sidecar in training

The recipe is model-specific and lives above this schema, not in it. The
general shape:

1. **Join** the sidecar to your episode index on `episode_id` (or
   `episode_index`).
2. **Filter** out `calibra_disposition == "DROP"`.
3. **Condition** on the numeric columns. Two common approaches:
   - *Bucketed tokens* — bin `quality_risk` / `coverage_value` into a few
     levels and prepend a learned embedding per level (ACT, Diffusion Policy).
     Robust, cheap, works with a small dataset.
   - *Continuous features* — concatenate the raw scores to the policy's
     conditioning vector (VLA-style models with a text/goal encoder).
4. **Weight** the loss by `weight` where present (`DOWNWEIGHT` rows), treating
   `null` as `1.0`.

At inference, pass the "best" bucket / a `quality_risk` of `0` so the policy
imitates the clean demonstrations.

## Choosing a dataset to evaluate this on

Whether the metadata *helps training* is an empirical question, and the
dataset matters:

- **Heterogeneous / real-world** — DROID, and other multi-operator,
  multi-scene collections. Wide quality and coverage spread, so `ANNOTATE`
  and `DOWNWEIGHT` carry real signal.
- **Multi-task** — multi-task ALOHA, multi-environment LeRobot datasets.
  Tests whether conditioning helps the policy use lower-quality data from
  tasks it has few clean demos of.
- **Simpler control** (optional) — a single-task set as a floor.

**PushT and other scripted single-task datasets are for integration
validation only.** They are near-homogeneous (≈0.03 spread between `KEEP` and
`ANNOTATE` mean `coverage_value` in practice), so they cannot tell you whether
`ANNOTATE` episodes carry information a pruned run would lose — the whole
point of the experiment. Use them to confirm the export path works, not to
measure its value.

When you do run the comparison (full vs. Calibra-pruned, each with and without
metadata conditioning), track **performance on rare / under-covered slices**
alongside overall success rate and GPU-hours — that is where conditioning on
retained-but-mediocre data has the best chance of beating pure pruning.

## API

```python
from calibra.schema.annotations import AnnotationManifest

manifest = AnnotationManifest.load("./calibra_meta/")   # reads jsonl + manifest
for row in manifest.annotations:
    ...
```

`calibra.annotate.build_annotation_manifest(curation_report, ...)` builds a
manifest from any `CurationReport`;
`AnnotationManifest.write(dir, parquet=True)` emits the parquet file.
`calibra.pruning.pruning_result_to_curation_report(result, batch, report=...,
redundant_disposition=Disposition.ANNOTATE)` — also reachable as
`result.to_curation_report(batch, ...)` — is what `--annotate` calls; the same
`CurationReport` type is what `EpisodeCurator.curate()` returns.
