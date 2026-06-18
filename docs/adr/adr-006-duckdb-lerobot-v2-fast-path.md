# ADR-006: DuckDB Fast Path for LeRobot v2 Parquet Scanning

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

LeRobot v2 datasets are stored as Parquet shards with a well-defined schema.
Columns include:

- Scalar/vector columns: `timestamp`, `action`, `observation.state`,
  `episode_index`, `frame_index`
- Image columns: `observation.images.<camera>` — these contain encoded image
  bytes and can be very large (gigabytes on a typical dataset)

The HuggingFace `datasets` library (the natural reader for LeRobot) loads
all columns into memory by default. For Calibra's use case — analyzing
temporal structure, action smoothness, and state coverage — image data is
never needed. Loading it wastes RAM and adds significant latency.

---

## Decision

The LeRobot v2 adapter uses DuckDB as a fast path to scan Parquet files
directly, skipping image feature columns before any data is loaded into memory.

The fast path is activated when:
1. The dataset is stored locally in v2 format (presence of `meta/info.json`).
2. `duckdb` is importable (included in `calibra[lerobot]` extras).

DuckDB reads only the non-image columns from the Parquet shards. It identifies
image columns by checking the `features` metadata in `meta/info.json` for
columns whose type describes an encoded image.

The HuggingFace `datasets` library remains available as a fallback for:
- Hub download paths (`hf://` URIs or repo IDs like `lerobot/pusht`)
- LeRobot v1 format (uses `dataset_dict.json` instead of `meta/info.json`)
- Cases where DuckDB is not importable

In fallback mode, image columns are filtered from the resulting Arrow table
before conversion to numpy, to limit peak RAM use even when the full table
must be loaded.

---

## Alternatives Considered

### Always use `datasets` library, filter images post-load
Load the full dataset with `datasets`, then drop image columns from the
resulting in-memory table.
Rejected: the image data is loaded into RAM before it can be dropped;
on large datasets this can exhaust available memory before the filter runs.

### Implement a custom Parquet reader without DuckDB
Use `pyarrow.parquet.read_table(..., columns=[...])` directly to select
only non-image columns.
Considered as an alternative to DuckDB: `pyarrow` is already a dependency
of `calibra[lerobot]`. However, DuckDB's columnar scan engine handles Parquet
shard globbing and predicate pushdown more ergonomically, and `duckdb` is
already in the dependency set. If DuckDB becomes a liability, migrating to
`pyarrow.parquet` directly is the natural fallback.

### Require users to pre-filter datasets before running Calibra
Document that image columns should be removed before analysis.
Rejected: unacceptable UX; users should not need to preprocess their datasets
to run a diagnostic tool.

---

## Constraints

- The DuckDB fast path is only available for local v2 datasets. Hub datasets
  must go through the `datasets` library (which handles auth, caching, and
  streaming from the Hub).
- The fast path must produce byte-for-byte equivalent `EpisodeBatch` objects
  to the fallback path, except for ordering within episodes (which is
  normalized by sort on `episode_index`, `frame_index`).
- `duckdb` must be imported lazily inside the adapter; it must not be a
  top-level import that would fail when `calibra[lerobot]` is not installed.

---

## Consequences

**Positive:**
- Typical RAM usage for analysis of large LeRobot v2 datasets drops from
  O(full dataset size) to O(non-image columns only), often a 10×–100× reduction.
- Analysis latency is significantly lower for disk-local v2 datasets.

**Negative:**
- Two code paths (DuckDB fast path + `datasets` fallback) must be maintained
  and kept equivalent.
- `duckdb` is an additional optional dependency. Users who notice the fast
  path being skipped because DuckDB is not installed may be confused.
  The adapter logs a debug message when falling back.

---

## References

- `calibra/ingestion/adapters/lerobot.py` — fast path implementation
- `pyproject.toml` — `duckdb>=0.9` in `calibra[lerobot]` extras
