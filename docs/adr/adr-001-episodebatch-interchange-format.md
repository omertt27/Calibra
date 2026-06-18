# ADR-001: EpisodeBatch as the Universal Ingestion-to-Analysis Interchange Format

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Calibra must support multiple robotics dataset formats (LeRobot, RLDS, HDF5,
MCAP, Isaac Lab) and multiple independent diagnostic analyzers. Without a
shared intermediate representation, each analyzer would need to understand
every format, creating an N×M coupling matrix that makes adding either a new
format or a new analyzer expensive and error-prone.

The two halves of the system have different concerns:

- **Ingestion layer**: knows how files are laid out on disk, how timestamps are
  encoded, which columns are images vs. scalars, where episode boundaries fall.
- **Analysis layer**: knows nothing about file formats; it works purely on
  time-series arrays of actions and observations.

---

## Decision

All format-specific adapters must produce an `EpisodeBatch` and nothing else.
All analyzers must accept an `EpisodeBatch` and nothing else.

`EpisodeBatch` is the single seam between the two halves of the system. No
analyzer ever imports an adapter; no adapter ever imports an analyzer.

The schema is defined in `calibra/schema/episode.py` and contains:

- `Episode`: timestamps, observations dict, actions, per-modality timestamps,
  metadata.
- `EpisodeBatch`: a list of `Episode` objects plus dataset-level metadata
  (name, format string, source path).

Arrays are typed as `np.ndarray`. No format-specific types (HDF5 groups,
TensorFlow tensors, Arrow tables) may appear in the schema.

---

## Alternatives Considered

### Adapter-per-analyzer coupling
Each analyzer receives a format-specific object and extracts what it needs.
Rejected: creates N×M coupling; adding one analyzer requires updating every
adapter; adding one format requires updating every analyzer.

### Streaming iterator instead of a materialized batch
Yield one episode at a time rather than loading all episodes into memory.
Rejected at this stage: several analyzers need cross-episode statistics
(entropy, PCA variance, episode length distribution) that require the full
batch to be in memory at once. A streaming interface can be added later when
memory becomes a constraint on large datasets.

### Dataset-library-native types (HuggingFace Dataset, tf.data.Dataset)
Use the native dataset type as the interchange, allowing format-specific
optimizations inside analyzers.
Rejected: couples analyzers to specific libraries, making the library a
transitive dependency of every analyzer and preventing format-specific
dependencies from being optional.

---

## Consequences

**Positive:**
- Adding a new adapter requires no changes to any analyzer.
- Adding a new analyzer requires no changes to any adapter.
- Analyzers can be unit-tested against synthetic `EpisodeBatch` objects without
  any format dependency installed.
- The schema is a stable API surface; breaking it requires an explicit decision.

**Negative:**
- Very large image arrays are loaded into RAM because the adapter must
  materialize them as `np.ndarray`. This is mitigated for the LeRobot v2
  adapter by the DuckDB fast path (see ADR-006), which skips image columns
  before populating the batch.
- Some format-specific metadata that doesn't fit the schema is lost. The
  `extra` dict on both `Episode` and `EpisodeBatch` provides an escape hatch
  for format-specific metadata without polluting the core schema.

---

## References

- `calibra/schema/episode.py` — EpisodeBatch and Episode definitions
- `calibra/ingestion/base.py` — DatasetReader abstract base class
- `calibra/analyzers/base.py` — Analyzer abstract base class
