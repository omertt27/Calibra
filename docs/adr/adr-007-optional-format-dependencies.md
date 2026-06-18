# ADR-007: Optional Format Dependencies with Lazy Imports

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Calibra supports five dataset formats, each requiring a different set of
heavy third-party dependencies:

| Format | Dependencies |
|--------|-------------|
| LeRobot | `datasets`, `pyarrow`, `duckdb` |
| RLDS | `tensorflow`, `tensorflow-datasets` |
| HDF5 | `h5py` |
| MCAP | `mcap`, `mcap-ros2-support` |
| Isaac Lab | _(subset of above)_ |

`tensorflow` alone is ~500 MB. Requiring all dependencies for all formats
would make Calibra impractical for users who only work with, for example,
LeRobot datasets and do not have a TensorFlow-compatible environment.

Calibra's core value (analyzers, reports, comparison) has no dependency on
any of these format libraries. Only the ingestion adapters depend on them.

---

## Decision

Format-specific dependencies are declared as optional extras in
`pyproject.toml`:

```
calibra[hdf5]     → h5py
calibra[lerobot]  → datasets, pyarrow, duckdb
calibra[rlds]     → tensorflow, tensorflow-datasets
calibra[mcap]     → mcap, mcap-ros2-support
calibra[all]      → all of the above
```

Inside each adapter, format-specific imports are performed lazily inside
helper functions (e.g. `_require_datasets()`, `_require_duckdb()`) that
raise `ImportError` with a clear installation hint if the dependency is
missing. These helpers are called at the top of `read()`, not at module
import time.

The top-level `calibra` package (`calibra/__init__.py`) does not import any
adapter. Adapters are only imported when `calibra.ingestion.registry` is
imported (e.g. when `load()` is called). This means that importing
`calibra.pipeline` or `calibra.schema` does not trigger any format-specific
import.

---

## Alternatives Considered

### Single required dependency set
Declare all format dependencies as required.
Rejected: makes `pip install calibra-robotics` fail in environments without
TensorFlow, CUDA-related libraries, or ROS 2. The robotics ecosystem is
fragmented enough that no single environment has all tools available.

### Separate packages per format (e.g. `calibra-lerobot`, `calibra-rlds`)
Split each adapter into its own installable package.
Rejected: complicates the import structure, versioning, and documentation;
users need to know to install `calibra-lerobot` in addition to `calibra`.
The extras model is the standard Python approach for this pattern.

### Dynamic import of adapters via `importlib` at runtime
Discover adapter modules by scanning the `adapters/` directory and importing
each with a try/except around the format dependencies.
Rejected: makes adapter registration implicit and difficult to audit; any
module-level syntax error in an adapter becomes a silent registration failure
rather than an import error. The current approach — `can_read` registers
even with missing dependencies, `read` raises `ImportError` — is more explicit.

---

## Constraints

- The `_require_<lib>()` pattern must be used consistently. A bare
  `import duckdb` at module level in an adapter is a bug — it makes the
  adapter unimportable when the dependency is absent, which silently removes
  it from the registry.
- Error messages from `_require_<lib>()` must include the exact
  `pip install` command needed.
- `TYPE_CHECKING` guards (`if TYPE_CHECKING: import ...`) may be used for
  type annotations without triggering runtime imports.

---

## Consequences

**Positive:**
- `pip install calibra-robotics` (bare) installs in any Python 3.10+
  environment, with only `pydantic` and `numpy` as hard dependencies.
- Users who work exclusively with one format install only what they need.
- Adapter unit tests can mock the format library and verify that the
  `ImportError` message is correct.

**Negative:**
- Contributors must remember the lazy import pattern. A module-level import
  of a format library in an adapter will not be caught by tests unless the
  CI environment lacks that dependency. The pattern is documented in the
  adapter authoring guide and enforced by code review.

---

## References

- `pyproject.toml` — optional extras definitions
- `calibra/ingestion/adapters/lerobot.py` — `_require_datasets()`,
  `_require_duckdb()` pattern
- `calibra/ingestion/registry.py` — import-time registration
