# ADR-005: Configuration-Driven Observation Key Normalization Layer

**Status:** Accepted  
**Date:** 2024-01-01  
**Deciders:** Calibra core team

---

## Context

Different robotics dataset formats use incompatible naming conventions for
logically identical signals. For example:

| Format | RGB camera key | Proprioception key |
|--------|---------------|-------------------|
| LeRobot v2 | `observation.images.top` | `observation.state` |
| Isaac Lab | `obs/camera_rgb` | `obs/proprio_state` |
| Robomimic/HDF5 | `agentview_image` | `robot0_eef_pos` |
| Custom HDF5 | `observation.image.camera1` | `observation.state.joint_positions` |

If adapters pass raw keys through to `EpisodeBatch`, every analyzer must
handle all naming variants. Alternatively, each adapter could normalize
independently, but that scatters the mapping logic and produces inconsistent
canonical names across adapters.

---

## Decision

A dedicated normalization layer is applied at the end of every adapter's
`read()` call before the `EpisodeBatch` is returned. This layer is
implemented in `calibra/schema/normalization.py` and exposed as the
`normalize_obs_keys(observations, extra_mapping=None)` function.

The normalization logic applies three rules in order:

1. **Exact match**: look up the raw key in the combined mapping dict.
2. **Prefix-strip rules**: strip a known prefix (e.g. `images.` → `camera_`)
   and re-check the mapping.
3. **Pass-through**: if no rule matches, return the key unchanged.

The default mapping (`_DEFAULT_MAPPING`) and prefix rules (`_PREFIX_RULES`)
are defined as module-level constants in `calibra/schema/normalization.py`,
making them the single authoritative source. Adapters may supply an
`extra_mapping` override for dataset-specific keys that the default mapping
does not cover.

Only observation keys are normalized. Action column names are treated as
opaque numeric vectors — they are never renamed.

---

## Alternatives Considered

### Per-adapter normalization
Each adapter normalizes its own keys to canonical names independently.
Rejected: the canonical name for the same signal may differ across adapters
if there is no shared reference; any update to canonical naming requires
finding and updating every adapter.

### Analyzer-side normalization
Analyzers that need a specific modality probe for known variants
(`if "camera_top" in obs or "images.top" in obs`).
Rejected: every analyzer must re-implement the same heuristic; adding support
for a new format requires modifying every analyzer that uses visual data.

### Leaving keys unnormalized; using a modality detector
Detect modality type from array shape or dtype rather than key name
(e.g. arrays with 3 channels of uint8 are cameras).
Rejected: fragile for scalar/vector observations where shape is not
diagnostic; also makes test construction harder since test batches must
contain correctly-shaped arrays.

### YAML-file-only configuration
Store the entire mapping in `calibra/core/mappings.yaml` and load it at runtime.
Partially adopted: `calibra/core/mappings.yaml` exists for other normalization
metadata, but the observation key mapping is kept in Python for IDE
discoverability, static analysis, and import-time availability without I/O.

---

## Constraints

- Canonical observation key names follow the pattern `<modality>_<descriptor>`,
  using only lowercase letters, digits, and underscores. No dots, slashes, or
  spaces in canonical names.
- Canonical modality prefixes: `camera_`, `joint_`, `eef_`, `gripper_`,
  `state` (undifferentiated proprio), `proprio`.
- If two raw keys resolve to the same canonical key, the last one wins with a
  `warnings.warn`. Callers must resolve the ambiguity via `extra_mapping`.

---

## Consequences

**Positive:**
- Analyzers use a single consistent vocabulary regardless of source format.
- The complete key mapping is readable in one file; auditing it for coverage
  takes seconds.
- Adding support for a new format's naming convention requires only an addition
  to `_DEFAULT_MAPPING`, with no analyzer changes.
- `extra_mapping` provides a per-adapter escape hatch without modifying shared
  configuration.

**Negative:**
- Pass-through for unrecognized keys is silent; an analyzer that relies on a
  canonical key it expects will get an empty observations dict for that key.
  The correct debugging step (checking what keys are actually present) is
  not immediately obvious. This is a known rough edge documented in the adapter
  authoring guide.

---

## References

- `calibra/schema/normalization.py` — `normalize_obs_keys`, `_DEFAULT_MAPPING`,
  `_PREFIX_RULES`
- `calibra/core/mappings.yaml` — supplementary normalization metadata
- `calibra/ingestion/adapters/lerobot.py` — example of `normalize_obs_keys`
  usage inside an adapter
