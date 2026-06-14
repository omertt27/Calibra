# LDLJ (Logarithmic Dimensionless Jerk)

**Confidence: LOW for cross-dataset use. UNKNOWN for within-type use.**

---

## What it measures

LDLJ = −log((T³ / v_max²) × ∫||jerk||² dt)

A smoothness metric from motor control literature. Higher (less negative) is
smoother. Designed to be dimensionless by normalizing by duration and peak velocity.

Typical values from the literature on arm motion: −3 to −7 for smooth movements,
below −10 for movements with significant discontinuities.

---

## The key finding

**LDLJ should not be compared across datasets with different control modes or
significantly different control frequencies.**

Evidence from two datasets:

| Dataset | Control mode | Freq | Action dim | LDLJ (mean) | Flag |
|---------|-------------|------|-----------|-------------|------|
| lerobot/pusht | velocity (2D) | ~10Hz | 2 | −16.34 | CRITICAL |
| lerobot/aloha_sim_insertion_human | joint position (14D) | 50Hz | 14 | −20.43 | CRITICAL |

aloha scores *worse* than pusht despite being position-controlled and physically
smoother. This is counterintuitive and reveals a limitation of the metric at scale.

### Why this happens

The LDLJ formula is dimensionless in units but sensitive to two confounders that
the normalization does not fully cancel:

**Control frequency.** At 50Hz, numerical differentiation (position → velocity →
acceleration → jerk) amplifies small positional oscillations. A smooth arm motion
at 50Hz produces larger jerk values than the same motion at 10Hz because each
derivative step is divided by a smaller dt. The T³ normalization partially compensates
but does not fully absorb frequency differences.

**Action space dimensionality.** The jerk integral accumulates L2 norm over all
action dimensions. For a 14D arm where multiple joints are active simultaneously,
the integral is larger than for a 2D velocity command even if per-axis motion is
smoother. The v_max² normalization compensates partially but depends on which
dimension dominates the L2 norm.

---

## Rules with evidence

**Rule 1: LDLJ is unreliable for cross-control-mode comparison.**
Confirmed on 2 datasets. Do not compare LDLJ between position-command and
velocity-command datasets.

**Rule 2: LDLJ is unreliable for cross-frequency comparison.**
Hypothesis (not yet confirmed with a third dataset, but strongly suggested by the
aloha result). Do not compare LDLJ between datasets with control frequency
differences > ~2×.

**Rule 3: LDLJ may be valid for within-type comparison.**
If two datasets share the same control mode and similar frequency, LDLJ should
produce comparable values. This is unvalidated — we do not yet have two
within-type datasets to compare.

---

## Open questions (needs more datasets)

1. **Does LDLJ separate quality within a control type?** If we have a good and a
   degraded version of the same type of dataset (e.g., two pusht collections with
   different teleop quality), does LDLJ capture the difference? This is the most
   important open question.

2. **Is there a frequency-normalized LDLJ variant that restores cross-dataset
   comparability?** The formula could potentially be modified to normalize by dt
   in the jerk integral, making it frequency-invariant. Needs derivation and
   testing.

3. **What is the typical LDLJ range for real hardware?** Both datasets are simulated.
   Real hardware adds sensor noise that may change the baseline significantly.

---

## Recommendation for Phase 3 (policy-conditioning)

When a user specifies a policy family, the policy-conditioning layer should:
- Suppress the LDLJ flag when comparing datasets across control modes or frequencies
- Substitute jerk spike rate and velocity discontinuity rate as the primary smoothness
  signals in cross-type contexts
- Reserve LDLJ for within-type comparisons or for flagging extreme outliers (e.g.,
  episodes with LDLJ < −25 within a position-command dataset)

---

## Datasets profiled

- `references/pusht_velocity_command.json` — velocity-command, ~10Hz (LDLJ −16.34)
- `references/aloha_sim_insertion_human.json` — position-command, 50Hz (LDLJ −20.43)
