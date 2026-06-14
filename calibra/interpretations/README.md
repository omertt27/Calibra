# Calibra Interpretation Rules

Per-metric documentation of what Calibra's signals mean in practice: what is
validated, what is a hypothesis, and what we don't know yet.

These documents grow as more datasets are profiled. A rule with evidence from
2 datasets is a hypothesis. A rule with evidence from 10+ datasets across multiple
hardware platforms and control modes is a fact.

**Current dataset count: 2** (both simulated)  
**Next priority: real hardware dataset (BridgeData V2, DROID)**

---

## Metric index

| Metric | Document | Confidence | Key finding |
|--------|----------|-----------|-------------|
| Velocity discontinuity rate | [velocity_discontinuity.md](velocity_discontinuity.md) | **HIGH** | Cleanly separates control modes (16.7% velocity-cmd vs 2.4% position-cmd). Threshold correctly calibrated. |
| Jerk spike rate | [jerk_spike_rate.md](jerk_spike_rate.md) | **MODERATE** | Same direction as velocity discontinuity. More frequency-robust than LDLJ. |
| LDLJ | [ldlj.md](ldlj.md) | **LOW (cross-dataset)** | Do NOT compare across control modes or frequencies. aloha scores worse than pusht despite being smoother — frequency/dimensionality artifact. |
| Timestamp jitter | [temporal_stability.md](temporal_stability.md) | **NOT VALIDATED** | Sim datasets are uninformative (machine-precision timestamps). Needs real hardware. |
| Timestamp dropout | [temporal_stability.md](temporal_stability.md) | **NOT VALIDATED** | Same as jitter — zero on all sim datasets. |
| Action/state entropy | [entropy_and_coverage.md](entropy_and_coverage.md) | **LOW** | Both datasets healthy (~5 bits/dim). No failure mode observed yet. |
| PCA variance concentration | [entropy_and_coverage.md](entropy_and_coverage.md) | **LOW** | aloha shows 66.9% in top-2 PCs — task-structure effect, not quality signal. |
| Contact density | [task_structure.md](task_structure.md) | **LOW** | Reflects task type, not quality. pusht 21.7% vs aloha 90.7%. |
| Grasp events | [task_structure.md](task_structure.md) | **MODERATE** | Correct on aloha (1.0/ep, exactly). Needs real hardware validation. |
| Trajectory diversity | [task_structure.md](task_structure.md) | **LOW** | Weak signal on both datasets. Needs known multi-modal dataset to validate. |

---

## Rules that have earned confidence

These rules are supported by at least two datasets with different characteristics
and no dataset-specific hardcoding.

**Rule 1 — Velocity discontinuity separates control modes without configuration.**  
A velocity discontinuity rate > 10% on a position-command dataset is anomalous.
A rate of 10–20% on a velocity-command dataset is expected from human teleop.

**Rule 2 — LDLJ is not a cross-mode or cross-frequency comparator.**  
Do not compare LDLJ values between position-command and velocity-command datasets,
or between datasets with control frequency differences > ~2×. Use velocity
discontinuity rate and jerk spike rate for cross-type comparisons.

**Rule 3 — Sim temporal signals are uninformative.**  
Jitter CV and dropout rate are zero (or near machine-precision) on all simulated
datasets. Do not use sim data to calibrate temporal thresholds.

---

## Rules that are still hypotheses

These are plausible but rest on 2 datasets or on domain knowledge, not measurement.

- Jitter CV thresholds (15%/30%) are calibrated from synthetic fixtures. They may
  fire on normal USB-camera + ROS recording stacks.
- The 20% velocity-change threshold for discontinuity detection may be too strict
  at > 100Hz control frequencies.
- Entropy thresholds assume self-adapting bins are sufficient. Without a fixed
  `action_range`, mode collapse in a narrow subspace will be missed.
- Trajectory diversity score distinguishes strategies. This is unvalidated — no
  known multi-modal dataset has been profiled yet.

---

## What the next dataset should be

Priority: **real hardware with variable-length episodes**.

BridgeData V2 and DROID are the best candidates. Both are:
- Position-command (7-DOF arm)
- Real hardware (validates temporal metrics for the first time)
- Large (100s to 1000s of episodes — stress-tests the groupby pipeline)
- Variable episode length (validates short-episode detection)

A successful real-hardware profile would either confirm the temporal thresholds
or reveal they need recalibration — either outcome is valuable.
