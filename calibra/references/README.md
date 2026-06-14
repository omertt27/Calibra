# Calibra Reference Profiles

Real-dataset baselines for interpreting Calibra diagnostic output. Each profile
is a raw metric distribution — no thresholds applied — produced by running
`scripts/profile_pusht.py` against a publicly available dataset.

Use these as anchors when reading a new dataset's Calibra report. A signal that
looks alarming in isolation may be normal for its dataset type; one that looks
fine may be surprisingly clean compared to peers.

---

## Cross-dataset comparison

| Metric | pusht (velocity cmd, sim) | aloha insertion (position cmd, sim) | aloha mobile cabinet (position cmd, **hardware**) | Verdict |
|--------|--------------------------|-------------------------------------|--------------------------------------------------|---------|
| Episodes / Steps | 206 / 25,650 | 50 / 25,000 | 85 / 127,500 | — |
| Episode length (steps) | mean 125, std 36 | exactly 500 | exactly 1,500 | — |
| Control frequency | ~10 Hz | 50 Hz | 50 Hz | — |
| Action dim | 2 | 14 | 14 | — |
| Jitter CV | 2.9e-6 | 1.1e-5 | 3.1e-5 | Sim and hardware similar here |
| Dropout | 0.0% | 0.0% | 0.0% | — |
| **LDLJ (mean)** | **−16.34** | **−20.43** | **−24.08** | All CRITICAL — see note |
| **Jerk spike rate** | **4.9% WARNING** | **0.69% OK** | **1.0% OK** | Position cmd consistently cleaner |
| **Velocity disc. rate** | **16.7% CRITICAL** | **2.4% WARNING** | **1.3% OK** | Clean separation by control mode |
| Action entropy (bits/dim) | 5.30 | 4.85 | 4.67 | All healthy |
| Contact fraction | 21.7% | 90.7% | 78.9% | Task-type difference |
| Grasps per episode | none (no gripper) | 1.0 | 4.0 | Task structure detected correctly |
| **Episode outliers (Calibra)** | **27 / 206** | **n/a (fixed length)** | **8 / 85** | Aggregate-invisible corruption |

### Verdict: Outcome 1 — velocity discontinuity threshold is correctly calibrated

The velocity discontinuity rate separates cleanly by action space semantics:

- **pusht (velocity commands): 16.7% → CRITICAL**. Human teleoperation of direct
  velocity inputs allows instantaneous reversals — the robot has no physical
  inertia constraint at the command level. Frequent direction changes are a
  structural property of this control mode, not a data quality failure.

- **aloha (joint positions): 2.4% → WARNING**. Position-command joint targets are
  physically bounded by the previous position — large velocity changes require
  large position deltas, which teleoperators avoid naturally. The 2.4% rate is
  low but not zero (some sharp approach-to-grasp transitions).

The 7× ratio between these two numbers (16.7% vs 2.4%) under identical threshold
settings confirms the metric is sensitive to a real and meaningful difference.
Calibra correctly distinguishes dataset types without any dataset-type input from
the user. This is the behavior that makes the tool useful before Phase 3
policy-conditioning is built.

### The unexpected finding: LDLJ is not directly comparable across dataset types

Both pusht and aloha score CRITICAL on LDLJ (−16.3 and −20.4 respectively), with
aloha scoring *worse* despite using position commands. The LDLJ formula is
theoretically dimensionless — it normalizes by T³ and v_max² — but in practice it
is sensitive to control frequency and action space dimensionality in ways the
normalization does not fully cancel. At 50 Hz over 500 steps, numerical
differentiation produces high jerk values from small positional oscillations; at
10 Hz over ~125 steps, the same physical jerk manifests as a different LDLJ.

**Consequence:** LDLJ is a reliable within-type signal (comparing two 50Hz
arm datasets, or two velocity-command datasets) but should not be compared across
control modes or frequencies. Do not use the absolute LDLJ value to compare
pusht-style and aloha-style datasets. Use velocity discontinuity rate and jerk
spike rate for cross-type comparisons.

This is a known limitation to address in Phase 3: the policy-conditioning layer
should suppress or recalibrate LDLJ reporting when comparing datasets with
incompatible control frequencies.

---

## [`pusht_velocity_command.json`](pusht_velocity_command.json)

**Dataset:** `lerobot/pusht`  
**Task:** Push a T-shaped block to a target pose using a 2D velocity-command interface (teleop).  
**Episodes:** 206 | **Steps:** 25,650 | **Action space:** 2D velocity (dx, dy) — no gripper  
**Origin:** Simulated (Chi et al., 2023 — Diffusion Policy paper)

### Key numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Timestamp jitter CV | 2.86e-6 | Near machine-precision — sim timestamps are exact |
| Dropout rate | 0.0% | Simulated, no dropped frames |
| LDLJ (mean) | −16.34 | Characteristic of human velocity-command teleop; not comparable to position-cmd datasets |
| Jerk spike rate | 4.9% | At edge of 5% critical threshold |
| Velocity discontinuity rate | 16.7% | Structural — p50=16.3%, p95=27.5%; velocity-command artifact |
| Action entropy | 5.30 bits/dim | Healthy coverage of 2D velocity space |
| Contact fraction | 21.7% | Steps in slow/contact phase (velocity envelope proxy, no gripper) |

### What this profile tells you

**Temporal metrics are not informative from sim data.** Jitter and dropout only
come alive on real hardware. Do not use pusht to calibrate temporal thresholds.

**16.7% velocity discontinuity is the velocity-command floor.** Any velocity-command
dataset below 10% is meaningfully smoother than this reference. Any dataset above
25% is outlying even for this control mode.

**LDLJ of −16.34 is the velocity-command reference.** Interpret LDLJ only within
datasets of the same control mode and frequency.

---

## [`aloha_sim_insertion_human.json`](aloha_sim_insertion_human.json)

**Dataset:** `lerobot/aloha_sim_insertion_human`  
**Task:** Bimanual peg insertion (sim), 2×7-DOF joint position control (teleop via ALOHA hardware).  
**Episodes:** 50 | **Steps:** 25,000 | **Action space:** 14D joint positions (7 per arm) + gripper  
**Origin:** Simulated (Zhao et al., 2023 — ACT paper)

### Key numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Timestamp jitter CV | 1.1e-5 | Sim — not informative |
| Dropout rate | 0.0% | Sim — not informative |
| LDLJ (mean) | −20.43 | Worse than pusht despite position control — frequency/dim sensitivity; not cross-comparable |
| Jerk spike rate | 0.69% | Well within OK (< 2% threshold); clean position trajectories |
| Velocity discontinuity rate | 2.4% | Position-command floor; contrast with pusht's 16.7% |
| Action entropy | 4.85 bits/dim | Healthy but slightly lower than pusht — structured task |
| PCA top-2 variance | 66.9% | 14D space concentrates in ~2 effective DOFs — expected for insertion |
| Contact fraction | 90.7% | Most steps near the insertion site — high-contact task |
| Grasps per episode | exactly 1.0 | Perfect grasp detection; every demo picks up the peg once |

### What this profile tells you

**Episode structure is perfectly uniform.** All 50 episodes are exactly 500 steps.
This means the short-episode outlier detector produces trivial output (IQR = 0).
The tool correctly avoids false positives but also cannot detect length anomalies
in fixed-length datasets. Expected behavior.

**Grasp detection works on aloha.** Every episode has exactly one gripper-close
event, matching the task (pick up peg, insert). This validates the task structure
analyzer on a real manipulation dataset.

**2.4% velocity discontinuity is the position-command floor.** Any position-command
arm dataset above ~8% warrants investigation. Below 2% is unusually smooth.

**LDLJ is not the right cross-dataset comparator.** Use jerk spike rate and velocity
discontinuity rate for comparisons involving datasets with different control modes
or frequencies.

---

## [`aloha_mobile_cabinet.json`](aloha_mobile_cabinet.json)

**Dataset:** `lerobot/aloha_mobile_cabinet`  
**Task:** Mobile ALOHA bimanual robot opening a cabinet door.  
**Episodes:** 85 | **Steps:** 127,500 | **Action space:** 14D joint positions (7 per arm), 50 Hz  
**Origin:** Real hardware

### Key numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Timestamp jitter CV | 3.1e-5 | Low but above machine-precision — hardware timing is clean |
| Dropout rate | 0.0% | No dropped frames |
| LDLJ (mean) | −24.08 | Worse than sim aloha — expected on real hardware with oscillation noise |
| Jerk spike rate | 1.0% | OK (< 2% threshold); position control is smooth |
| Velocity discontinuity rate | **1.3%** | **Second real-data point supporting VD-001** (position cmd < 5%) |
| Action entropy | 4.67 bits/dim | Healthy |
| Contact fraction | 78.9% | Contact-rich task — consistent with cabinet manipulation |
| Grasps per episode | 4.0 | 4 gripper events per episode (door handle grasp, pull open, two more) |
| **Episode outliers** | **8 / 85** | **Invisible to aggregate metrics** |

### The documented real catch

`aloha_mobile_cabinet` passes every aggregate check — velocity discontinuity OK,
jitter OK, spike rate OK, entropy OK. A user inspecting only the summary report
would conclude this is a clean dataset.

Calibra's episode-level anomaly detection finds **8 of 85 episodes** are outliers
within this dataset's own distribution:

```
ep_2          vel_disc_rate 3.0× MAD    → start of dataset (equipment warm-up)
ep_27         vel_disc_rate 3.3× MAD    → investigate before training
ep_35         vel_disc_rate 3.3× MAD    → investigate before training
ep_46         vel_disc_rate 3.0× MAD    → investigate before training
ep_54         vel_disc_rate 4.8× MAD    → investigate before training  ← worst
ep_68         vel_disc_rate 3.3× MAD    → investigate before training
ep_74         spike_rate    3.3× MAD    → investigate before training
ep_81         vel_disc_rate 3.8× MAD    → end of dataset (fatigue / drift)
```

**ep_54** deviates 4.8× MAD from the per-episode median on velocity discontinuity
rate. The aggregate vel_disc_rate is 1.29% (well within the 2% OK threshold), which
means this outlier episode contributes noise that the average absorbs. A policy
trained on this dataset would see spurious high-discontinuity targets from ep_54
with no warning from any aggregate diagnostic.

**The start/end pattern** (ep_2 and ep_81) is consistent with equipment warm-up
artifacts at session start and operator fatigue or hardware drift at session end.
This is not identifiable from aggregate metrics.

**To remove flagged episodes:**
```python
from calibra.comparison.curator import EpisodeCurator
curator = EpisodeCurator(strategy="remove")
```

### What this profile tells you

**Position-command hardware datasets are not always clean.** The aloha_sim datasets
(perfect timing, no hardware noise) are not representative of real hardware. This
dataset shows real but subtle episode-level variance that simulation hides.

**VD-001 holds on real hardware.** Aggregate vel_disc_rate of 1.29% confirms
the < 5% position-command claim on a second, independent dataset. The per-episode
outliers don't change the aggregate verdict — they reveal local corruption.

**Temporal metrics are still uninformative.** Jitter CV 3.1e-5 is still very low
despite being real hardware. This dataset may use locked-timestep playback.
Need BridgeData V2 or DROID to see real hardware timing noise.
