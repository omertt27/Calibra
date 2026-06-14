# Jerk Spike Rate

**Confidence: MODERATE** (2 datasets, same direction as velocity discontinuity,
but n too small to confirm thresholds)

---

## What it measures

Fraction of steps where the L2 norm of jerk exceeds 5× the median jerk magnitude
within the episode. Jerk here is the third derivative of the action signal (or
fewer derivatives if the action is already velocity or acceleration).

Spike detection is relative (multiples of median), not absolute, so it is
scale-invariant. A spike is meaningful relative to the episode's own motion
baseline.

---

## Rules with evidence

**Rule 1: Jerk spike rate separates control modes in the same direction as
velocity discontinuity rate.**

| Dataset | Control mode | Spike rate | Flag |
|---------|-------------|-----------|------|
| lerobot/pusht | velocity (2D) | 4.9% | WARNING |
| lerobot/aloha_sim_insertion_human | joint position (14D) | 0.69% | OK |

7× ratio, same as velocity discontinuity. Both metrics point the same direction.

**Rule 2: The 2% warning threshold is correctly calibrated for position-command
datasets.**

aloha at 0.69% sits comfortably below the 2% warning level, consistent with clean
teleop position commands.

**Rule 3: Jerk spike rate is more frequency-robust than LDLJ.**

Unlike LDLJ, jerk spike rate uses a within-episode relative threshold (5× median).
The median jerk naturally scales with the control frequency, so the ratio is
approximately frequency-invariant. This makes it a better cross-dataset comparator
than LDLJ. (Hypothesis — not yet confirmed with high-frequency vs low-frequency
matched data.)

---

## Open questions (needs more datasets)

1. **Does the 5× multiplier hold across action types?** The spike_k=5 was set from
   synthetic data. Real datasets with bimodal motion (slow approach, fast grasp)
   may have episodes where the fast phase is systematically flagged as "spikes"
   relative to the slow-phase median.

2. **How does it behave on high-speed manipulation?** Datasets like DROID or
   Berkeley Humanoid may have much higher absolute velocities. The relative
   threshold should handle this, but needs validation.

3. **Correlation with task success.** Same question as velocity discontinuity: do
   failed episodes have higher spike rates? This would confirm the metric has
   quality-predictive value beyond describing the control mode.

---

## Known failure modes

**Bimodal motion episodes.** In a task where the robot moves fast to approach and
then slows for fine manipulation, the fast-approach steps could register as "spikes"
relative to the manipulation-phase median. This would inflate spike rates for
tasks with strong speed transitions. Watch for this in pick-and-place datasets.

---

## Datasets profiled

- `references/pusht_velocity_command.json` — velocity-command reference (4.9%)
- `references/aloha_sim_insertion_human.json` — position-command reference (0.69%)
