# Task Structure Metrics

**Confidence: MODERATE for grasp detection. LOW for contact density and trajectory diversity.**

---

## What it measures

**Contact density:** Fraction of steps where the robot is in a "contact or slow
phase" — estimated via gripper state (if available) plus low-velocity envelope.
High contact density means the robot spends most of its time in close interaction
with the object, which is expected for manipulation tasks but abnormal for
navigation or reaching.

**Grasp events per episode:** Number of gripper-close transitions per episode.
Detected via gripper dimension binary state changes (requires gripper dim
auto-detection to succeed).

**Trajectory diversity:** 2-means clustering separation score on PCA-compressed
episode features. Values near 0 mean all trajectories are similar; values near 1
mean clear bimodal strategy clusters. This is a weak signal — it detects
structure, not quality.

---

## Observations from real data

| Dataset | Contact fraction | Grasps/ep | Diversity score |
|---------|-----------------|----------|----------------|
| lerobot/pusht | 21.7% | N/A (no gripper) | 0.46 |
| lerobot/aloha_sim_insertion_human | 90.7% | 1.0 (exact) | 0.59 |

**Grasp detection worked correctly on aloha.** Every one of the 50 episodes
has exactly 1.0 gripper-close events, matching the task (pick up peg, insert).
No false positives, no missed grasps. This validates the gripper auto-detection
and grasp-counting logic on a real bimanual manipulation dataset.

**Contact density reflects task structure, not data quality.** 21.7% for pusht
(pushing task, mostly free motion) versus 90.7% for aloha (insertion task, always
near the object) is a task-type difference, not a quality signal. These numbers
are useful for characterizing what kind of task is in the dataset, not for
flagging problems.

**Trajectory diversity is a weak signal on both datasets.** Scores of 0.46 and
0.59 both fall in the "weak evidence of multiple strategies" bucket. This metric
needs datasets with known multi-modal demonstrations (e.g., a task with two
valid solution paths) to validate whether it actually detects strategy diversity.

---

## Open questions (needs more datasets)

1. **Does grasp detection hold on real hardware datasets?** Sim gripper states are
   binary and clean. Real hardware gripper sensors often have noise or partial-close
   states. BridgeData V2 or DROID would test this.

2. **What does contact density look like for navigation datasets?** Mobile ALOHA
   or any dataset with significant locomotion should show much lower contact
   fractions (< 10%). If it shows > 50%, that would be a task misclassification
   worth investigating.

3. **Can trajectory diversity detect real quality differences?** A dataset where
   half the demonstrations are correct and half are operator errors might show
   two strategy clusters — one for successful approaches, one for failed attempts.
   This would make the diversity metric useful for quality assessment, not just
   characterization.

4. **The short-episode detector is blind to fixed-length datasets.** aloha has
   exactly 500 steps per episode (IQR = 0, lower fence = 500), so no outlier
   detection is possible. This is expected behavior but means the metric is only
   useful on variable-length datasets. Most real robot teleoperation produces
   variable-length episodes — this limitation mainly affects sim datasets.

---

## Known failure modes

**Contact density via velocity envelope is noisy without a gripper.** For datasets
without a gripper dimension (like pusht), contact estimation falls back to
detecting low-velocity phases. This is a proxy, not a direct measurement. It can
confuse resting states, approach phases, and genuine contact.

**Gripper auto-detection can fail on unusual action schemas.** The detector looks
for dimensions where ≥65% of values are near extremes. Some grippers use
proportional control (not binary) and would not be detected. This affects grasp
counting and contact fraction for those datasets.

---

## Datasets profiled

- `references/pusht_velocity_command.json` — no gripper; contact 21.7%, diversity 0.46
- `references/aloha_sim_insertion_human.json` — gripper active; 1.0 grasps/ep, contact 90.7%, diversity 0.59
