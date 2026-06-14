# Temporal Stability (Jitter and Dropout)

**Confidence: NOT YET VALIDATED on real hardware. Sim datasets are uninformative.**

---

## What it measures

**Jitter (timestamp_jitter_cv):** Coefficient of variation of inter-step time
deltas. Measures how consistent the control loop frequency is across the dataset.
High jitter means the robot's control loop was running at an irregular rate during
data collection.

**Dropout (timestamp_dropout_rate):** Fraction of steps where the inter-step gap
exceeds 3× the median step duration. These are likely dropped ticks, missed frames,
or recording gaps.

---

## The key finding: sim datasets are uninformative

Both datasets profiled so far are simulated. Timestamps in sim are exact:

| Dataset | Jitter CV | Dropout |
|---------|----------|---------|
| lerobot/pusht (sim) | 2.86e-6 | 0.0% |
| lerobot/aloha_sim_insertion_human (sim) | 1.1e-5 | 0.0% |

These values are near machine-precision — they tell us the sim produces perfect
timestamps, not that the thresholds are correctly calibrated. No real hardware has
been profiled yet.

---

## Rules with evidence

None that can be stated with confidence from sim data alone.

**Placeholder thresholds (from synthetic fixtures, not real data):**
- Jitter CV warning: 15%, critical: 30%
- Dropout warning: 1%, critical: 5%

These were designed by engineering synthetic defects. They may be too tight or too
loose for real hardware and should be treated as hypotheses until validated.

---

## What real hardware will likely show

Based on robotics system knowledge (not measurement):

- **USB cameras at 30Hz:** Jitter CVs of 5–20% are common from USB scheduling
  latency. The 15% warning threshold may fire frequently on otherwise-fine datasets.
- **ROS topic recording:** Missed publications and callback delays create real
  dropout patterns. 1% dropout in a 30Hz dataset means roughly 3 missed frames
  per 10-second episode — noticeable but common.
- **Synchronized hardware (e.g. EtherCAT):** Jitter CV should be < 1% and dropout
  near 0%. Anything above this warrants investigation on synchronized systems.

These are predictions. They need to be checked against BridgeData V2, DROID, or
any dataset collected from real hardware.

---

## Open questions (needs real hardware datasets)

1. **What is the jitter CV distribution for real USB camera + ROS setups?** This
   is the most common real-world robotics recording stack. If typical jitter CVs
   are 10–20%, the 15% warning threshold will fire on normal data — requiring
   recalibration.

2. **Does jitter CV correlate with training degradation?** The hypothesis is that
   high jitter impairs time-series policies (Diffusion Policy, temporal
   transformers). This needs empirical validation with training results.

3. **Are dropouts clustered or random?** A single 50-frame dropout at the start of
   an episode (recording artifact) is different from 50 individual 1-frame dropouts
   throughout (hardware instability). The current metric does not distinguish these.
   A dropout-cluster metric may be more informative.

---

## Priority for next profiling run

**High.** This is the most important unvalidated metric cluster. Any dataset
collected from real hardware (BridgeData V2, DROID) will immediately reveal whether
the thresholds are reasonable.

---

## Datasets profiled

- `references/pusht_velocity_command.json` — sim, uninformative (jitter 2.86e-6, dropout 0%)
- `references/aloha_sim_insertion_human.json` — sim, uninformative (jitter 1.1e-5, dropout 0%)
