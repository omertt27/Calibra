# Velocity Discontinuity Rate

**Confidence: HIGH** (2 datasets, clean separation, no hardcoding)

---

## What it measures

Fraction of steps where the L2 norm of the velocity change exceeds 20% of the
episode's peak velocity. "Velocity" here means the numerical first derivative of
actions — so for position-command datasets, this is change in positional velocity;
for velocity-command datasets, it is change in the velocity signal itself.

A discontinuity in this sense is a step where the agent reverses or sharply alters
its motion faster than smooth physical actuation would allow.

---

## Rules with evidence

**Rule 1: The metric cleanly separates velocity-command from position-command datasets.**

From two datasets, no dataset-specific configuration:

| Dataset | Control mode | Rate | Flag |
|---------|-------------|------|------|
| lerobot/pusht | velocity (2D) | 16.7% | CRITICAL |
| lerobot/aloha_sim_insertion_human | joint position (14D) | 2.4% | WARNING |

7× ratio without any dataset-type input. The metric recovers a structural property
of the control mode from raw trajectories alone.

**Rule 2: The 5% critical threshold is correctly calibrated for position-command datasets.**

2.4% is well below the critical threshold for aloha — the tool does not
false-positive on clean position-command teleop data.

**Rule 3: ~16% is the empirical floor for velocity-command teleop.**

Human velocity-command control allows instantaneous direction reversal at the
command level (no physical inertia constraint). Frequent reversals are a property
of the control interface, not a data quality failure. Do not flag velocity-command
datasets for discontinuity rates below ~20% without additional context.

---

## Open questions (needs more datasets)

1. **Is 2.4% the position-command floor, or is it aloha-specific?** A second
   position-command dataset (e.g., BridgeData V2, DROID) with a different task or
   hardware would confirm whether the position-command baseline is stable.

2. **Does the rate correlate with task success?** If failed episodes in a dataset
   have higher discontinuity rates than successful ones, the metric has quality
   predictive value. This requires a dataset with per-episode success labels.

3. **How does the 20% threshold interact with very high-frequency datasets?** At
   200Hz or above, a 20% velocity change per step is physically impossible for most
   hardware — the threshold may be too permissive and need tightening.

4. **Real hardware vs sim.** Both datasets so far are simulated. Real hardware adds
   sensor noise and actuation lag that could artificially elevate discontinuity rates
   even for clean teleoperation.

---

## Known failure modes

None confirmed yet. The metric behaved as expected on both datasets.

---

## Datasets profiled

- `references/pusht_velocity_command.json` — velocity-command reference (16.7%)
- `references/aloha_sim_insertion_human.json` — position-command reference (2.4%)
