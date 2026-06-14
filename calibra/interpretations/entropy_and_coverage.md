# Entropy and Coverage Metrics

**Confidence: LOW** (2 datasets, both healthy — no failure mode observed yet)

---

## What it measures

**Action entropy (bits/dim):** Marginal entropy of the action distribution,
computed per-dimension and averaged. High entropy means actions are spread across
the action space; low entropy means the policy is demonstrating a narrow range of
behaviors (mode collapse, repetitive demonstrations).

**State entropy (bits/dim):** Same computation over the observation state. Low
state entropy means the robot is only visiting a small portion of the workspace
in the demonstrations.

**PCA top-2 variance fraction:** Fraction of total action variance explained by
the first two principal components. High concentration (> 80%) means the 
high-dimensional action space is effectively low-dimensional — the demonstrations
are constrained to a near-linear manifold.

---

## Observations from real data

| Dataset | Action entropy | State entropy | PCA top-2 |
|---------|---------------|--------------|-----------|
| lerobot/pusht | 5.30 bits/dim | 5.34 bits/dim | N/A (2D) |
| lerobot/aloha_sim_insertion_human | 4.85 bits/dim | 4.89 bits/dim | 66.9% |

Both datasets show healthy entropy (~5 bits/dim). Neither exhibits the mode
collapse that these metrics are designed to catch.

The aloha PCA result is interesting: 66.9% of 14D action variance lives in 2
principal components. For a bimanual peg insertion task this makes sense — most
of the motion is the final insertion approach, which is nearly linear in joint
space. The tool correctly classifies this as "spread across many PCs" (i.e., not
alarming concentration) because the threshold is 80%. Whether 66.9% is a signal
worth flagging depends on the task structure.

---

## Open questions (needs more datasets)

1. **What does mode collapse actually look like in these numbers?** We have not yet
   seen a real dataset with low entropy. Synthetic fixtures were engineered to
   produce it. A real case — e.g., a dataset where all demonstrations follow a
   single narrow trajectory — would let us validate the threshold and the diagnostic.

2. **Is 5 bits/dim universally "healthy," or does the healthy range vary by task?**
   A contact-rich task with many approach strategies should have higher entropy than
   a task with a single correct solution (e.g., peg insertion). The threshold may
   need to be task-conditioned.

3. **How does the PCA concentration metric behave for tasks with genuine
   low-dimensional structure?** Peg insertion is nearly linear in joint space —
   66.9% in top-2 PCs is expected, not alarming. A screwing task might be even
   more concentrated. The current threshold (80% = alarm) may be too strict for
   high-precision tasks.

4. **Does the entropy metric catch dataset corruption?** If half the demonstrations
   are accidentally duplicated (a real failure mode in large-scale collection), the
   entropy would remain high but the effective diversity would drop. Entropy is not
   sensitive to this class of failure.

---

## Known failure modes

**Self-calibrating bins without a fixed reference range.** The entropy computation
bins the action distribution over its observed range. If the dataset has a narrow
action distribution, the bins adapt to that range and entropy appears high.
`action_range` must be specified explicitly (e.g., `action_range=(-1, 1)` for
normalized joint angles) to detect mode collapse against a reference range.

This is documented in the codebase but easy to miss. Interpretation of entropy
numbers should always note whether a reference range was provided.

---

## Datasets profiled

- `references/pusht_velocity_command.json` — action 5.30 bits/dim, state 5.34 bits/dim
- `references/aloha_sim_insertion_human.json` — action 4.85 bits/dim, state 4.89 bits/dim, PCA top-2 66.9%
