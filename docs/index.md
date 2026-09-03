# Calibra

<p align="center">
  <img src="logo.svg" alt="Calibra — train on less robot data, spend less compute" width="480"/>
</p>

**Stop wasting GPU hours on robot data that doesn't improve your policy.**

Calibra audits dataset integrity, measures quality and coverage, and builds quality-aware coresets — so you train on less data, spend less compute, and know exactly why before training begins.

```bash
pip install calibra-robotics
calibra integrity /data/my_demos.h5
calibra compare hf://lerobot/my_dataset aloha
calibra certify /data/my_demos --reference aloha --policy diffusion
calibra prune   /data/100k_episodes --keep 0.3 --out coreset.json
calibra retarget /data/isaac_lab.h5 --out retargeted/
```

---

## The workflow

Calibra is organized around the questions practitioners actually ask about a new dataset, in the order they ask them — not around a list of analyzers.

```mermaid
graph LR
    A["1. Integrity<br/>Can I trust my data?"] --> B["2. Quality<br/>Is it clean?"]
    B --> C["3. Coverage<br/>Is it diverse?"]
    C --> D["4. Optimize<br/>Can I train faster?"]
```

| Step | Question | Command | Docs |
|---|---|---|---|
| 1. Integrity | Can I trust this dataset? | `calibra integrity` | [Integrity Checks](integrity.md) |
| 2. Quality | Is it clean? | `calibra audit` / `calibra score` | [Command Reference](commands.md) |
| 3. Coverage | Is it diverse enough? | `calibra review` | [Command Reference](commands.md) |
| 4. Optimize | Can I train faster/cheaper? | `calibra prune` | [Command Reference](commands.md) |

Integrity checks are deliberately the first thing Calibra runs: timestamp sync, episode completeness, and camera freeze/duplicate-frame detection catch the failures practitioners hit most often, before anything about quality or diversity matters. See [Integrity Checks](integrity.md) for the full list.

**New here?** [Using Calibra](guide.md) walks through all four steps end to end on one dataset, with annotated output and how to read each score.

---

## Why Calibra?

Robot learning labs collect thousands of demonstration episodes. Naively training on all of them:

- ❌ **Silently trains on bad data** — jerk spikes, dropped frames, communication lag, and stuck actuators all look like valid training signals to your policy.
- ❌ **Wastes compute on redundancy** — in a 10,000-episode dataset, 60–80% of episodes are near-duplicates. GPU cost scales with volume, not uniqueness.
- ❌ **Produces undiagnosable failures** — when a policy stalls or flails, you have no idea whether the cause is the architecture, the training recipe, or the data itself.

On LeRobot PushT, Calibra reduced training data by **75%** — retaining 41 of 165 episodes — while staying within 0.5% of full-dataset prediction error and preserving 67% more rare behaviors than random selection.

Calibra achieves this by running deterministic mathematical estimators to flag anomalies and prune redundant episodes before model training ever begins.

---

## Core Features

- **Six Powerful CLI Commands:**
    - `audit`: Diagnose dataset anomalies with bootstrap confidence intervals.
    - `compare`: Evidence-backed comparison against reference baselines.
    - `certify`: Structured pass/fail quality gates for CI/CD pipelines.
    - `prune`: Coreset selection filtering quality failures and maximizing behavioral diversity.
    - `corrupt`: Metric sensitivity verification by injecting synthetic noise.
    - `retarget`: Relative end-effector (EEF) action space conversion.
- **Multiple Formats Supported:** LeRobot v1/v2, HDF5 (Isaac Lab, Robomimic), TF Datasets/RLDS, MCAP/ROS2.
- **Evidence-Backed Assertions:** All analytical findings are linked to the Claims Registry which contains falsification criteria.
