---
title: "Calibra: Catch Bad Robot Data Before You Waste GPU Time"
thumbnail: /blog/assets/calibra/thumbnail.png
authors:
  - user: omerTT
---

# Calibra: Catch Bad Robot Data Before You Waste GPU Time

Robot learning labs spend weeks collecting demonstration data and days training
policies — only to discover mid-evaluation that the dataset had jerk spikes,
dropped frames, and missing language annotations the whole time.

**[Calibra](https://github.com/omerTT/Calibra)** is an open-source dataset
observability tool for robotics imitation learning. It runs deterministic
mathematical estimators (no LLMs, no heuristics) over your episodes and tells
you exactly what is wrong — and which episodes to remove — before training starts.

```bash
pip install calibra-robotics
calibra certify /data/my_demos.h5 --policy gr00t
```

---

## The Problem

When you collect 1,000 robot demonstrations, you typically don't look at them.
You feed them directly to a training script and hope the policy learns.

In practice, a typical dataset contains:

- **Jerk spikes** — communication lag or abrupt operator corrections create
  large velocity discontinuities that look like valid training signal to a
  diffusion policy but are actually noise.
- **Timestamp dropout** — Isaac Sim drops physics steps under load, producing
  duplicate timestamps that corrupt temporal metrics and sequence models.
- **Redundancy** — in a 10,000-episode dataset, 60–80% of episodes are
  behavioral near-duplicates. Training on all of them inflates GPU cost and
  can reinforce dominant modes at the expense of rare but important behaviors.
- **GR00T incompatibility** — NVIDIA GR00T N1.7+ requires relative EEF actions,
  visual observations, and language annotations on every episode. Datasets from
  Isaac Lab ship in absolute world-frame coordinates with none of these set.

Calibra solves the data side.

---

## What It Does

### 1. Four analyzers, one report

```python
from calibra.ingestion.registry import load
from calibra.pipeline import Pipeline

batch  = load("lerobot/droid_100")
report = Pipeline().run(batch, policy_family="diffusion")
print(report.summary())
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA DIAGNOSTIC REPORT — droid_100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Episodes : 100  ·  Steps : 32,212

  temporal_stability      OK
  control_smoothness      CRITICAL  ← LDLJ = -19.4, jerk spikes 4.5%
  coverage_entropy        OK
  task_structure          INFO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| Analyzer | What it measures |
|---|---|
| `TemporalAnalyzer` | Jitter CV, dropout rate, camera–physics lag |
| `ControlSmoothnessAnalyzer` | LDLJ, jerk spike rate, velocity discontinuity |
| `CoverageEntropyAnalyzer` | Action entropy, PCA variance, episode length |
| `TaskStructureAnalyzer` | Contact density, grasp events, trajectory diversity |

All metrics report **95% bootstrap confidence intervals** computed over episodes
(not steps), avoiding artificially narrow intervals from correlated within-episode
samples.

### 2. Coreset selection — quality filter + greedy max-coverage

```python
from calibra.pruning import CoresetSelector

selector = CoresetSelector(keep_fraction=0.3, entropy_weight=0.4)
result   = selector.select(batch, report)

print(result.summary())
# Original: 1000 episodes → Coreset: 300 episodes
# Stage 1 (quality): removed 87  (jerk/dropout/short-episode failures)
# Stage 2 (diversity): removed 613  (near-duplicates under max-coverage)
```

Stage 2 uses **greedy farthest-point sampling** on action-space statistics:
O(N × K), handles ~50k episodes without approximation.

### 3. GR00T compatibility checks

Pass `policy_family="gr00t"` to activate five structural checks for NVIDIA
GR00T N1 fine-tuning:

```bash
calibra certify /data/isaac_lab_demos.h5 --policy gr00t
```

```
⚠  PROVISIONALLY CERTIFIED

Warnings:
  • gr00t.language_annotations: Only 48/50 episodes (96%) have a
    task_description string. Episodes without a task description receive
    a null language token, degrading policy generalisation.
```

Exit codes: `0` = CERTIFIED, `1` = PROVISIONALLY CERTIFIED, `2` = NOT CERTIFIED.
Wire into CI with `--json` for machine-readable output.

### 4. Retarget: absolute EEF → relative deltas

GR00T N1.7+ uses a Relative End-Effector action space. Isaac Lab records in
world-frame absolute coordinates.

```bash
calibra retarget /data/isaac_lab_demos.h5 --out retargeted/
# Episodes: 500 converted  ·  Action shape: (T−1, 6)  [dx, dy, dz, droll, dpitch, dyaw]
```

---

## Evidence-backed comparisons

Every interpretation in `calibra compare` output is backed by a falsifiable
claim in `calibra/claims/`. Each claim tracks its evidence count, confidence
tier (NOT VALIDATED → LOW → MEDIUM → HIGH → STRONG), and a stated
falsification condition.

```bash
calibra compare /data/my_demos aloha
```

```
VELOCITY DISCONTINUITY RATE
  Yours:  12.1%
  aloha   1.3%
  Delta:  +10.8%  ▲  Confidence: HIGH · [n=4 real-hardware aloha datasets]

  Significantly rougher than aloha. If using position commands: investigate
  control noise or abrupt operator corrections.
```

Calibra ships 13 reference profiles (pusht, aloha variants, DROID-100, SO-100,
BridgeData V2). Every community-contributed profile strengthens the evidence base.

---

## Try it

```bash
pip install 'calibra-robotics[lerobot]'

# Audit a public LeRobot dataset
calibra lerobot/pusht --policy diffusion

# Compare yours against DROID-100
calibra compare /data/my_demos droid_100
```

**Interactive walkthrough:** the
[GR00T workflow notebook](https://colab.research.google.com/github/omerTT/Calibra/blob/main/notebooks/gr00t_workflow.ipynb)
shows the full pipeline from raw Isaac Lab demos to a retargeted GR00T coreset,
running in ~10 seconds on CPU.

---

## Formats supported

LeRobot v1/v2 (Parquet + DuckDB), HuggingFace Hub IDs, HDF5 (Isaac Lab,
Robomimic), RLDS / TF Datasets, MCAP / ROS2 bags.

**GitHub:** https://github.com/omerTT/Calibra
