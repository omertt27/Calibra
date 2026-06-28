# Calibra

<p align="center">
  <img src="docs/logo.svg" alt="Calibra — dataset observability for robotics" width="480"/>
</p>

<p align="center">
  <a href="https://github.com/omerTT/Calibra/actions/workflows/ci.yml"><img src="https://github.com/omerTT/Calibra/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://pypi.org/project/calibra-robotics/"><img src="https://img.shields.io/pypi/v/calibra-robotics.svg" alt="PyPI"/></a>
  <a href="https://pypi.org/project/calibra-robotics/"><img src="https://img.shields.io/pypi/pyversions/calibra-robotics.svg" alt="Python Support"/></a>
  <a href="https://omerTT.github.io/Calibra/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Documentation"/></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Code Style: Ruff"/></a>
  <a href="https://pepy.tech/project/calibra-robotics"><img src="https://pepy.tech/badge/calibra-robotics/month" alt="PyPI Downloads"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>


**Dataset observability and coreset selection for robotics imitation learning.**

Calibra tells you what is wrong with your robot demonstrations — and removes the redundant ones — before you waste GPU time training on bad data.

```bash
pip install calibra-robotics
calibra compare hf://lerobot/my_dataset aloha
calibra certify /data/my_demos --reference aloha --policy diffusion
calibra prune   /data/100k_episodes --keep 0.3 --out coreset.json
calibra retarget /data/isaac_lab.h5 --out retargeted/

# New in v0.6.0
calibra predict /data/my_demos.h5 --policy gr00t          # predict training success before training
calibra predict /data/my_demos.h5 --record-outcome 0.82   # close the loop after training
calibra card    /data/my_demos.h5 --push                   # push quality card to HuggingFace Hub
calibra watch   /data/session/ --remediate                 # real-time operator feedback
calibra watch   --stream --remediate                       # pipe metrics from collect script
calibra calibrate                                          # re-fit weights from training history
```

---

## The problem

Robot learning labs collect thousands of demonstration episodes. Naively training on all of them:

- **Silently trains on bad data** — jerk spikes, dropped frames, communication lag, and stuck actuators all look like valid training signal to your policy.
- **Wastes compute on redundancy** — in a 10,000-episode dataset, 60–80% of episodes are near-duplicates. GPU cost scales with volume, not uniqueness.
- **Produces undiagnosable failures** — when a policy stalls or flails, you have no idea whether the cause is the architecture, the training recipe, or the data itself.

Calibra solves the data side.

---

## Commands

### 1. `audit` — full diagnostic report

```bash
calibra /data/robot_demos.h5
calibra lerobot/pusht --policy diffusion
calibra /data/demo.h5 --policy act --json
```

Runs four analyzers over every episode and flags anomalies with bootstrap confidence intervals and per-episode outlier detection.

### 2. `compare` — evidence-backed cross-dataset comparison

```bash
calibra compare /data/my_demos pusht
calibra compare hf://lerobot/my_dataset aloha
calibra compare /data/robot.h5 aloha --format hdf5 --gripper-dims 6,13
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra compare — my_dataset  vs.  aloha
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reference: lerobot/aloha_mobile_cabinet  (position-command · 14D · 85 episodes)
Yours:     my_dataset  (120 episodes)

────────────────────────────────────────────────────────
VELOCITY DISCONTINUITY RATE
  Yours:  12.1%
  aloha   1.3%
  Delta:  +10.8%  ▲

  Significantly rougher than aloha_mobile_cabinet.
  If using position commands: investigate control noise or
  abrupt operator corrections.

  Confidence: HIGH · [HIGH · n=2 (aloha_sim, aloha_mobile)]
────────────────────────────────────────────────────────
JERK SPIKE RATE
  Yours:  8.4%
  aloha   0.7%
  Delta:  +7.7%  ▲

  Higher spike rate than reference. Check for dropped
  frames, bad episode boundaries, or bimodal speed profiles.

  Confidence: MODERATE · [LOW-MODERATE · n=1 (aloha_sim)]
────────────────────────────────────────────────────────

RECOMMENDED ACTIONS
────────────────────────────────────────────────────────
  Prune episode(s) 14, 22, 41 — jerk outliers detected by MAD analysis.
  Velocity discontinuity rate is 12.1% (above 4% position-control
  threshold). Investigate command packet drops, hardware communication
  lag, or abrupt operator corrections.
────────────────────────────────────────────────────────
```

Every interpretation is backed by a falsifiable claim in `calibra/claims/` with an evidence count, confidence rating, and a stated falsification condition. Calibra never guesses.

### 3. `certify` — structured pass/fail certification

```bash
calibra certify /data/my_demos
calibra certify /data/my_demos --reference aloha --policy diffusion --strict
calibra certify hf://lerobot/my_dataset --json   # for CI pipelines
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA CERTIFICATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dataset  : my_demos
  Episodes : 120
  Steps    : 180000
  Policy   : diffusion
  Reference: aloha

  ──────────────────────────────────────────────────────────
  ⚠  PROVISIONALLY CERTIFIED

  Warnings:
    • ldlj: Mean LDLJ = -12.4 (threshold: >-10). Action trajectories
      contain significant jerk.

  ──────────────────────────────────────────────────────────
  REMEDIATION CHECKLIST
  ──────────────────────────────────────────────────────────
  1. [WARNING] ldlj: High jerk in demonstration data forces the policy
     to learn discontinuous action transitions. Consider applying action
     smoothing (e.g. Savitzky-Golay) before training.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Exit codes: `0` = CERTIFIED, `1` = PROVISIONALLY CERTIFIED (warnings), `2` = NOT CERTIFIED (critical failures). Wire into CI with `--json` for machine-readable output.

### 4. `prune` — coreset selection

```bash
calibra prune /data/100k_episodes --keep 0.3 --out coreset.json
calibra prune /data/my_ds --keep 0.5 --quality-only
calibra prune /data/my_ds --keep 0.25 --max-spike-rate 0.03 --max-vel-disc-rate 0.08
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA PRUNING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Original episodes  : 1000
  Quality failures   : 87   (removed in Stage 1)
  Diversity pruned   : 613  (removed in Stage 2)
  Coreset size       : 300  (30.0% of original)
  Method             : quality_filter + greedy_max_coverage
────────────────────────────────────────────────────────
  To use: filter your dataset to the episode IDs in keep_episode_ids.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Two-stage pipeline:
- **Stage 1 — Quality filter:** removes episodes that fail kinematic/temporal thresholds (jerk spike rate, velocity discontinuity, dropout, LDLJ, minimum length).
- **Stage 2 — Greedy max-coverage:** from the quality-passing pool, selects the K most behaviorally diverse episodes using farthest-point sampling on action-space statistics. O(N × K) — handles ~50k episodes without approximation.

Use `--entropy-weight 0.4` (or `--policy gr00t`) to bias selection toward high-entropy (informationally rich) episodes, which improves GR00T fine-tuning outcomes. Alternatively, use `--strategy influence` to select episodes based on estimated learning value (combining action novelty, task contact representation, and Shannon entropy).

Output `coreset.json` contains `keep_episode_ids`, `quality_fail_ids`, `diversity_pruned_ids`, and per-episode quality and diversity scores.

### 5. `corrupt` — validate metric sensitivity

```bash
calibra corrupt lerobot/pusht --drop-frames 0.10
calibra corrupt /data/robot.h5 --inject-spikes 0.05
calibra corrupt lerobot/pusht --add-jitter-ms 50 --drop-frames 0.08
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra corrupt — pusht
Corruptions: drop_frames=10.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Metric                      Original   Corrupted       Δ  React
──────────────────────────────────────────────────────────────────
  Timestamp dropout rate         0.0%       9.4%    +9.4%  🔴
  Timestamp jitter CV          3.0e-06    8.1e-06  +5.1e-06 🟡
  Jerk spike rate                4.9%       5.2%    +0.3%   —
  Velocity discontinuity        16.7%      16.9%    +0.2%   —
```

Inject synthetic corruptions into a known-good dataset to verify that your metrics actually respond to the defects they claim to detect.

### 6. `retarget` — convert absolute EEF actions to relative deltas

```bash
calibra retarget /data/isaac_lab_demos.h5 --out /data/retargeted/
calibra retarget /data/demos.h5 --pad --out retargeted/
calibra retarget /data/demos.h5 --obs-key-pos robot0_eef_pos \
                                 --obs-key-quat robot0_eef_quat
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  calibra retarget — isaac_lab_demos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Episodes converted : 500
  Episodes skipped   : 0
  Output directory   : /data/retargeted/
  Action shape       : (T−1, 6)  [dx, dy, dz, droll, dpitch, dyaw]
  Rotation units     : radians (intrinsic XYZ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

NVIDIA GR00T N1.7+ uses a **Relative End-Effector (EEF)** action space. Isaac Lab and robomimic HDF5 datasets record actions in absolute world-frame coordinates. `retarget` converts absolute 7-DoF poses `[x, y, z, qx, qy, qz, qw]` into 6-DoF local-frame deltas `[dx, dy, dz, droll, dpitch, dyaw]` — one `.npz` per episode.

Use `--pad` to append a zero row so output shape is `(T, 6)` instead of `(T−1, 6)` when your policy requires fixed-length sequences.

### 7. `predict` — predict training outcome before spending GPU time

```bash
calibra predict /data/my_demos.h5
calibra predict lerobot/my_dataset --policy diffusion --reference aloha
calibra predict /data/my_demos.h5 --policy gr00t --record-outcome 0.82
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA TRAINING OUTCOME PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dataset  : my_demos  ·  Episodes: 120  ·  Policy: gr00t
  🟢  Predicted Success: 81%  [range 71%–91%]  —  GOOD
  ──────────────────────────────────────────────────────────
  ⚠️  -8.0pt  ldlj
     Mean LDLJ = -12.4. High jerk forces discontinuous action transitions.
  ──────────────────────────────────────────────────────────
  NEXT STEPS
  ✓ Data quality is sufficient. Proceed with training.
  After training, close the loop:
    calibra predict <dataset> --record-outcome <actual_success_rate>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

`--record-outcome RATE` stores the observed training success rate alongside the diagnostic fingerprint in `~/.calibra/outcomes.jsonl`. Future predictions on similar datasets blend the heuristic score with these empirical observations via inverse-distance weighting. Run `calibra calibrate` after 10+ outcomes to re-fit the prediction weights from your lab's actual training history.

### 8. `card` — HuggingFace dataset quality card

```bash
calibra card /data/my_demos.h5
calibra card lerobot/my_dataset --policy diffusion --out quality_card.md
calibra card /data/my_demos.h5 --push   # push directly to HuggingFace Hub README
```

Generates a structured Markdown quality card with certification badge, per-metric status table, and predicted training outcome. Embed it in your dataset's HuggingFace Hub README so other researchers can see data quality at a glance.

### 9. `watch` — real-time teleoperation quality monitor

```bash
calibra watch /data/collection_session/
calibra watch /data/session/ --remediate          # print fix instructions on failure
calibra watch /data/session/ --log-file session.jsonl

# Stream mode: pipe metrics from your collection script
python collect_demos.py | calibra watch --stream --remediate
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA WATCH — real-time data quality monitor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Remediation advice: ON
  Watching: /data/collection_session/

  ✅ [   1] ep_001.h5         PASS — all metrics OK
  ✅ [   2] ep_002.h5         PASS — all metrics OK
  ❌ [   3] ep_003.h5         FAIL — jerk_spike_rate = 0.087
       ↳ RE-RECORD: Move more smoothly — avoid abrupt stops and direction changes.
  ✅ [   4] ep_004.h5         PASS — all metrics OK
```

`--remediate` prints a specific operator instruction on every FAIL/WARN: what caused the failure and exactly how to fix the motion. Operators get feedback within seconds of saving an episode instead of discovering problems during training hours later.

`--stream` reads JSON metric lines from stdin, enabling integration with teleoperation software without filesystem round-trips. See `examples/lerobot_watch_integration.py` for a drop-in integration snippet.

---

## Empirical Validation

Calibra is backed by empirical testing on real robotics datasets and coreset selection experiments.

### Predictor success correlation

Offline predicted success probabilities (`calibra predict`) achieve a **Spearman rank correlation (ρ) of 0.5971** (p = 0.0146, statistically significant) with actual downstream policy success rates across 16 standard datasets (ALOHA, DROID-100, BridgeData, PushT, SVLA SO-100).

### Ablation study: which selection component drives the gains?

A five-condition ablation isolates which part of Calibra's pipeline contributes to the improvement. All experiments use BC-MLP policies, 5 random seeds for the baseline, 200–300 epochs, RTX 2080. Gains are relative to the random-k baseline at the same episode budget.

**Without contact-aware quality filter (baseline):**

| Condition | ALOHA mobile (clean, 14D) | DROID-100 (noisy, 7D) | PushT real (contact, 8D) |
|---|---|---|---|
| Random k (5 seeds) | 0.0209 ± 0.003 | 1.995 ± 0.177 | 0.222 ± 0.080 |
| Quality-filter only | +8.8% | **−5.8% (hurts)** | **−29.2% (hurts)** |
| Diversity-only | +13.0% | +16.9% | +48.4% |
| Calibra full pipeline | +22.6% | +16.9% | **−30.5% (hurts)** |

**With contact-aware quality filter (`contact_aware=True`, default):**

| Condition | ALOHA mobile | DROID-100 | PushT real |
|---|---|---|---|
| Quality-filter only | +8.8% (unchanged) | −5.8% (unchanged) | **+37.1%** |
| Diversity-only | +13.0% (unchanged) | +16.9% (unchanged) | +46.3% (unchanged) |
| **Calibra full pipeline** | **+22.6%** (unchanged) | **+16.9%** (unchanged) | **+39.3%** |

The contact-aware filter detects that real PushT's velocity discontinuities are contact-driven (vel_disc/spike ratio = 24.4) and relaxes the vel_disc threshold by 3×. This transforms a −30.5% failure into a +39.3% gain while leaving ALOHA and DROID completely unchanged (their ratios are 1.9 and 1.6 — well below the 3.0 threshold).

**Quality filtering failure modes (without contact-aware fix):**

1. **Heterogeneous datasets (DROID)** — quality-only collapses coverage of rare robot morphologies. Removing the noisiest episodes inadvertently removes the only representatives of certain robot types.

2. **Contact-rich tasks (real PushT)** — velocity discontinuities during contact events (block-hits, pushes) are classified as noise. The quality filter removes the most contact-rich demonstrations, which have the highest learning signal.

The contact-aware fix addresses failure mode 2 by using the vel_disc/spike ratio as a contact detector. Failure mode 1 (morphology collapse) remains an open problem — see `calibra/strategy.py` for the regime-dependent configuration that mitigates it via relaxed thresholds.

---

### Dataset regime space

Calibra's diagnostic metrics predict which selection regime applies before any training:

```python
from calibra.pipeline import Pipeline
from calibra.strategy import diagnose_regime

report = Pipeline().run(batch)
diagnosis = diagnose_regime(report)
print(diagnosis.regime)        # SelectionRegime.LOW_NOISE / MODERATE_NOISE / HIGH_NOISE
print(diagnosis.explanation)   # human-readable mechanism description
selector = CoresetSelector(keep_fraction=0.3, **diagnosis.recommended_config)
```

Regime classification across tested datasets:

| Dataset | vel_disc/spike ratio | Diagnosed regime | Calibra full (before fix) | Calibra full (after fix) |
|---|---|---|---|---|
| ALOHA mobile | 1.9 | LOW NOISE | +22.6% | +22.6% (unchanged) |
| DROID-100 | 1.6 | MODERATE NOISE | +16.9% | +16.9% (unchanged) |
| PushT real | **24.4** | HIGH NOISE (contact) | **−30.5%** | **+39.3%** |

The vel_disc/spike ratio correctly identifies PushT real as contact-driven and triggers the 3× threshold relaxation. See [`experiments/regime_space.py`](experiments/regime_space.py) for the visualization.

> **Note:** These results represent a hypothesis supported by 3 datasets, not an established law. The pattern is consistent and reproducible but requires validation across more policy families and embodiments before being treated as a general principle.

---

### Retention curves

Calibra's advantage over random selection is stable across the full data-fraction range (10%–70%) on clean and heterogeneous datasets. On contact-rich tasks, diversity-only selection (without quality filtering) should be used instead of the full pipeline.

**ALOHA mobile — Calibra full vs. random at each keep fraction:**

| Keep | k | Calibra vs. Random | Calibra vs. Full |
|---|---|---|---|
| 10% | 7 | +50.0% | −101% |
| 20% | 14 | +35.5% | −65% |
| 30% | 20 | +29.2% | −46% |
| 50% | 34 | +7.8% | −29% |
| 70% | 48 | +7.2% | −17% |

Advantage over random is largest at small budgets (+50% at 10%) and narrows as budget grows — on clean structured datasets, more data continues to help, so the coreset closes but does not eliminate the full-data gap.

**DROID-100 — Calibra full vs. random:**

| Keep | Calibra vs. Random | Calibra vs. Full |
|---|---|---|
| 10% | +20.9% | +16.8% |
| 20% | +15.2% | +10.5% |
| 30% | +10.1% | +11.8% |
| 50% | +7.2% | +11.3% |
| 70% | +4.5% | +7.3% |

On DROID, Calibra with 10% of episodes outperforms the full dataset by 16.8%.

To reproduce all ablations:
```bash
pip install "calibra-robotics[lerobot]" torch
python experiments/ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet --n-epochs 300 --seeds 5 --curve --save-fig
python experiments/ablation_benchmark.py --dataset lerobot/droid_100 --n-epochs 300 --seeds 5 --curve --save-fig
python experiments/ablation_benchmark.py --dataset lerobot/columbia_cairlab_pusht_real --n-epochs 200 --seeds 5 --curve --save-fig
python experiments/regime_space.py --save
```

---

## Install

> **PyPI package name:** `calibra-robotics`  (the `calibra` name on PyPI is an unrelated package)

```bash
# Core (numpy + pydantic only — no format adapters)
pip install calibra-robotics

# With LeRobot / HuggingFace Hub support (recommended)
pip install 'calibra-robotics[lerobot]'   # Parquet, DuckDB, Hub IDs

# Other format adapters
pip install 'calibra-robotics[hdf5]'      # HDF5 (Isaac Lab, Robomimic)
pip install 'calibra-robotics[rlds]'      # RLDS / TensorFlow Datasets
pip install 'calibra-robotics[mcap]'      # MCAP / ROS2 bags

# Everything
pip install 'calibra-robotics[all]'
```

---

## Python API

### Quick metric checks (no pipeline required)

```python
from calibra.metrics import (
    compute_velocity_discontinuity_rate,
    compute_jerk_spike_rate,
    compute_ldlj,
    compute_action_entropy,
    compute_jitter_cv,
)

# actions / states: np.ndarray of shape (T, D)
disc   = compute_velocity_discontinuity_rate(actions, states, dt=0.02)
jerk   = compute_jerk_spike_rate(states, dt=0.02, sigma_limit=5.0)
ldlj   = compute_ldlj(positions, dt=0.02)
entopy = compute_action_entropy(actions)
cv     = compute_jitter_cv(timestamps)
```

### SQL-level per-episode queries (local v2 datasets)

```python
from calibra.core import LazyDatasetReader

with LazyDatasetReader("/data/lerobot/aloha_mobile") as reader:
    print(reader.fps, reader.episode_count())

    # Query only the columns you need — images never leave the Parquet pages
    table = reader.query_proprioception_tensors(
        ["observation.state", "action"], episode_idx=0
    )
    actions = table["action"].to_pylist()
```

### Full pipeline

```python
from calibra.ingestion.registry import load
from calibra.pipeline import Pipeline

batch  = load("lerobot/pusht")           # Hub ID, local path, or hf:// URI
report = Pipeline().run(batch, policy_family="diffusion")
print(report.summary())
```

### Coreset selection

```python
from calibra.pruning import CoresetSelector

selector = CoresetSelector(
    keep_fraction=0.3,
    max_spike_rate=0.05,
    max_vel_disc_rate=0.15,
)
result = selector.select(batch, report)

print(result.summary())
# result.keep_episode_ids → filter your Parquet shards
```

### Custom schema mapping

```python
from calibra.core import SchemaNormalizer

# YAML config for your robot's naming convention
n = SchemaNormalizer(config_path="my_robot/mappings.yaml")
normalized = n.normalize({"my_robot/joints/q": arr, "my_robot/ee": arr2})
```

---

## How it works

```
Dataset (Parquet / HDF5 / RLDS / MCAP / Hub ID / hf:// URI)
    │
    ▼  Format adapters — metadata-first, DuckDB lazy scan skips images
EpisodeBatch — normalised internal representation
    │
    ▼  Four analyzers (parallelisable)
DiagnosticReport — flags + 95% bootstrap CIs + per-episode arrays
    │
    ├──▶  audit   — terminal summary + MAD outlier table
    ├──▶  compare — evidence-backed cross-dataset comparison
    ├──▶  certify — CERTIFIED / PROVISIONALLY CERTIFIED / NOT CERTIFIED
    └──▶  prune   — quality filter + greedy max-coverage coreset
```

### Analyzers

| Analyzer | Metrics computed |
|---|---|
| `TemporalAnalyzer` | timestamp jitter CV, dropout rate, camera lag std, action-obs misalignment |
| `ControlSmoothnessAnalyzer` | LDLJ, jerk spike rate, velocity discontinuity rate, action-state divergence |
| `CoverageEntropyAnalyzer` | action entropy (bits/dim), state entropy, PCA top-2 variance, episode length distribution |
| `TaskStructureAnalyzer` | contact density, grasp events per episode, trajectory diversity score, short episode fraction |

All metrics report 95% bootstrap confidence intervals computed over episodes (not steps), avoiding artificially narrow intervals from correlated within-episode samples.

### Claim registry

Every interpretation in `calibra compare` output is backed by a falsifiable claim in `calibra/claims/`. Each claim tracks:

- **assertion** — what the metric is expected to show for a given dataset class
- **evidence** — which datasets have been profiled and whether they support the claim
- **confidence** — derived from evidence count: NOT VALIDATED → LOW → MEDIUM → HIGH → STRONG
- **falsification condition** — exactly what data would invalidate the claim
- **pending tests** — the highest-value next dataset to profile

See [`docs/claims.md`](docs/claims.md) for the full registry.

**Ratio rule:** the number of reference profiles must be ≥ the number of active claims.  
Enforce in CI with: `python scripts/generate_claims_doc.py --check`

### Reference profiles

Three empirical baselines are shipped:

| Reference | Control | Freq | DOF | Episodes | Hardware |
|---|---|---|---|---|---|
| `pusht` | velocity | 10 Hz | 2 | 206 | sim |
| `aloha_sim` | position | 50 Hz | 14 | 50 | sim |
| `aloha_mobile_cabinet` | position | 50 Hz | 14 | 85 | ✓ real |
| `aloha_mobile_shrimp` | position | 50 Hz | 14 | 100 | ✓ real |
| `aloha_sim_insertion_scripted` | position | 50 Hz | 14 | 50 | sim |
| `aloha_sim_transfer_cube_scripted` | position | 50 Hz | 14 | 50 | sim |
| `aloha_sim_transfer_cube_human` | position | 50 Hz | 14 | 50 | sim |
| `aloha_static_battery` | position | 50 Hz | 14 | — | ✓ real |
| `aloha_static_candy` | position | 50 Hz | 14 | — | ✓ real |
| `aloha_static_coffee` | position | 50 Hz | 14 | — | ✓ real |
| `aloha_static_cups_open` | position | 50 Hz | 14 | — | ✓ real |
| `pusht_image` | velocity | 10 Hz | 2 | — | sim |
| `droid_100` | position | 15 Hz | 7 | 100 | ✓ real |
| `svla_so100_pickplace` | position | 15 Hz | 6 | 50 | ✓ real |
| `svla_so100_stacking` | position | 15 Hz | 6 | 56 | ✓ real |
| `bridgedata_v2` | velocity | 5 Hz | 7 | 50415 | ✓ real |

Add your own with `scripts/profile_dataset.py` (see Contributing).

---

## Formats supported

| Format | Extra install | Notes |
|---|---|---|
| LeRobot v2 (Parquet shards) | `calibra[lerobot]` | DuckDB lazy scan — image columns never enter RAM |
| LeRobot v1 (HF datasets) | `calibra[lerobot]` | HuggingFace `datasets` + pandas groupby |
| HuggingFace Hub | `calibra[lerobot]` | `lerobot/pusht`, `hf://lerobot/pusht` |
| HDF5 (Isaac Lab, Robomimic) | `calibra[hdf5]` | Convention A + B |
| RLDS / TF Datasets | `calibra[rlds]` | tensorflow-datasets |
| MCAP / ROS2 bags | `calibra[mcap]` | mcap + mcap-ros2-support |

---

## Contributing

Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started with setting up development environments, formatting rules, testing, and submitting new dataset profiles or claims.

### Profile more datasets (highest-value contribution)


The evidence base for `calibra compare` grows with every new reference profile. Priority targets from `calibra/claims/` pending tests:

| Dataset | Why it matters |
|---|---|
| `lerobot/droid_100` | ✅ Profiled — large-scale real hardware, position control, validates VD-001 at scale |
| `lerobot/svla_so100_pickplace` | ✅ Profiled — SO-100 low-cost arm, pick-and-place |
| `nvidia/BridgeData2_LeRobot_v3` | ✅ Profiled — falsified VD-002 and JS-002; revealed frequency-dependent behaviour |
| Any Isaac Lab sim dataset | Validates TEMP-001 (sim jitter) across a second simulator |
| Any second 5Hz velocity-command dataset | Validates JS-004 and VD-003 successor claims |

```bash
python scripts/profile_dataset.py lerobot/droid_100 \
  --control-mode position \
  --out calibra/references/droid_100.json \
  --note "DROID, real hardware, various robots, 15Hz"
```

After profiling, open `calibra/claims/*.json`, find claims with this dataset in `pending_tests`, add an evidence entry, then regenerate the docs:

```bash
python scripts/generate_claims_doc.py
```

### Repository layout

```
calibra/
├── core/               # Public API: LazyDatasetReader, SchemaNormalizer, mappings.yaml
├── metrics/            # Standalone pure-numpy functions (no pipeline needed)
├── analyzers/          # Pipeline analyzers: temporal, smoothness, coverage, task_structure
├── ingestion/          # Format adapters (lerobot, hdf5, rlds, mcap) + registry
├── comparison/         # DatasetComparator, EpisodeCurator
├── schema/             # EpisodeBatch, DiagnosticReport, normalization layer
├── claims/             # Falsifiable claim registry (JSON + SPEC.md)
├── knowledge_base/     # claims.yaml (auto-generated — edit the source JSON files)
├── references/         # Profiled reference datasets (JSON)
└── interpretations/    # Metric interpretation docs (Markdown)

scripts/
├── profile_dataset.py      # Profile any dataset → references/<name>.json
└── generate_claims_doc.py  # Regenerate docs/claims.md + CI ratio check

docs/
└── claims.md               # Auto-generated from calibra/claims/ — do not edit
```

### Development

```bash
git clone https://github.com/omerTT/Calibra
pip install -e '.[all,dev]'
pytest              # 377 tests
ruff check .        # zero errors expected
```

---

## Calibra for IL vs. World Models

Calibra supports two data curation philosophies. The right one depends on your policy architecture. Both use the same tool — the same quality metrics, the same CLI — but apply different selection criteria in Stage 2 of `calibra prune`.

### IL / behaviour cloning (diffusion policy, ACT, GR00T fine-tuning)

```bash
# For diffusion policy, ACT, GR00T fine-tuning
calibra prune /data/demos --keep 0.3 --strategy diversity
calibra watch /data/session/ --remediate
```

**Selection goal:** remove corrupted episodes, keep behaviorally diverse ones.  
**Metrics that matter:** jerk spike rate, dropout, LDLJ, velocity discontinuity.

Stage 2 runs greedy max-coverage over the quality-passing pool — maximising behavioural spread across the action-space. A policy trained on a diverse coreset generalises better than one trained on near-duplicate demonstrations.

### World model training (JEPA-based, latent MPC)

```bash
# For JEPA-based world models (I-JEPA, V-JEPA successors, latent MPC)
calibra prune /data/demos --keep 0.3 --strategy world-model
calibra watch /data/session/ --world-model
```

**Selection goal:** select episodes that maximise what the world model doesn't yet know — highest latent prediction error relative to the current model state.  
**Requires:** `pip install torch` (PyTorch is an optional dependency).

Stage 2 scores each quality-passing episode by JEPA surprise (reconstruction error in latent space) and keeps the highest-surprise fraction. The result is a dataset that pushes the world model toward unexplored dynamics rather than reinforcing what it already knows.

### Surprise × quality decision table

| World-model surprise | Kinematic quality | Interpretation | Action |
|---|---|---|---|
| HIGH | FAIL (high jerk) | Corrupted episode | Prune |
| HIGH | PASS (smooth) | Genuinely novel dynamics | Keep |
| LOW | any | Redundant / well-covered | Prune |

The table is the key insight: surprise alone is not sufficient. A noisy actuator or a packet-drop produces high prediction error but teaches the world model nothing useful. Stage 1 quality filtering — identical in both workflows — is what separates genuinely novel episodes from corrupted ones before surprise scoring ever runs.

### Why the same data infrastructure works for both

Clean data is a prerequisite for both paradigms. A JEPA trained on jittery, dropout-heavy data learns corrupted latent representations that degrade downstream planning and latent MPC rollouts. The quality filtering stage (Stage 1 of `calibra prune`) is therefore identical for both — only Stage 2 (greedy diversity selection vs. surprise-maximisation) changes.

`calibra sim2real` reports a world-model transfer gap metric alongside the standard distribution-gap analysis, quantifying how much of the latent dynamics your sim fails to cover before you commit to training.

### When to use which

- **Using diffusion policy, ACT, or fine-tuning a GR00T/Octo checkpoint** → use `--strategy diversity`
- **Training a JEPA world model from scratch, or maximising the information content of a small dataset** → use `--strategy world-model`
- **Collecting new data in real-time** → `calibra watch --world-model` tells you which episodes are genuinely novel vs. redundant *as you collect*, so operators can prioritise effort on configurations the model hasn't seen

---

## What Calibra is not

- **Not a dataset score** ("your dataset is 7.4/10") — Calibra surfaces specific, falsifiable anomalies
- **Not an AI assistant** — it runs deterministic mathematical estimators, not a language model
- **Not a cloud service** — it runs entirely locally against your files
- **Not a replacement for domain expertise** — it tells you *what* to look at; you decide *what to do*

---

## License

[MIT](LICENSE)
