# Calibra

**Dataset observability and coreset selection for robotics imitation learning.**

Calibra tells you what is wrong with your robot demonstrations — and removes the redundant ones — before you waste GPU time training on bad data.

```bash
pip install calibra
calibra compare hf://lerobot/my_dataset aloha
calibra certify /data/my_demos --reference aloha --policy diffusion
calibra prune   /data/100k_episodes --keep 0.3 --out coreset.json
calibra retarget /data/isaac_lab.h5 --out retargeted/
```

---

## The problem

Robot learning labs collect thousands of demonstration episodes. Naively training on all of them:

- **Silently trains on bad data** — jerk spikes, dropped frames, communication lag, and stuck actuators all look like valid training signal to your policy.
- **Wastes compute on redundancy** — in a 10,000-episode dataset, 60–80% of episodes are near-duplicates. GPU cost scales with volume, not uniqueness.
- **Produces undiagnosable failures** — when a policy stalls or flails, you have no idea whether the cause is the architecture, the training recipe, or the data itself.

Calibra solves the data side.

---

## Six commands

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

Use `--entropy-weight 0.4` (or `--policy gr00t`) to bias selection toward high-entropy (informationally rich) episodes, which improves GR00T fine-tuning outcomes.

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

---

## Install

```bash
# Core (numpy + pydantic only — no format adapters)
pip install calibra

# With LeRobot / HuggingFace Hub support (recommended)
pip install 'calibra[lerobot]'   # Parquet, DuckDB, Hub IDs

# Other format adapters
pip install 'calibra[hdf5]'      # HDF5 (Isaac Lab, Robomimic)
pip install 'calibra[rlds]'      # RLDS / TensorFlow Datasets
pip install 'calibra[mcap]'      # MCAP / ROS2 bags

# Everything
pip install 'calibra[all]'
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

### Profile more datasets (highest-value contribution)

The evidence base for `calibra compare` grows with every new reference profile. Priority targets from `calibra/claims/` pending tests:

| Dataset | Why it matters |
|---|---|
| `lerobot/droid_100` | Large-scale real hardware, position control — validates VD-001 at scale |
| `lerobot/so100` | Low-cost hardware platform — establishes noise characteristics for affordable arms |
| `lerobot/bridge` (BridgeData V2) | Second velocity-command dataset — needed to validate JS-002, VD-002 |
| Any Isaac Lab sim dataset | Validates TEMP-001 (sim jitter) across a second simulator |

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
pytest              # 259 tests
ruff check .        # zero errors expected
```

---

## What Calibra is not

- **Not a dataset score** ("your dataset is 7.4/10") — Calibra surfaces specific, falsifiable anomalies
- **Not an AI assistant** — it runs deterministic mathematical estimators, not a language model
- **Not a cloud service** — it runs entirely locally against your files
- **Not a replacement for domain expertise** — it tells you *what* to look at; you decide *what to do*

---

## License

[MIT](LICENSE)
