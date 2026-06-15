# Calibra

**Dataset observability for robotics imitation learning.**

Calibra tells you what is wrong with your robot demonstrations — before you waste GPU time training on bad data.

---

## The 30-second demo

```bash
pip install 'calibra[lerobot]'
calibra compare hf://lerobot/my_dataset aloha
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra compare — my_dataset  vs.  aloha_mobile_cabinet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reference: lerobot/aloha_mobile_cabinet  (position-command · 14D · 120 episodes)
Yours:     my_dataset  (85 episodes)

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
...

RECOMMENDED ACTIONS
────────────────────────────────────────────────────────
  Prune episode(s) 14, 22, 41 — jerk outliers detected by MAD analysis.
  Velocity discontinuity rate is 12.1% (above 4% position-control
  threshold). Investigate command packet drops, hardware communication
  lag, or abrupt operator corrections.
────────────────────────────────────────────────────────
```

---

## What Calibra does

Calibra runs a diagnostic pipeline over robotics demonstration datasets and answers:

- **Is my data smooth enough?** (jerk spikes, velocity discontinuities, LDLJ)
- **Are my timestamps reliable?** (jitter, dropout, camera-joint misalignment)
- **Is my action distribution diverse enough?** (entropy, PCA collapse)
- **Which specific episodes should I prune before training?** (MAD-based outlier detection)
- **How does my dataset compare to known-good references?** (evidence-tracked claim registry)

Every interpretation is backed by empirical evidence from profiled datasets and labeled with a confidence rating. Calibra never guesses.

## What Calibra is not

- It is not a dataset certification score ("your dataset is 7.4/10")
- It is not an AI assistant for data cleaning
- It is not a cloud service or a dashboard
- It is not a replacement for domain expertise

Calibra surfaces anomalies. You decide what to do with them.

---

## Install

```bash
# Core (numpy + pydantic only)
pip install calibra

# With format support
pip install 'calibra[lerobot]'   # LeRobot / HuggingFace Hub (Parquet)
pip install 'calibra[hdf5]'      # HDF5 (Isaac Lab, Robomimic)
pip install 'calibra[rlds]'      # RLDS / TensorFlow Datasets
pip install 'calibra[mcap]'      # MCAP / ROS2 bags

# Everything
pip install 'calibra[all]'
```

---

## Usage

### Audit a dataset

```bash
calibra /data/robot_demos.h5
calibra lerobot/pusht --policy diffusion
calibra /data/demo.h5 --policy act --json
```

### Compare against a reference

```bash
calibra compare /data/my_demos pusht
calibra compare hf://lerobot/my_dataset aloha
calibra compare /data/robot.h5 aloha --format hdf5
```

Available references: `pusht`, `aloha`, `aloha_mobile_cabinet`
(see `calibra/references/` — add your own with `scripts/profile_dataset.py`)

### Validate metric detection with synthetic corruptions

```bash
# Does dropout_rate actually respond to dropped frames?
calibra corrupt lerobot/pusht --drop-frames 0.10

# Does spike_rate respond to injected discontinuities?
calibra corrupt /data/robot.h5 --inject-spikes 0.05

# Compound corruption
calibra corrupt lerobot/pusht --add-jitter-ms 50 --drop-frames 0.08
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra corrupt — pusht
Corruptions: drop_frames=10.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Metric                          Original   Corrupted          Δ  React
────────────────────────────────────────────────────────────────────────
  Timestamp dropout rate             0.0%       9.4%      +9.4%  🔴
  Timestamp jitter CV              3.0e-06    8.1e-06   +5.1e-06  🟡
  Jerk spike rate                    4.9%       5.2%      +0.3%   —
  Velocity discontinuity            16.7%      16.9%      +0.2%   —
  ...
```

### Profile a new dataset (add to reference library)

```bash
# LeRobot Hub
python scripts/profile_dataset.py lerobot/droid_100 \
  --control-mode position \
  --out calibra/references/droid_100.json

# Real hardware with known gripper dim
python scripts/profile_dataset.py lerobot/bridge \
  --control-mode position --gripper-dims 6 \
  --out calibra/references/bridge.json \
  --note "BridgeData V2, real hardware, Franka Panda, 7-DOF, 5Hz"
```

---

## How it works

Calibra has four layers:

```
Dataset (Parquet / HDF5 / RLDS / MCAP)
    ↓  Format adapters (metadata-first, DuckDB lazy scan for Parquet)
EpisodeBatch  (normalised internal representation)
    ↓  Analyzers (temporal, smoothness, coverage, task structure)
DiagnosticReport  (flags + confidence intervals + raw metrics)
    ↓  compare / corrupt CLI
Terminal output  (metric table + evidence lines + recommended actions)
```

### Analyzers

| Analyzer | Metrics |
|----------|---------|
| `TemporalAnalyzer` | timestamp jitter CV, dropout rate, camera lag, action-obs misalignment |
| `ControlSmoothnessAnalyzer` | LDLJ, jerk spike rate, velocity discontinuity rate |
| `CoverageEntropyAnalyzer` | action entropy, state entropy, PCA variance, episode length distribution |
| `TaskStructureAnalyzer` | contact density, grasp events, trajectory diversity, short episode fraction |

### Claim registry

Every interpretation in `calibra compare` output is backed by a falsifiable claim in `calibra/claims/`. Claims track:

- **assertion** — what the metric is expected to show
- **evidence** — which datasets have been profiled and whether they support the claim
- **confidence** — derived from evidence count (NOT VALIDATED → LOW → MEDIUM → HIGH → STRONG)
- **falsification condition** — what data would invalidate the claim
- **pending tests** — the highest-value next dataset to profile

See [`docs/claims.md`](docs/claims.md) for the full registry.

**Rule:** The number of reference profiles must be ≥ number of active claims.
Run `python scripts/generate_claims_doc.py --check` to enforce this in CI.

---

## Adding a reference dataset

1. Profile it:
   ```bash
   python scripts/profile_dataset.py lerobot/bridge \
     --control-mode position --gripper-dims 6 \
     --out calibra/references/bridge.json
   ```

2. Check which claims it supports or falsifies:
   ```bash
   # Open calibra/claims/*.json, find entries with "pending_tests"
   # that include this dataset or its class.
   # Add a supporting/falsifying evidence entry.
   ```

3. Regenerate the claim registry doc:
   ```bash
   python scripts/generate_claims_doc.py
   ```

4. Open a PR. The CI ratio check will fail if you added a claim without a profile.

**Priority reference datasets (see `calibra/claims/` pending_tests):**
- `lerobot/droid_100` — large-scale real hardware, position control
- `lerobot/bridge` — BridgeData V2, real hardware, Franka
- Any velocity-command real hardware dataset
- An Isaac Lab simulated dataset
- A deliberately corrupted dataset (validate metric sensitivity)

---

## Formats supported

| Format | Install | Adapter |
|--------|---------|---------|
| LeRobot v2 (Parquet) | `calibra[lerobot]` | DuckDB lazy scan, image columns never enter RAM |
| LeRobot v1 (HF datasets) | `calibra[lerobot]` | HuggingFace datasets + pandas groupby |
| HDF5 (Isaac Lab, Robomimic) | `calibra[hdf5]` | h5py, convention A+B |
| RLDS (TFDS) | `calibra[rlds]` | tensorflow-datasets |
| MCAP (ROS2) | `calibra[mcap]` | mcap + mcap-ros2-support |

---

## Contributing

The most valuable contribution right now is **profiling more datasets**.

See the ratio rule above and [`docs/claims.md`](docs/claims.md) for which datasets are pending.

```
calibra/
├── analyzers/          # Metric computation
├── claims/             # Falsifiable claim registry (JSON + SPEC.md)
├── ingestion/adapters/ # Format adapters
├── references/         # Profiled reference datasets (JSON)
├── schema/             # EpisodeBatch, DiagnosticReport, normalization
└── interpretations/    # Metric interpretation documentation (Markdown)

docs/
└── claims.md           # Auto-generated from calibra/claims/ — do not edit

scripts/
├── profile_dataset.py  # Profile any dataset → calibra/references/<name>.json
└── generate_claims_doc.py  # Regenerate docs/claims.md
```

---

## License

[MIT](LICENSE)
