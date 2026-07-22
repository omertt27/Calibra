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
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSL_1.1-blue.svg" alt="License: BSL 1.1"/></a>
</p>

**Data quality, curation, and compute efficiency for robot learning.**

Calibra identifies corrupted, redundant, and underrepresented robot demonstrations before training. It helps robotics teams select smaller, higher-value datasets, diagnose data failures, and avoid wasting GPU time on low-quality or repetitive episodes.

```bash
pip install calibra-robotics

calibra audit /data/demos.h5
calibra prune /data/demos.h5 --keep 0.3 --out coreset.json
calibra certify /data/demos.h5 --policy diffusion
```

> **Source-available and local-first** (BSL 1.1) — free for research and internal use; converts to Apache 2.0 on 2030-06-30. See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md). Not OSI open source: reselling Calibra as a hosted/managed service requires a commercial license.

---

## Why Calibra

Robot-learning performance depends not only on the policy architecture, but also on which demonstrations are used for training.

Robot learning labs collect thousands of demonstration episodes. Naively training on all of them:

- **Silently trains on bad data** — jerk spikes, dropped frames, communication lag, and stuck actuators all look like valid training signal to your policy.
- **Wastes compute on redundancy** — in a 10,000-episode dataset, 60–80% of episodes are near-duplicates. GPU cost scales with volume, not uniqueness.
- **Produces undiagnosable failures** — when a policy stalls or flails, you have no way to tell whether the cause is the architecture, the training recipe, or the data itself.

Calibra analyzes:

- temporal corruption and dropped frames
- motion smoothness and control discontinuities
- behavioral coverage and redundancy
- contact-rich and task-relevant trajectories
- latent novelty for world-model training

It then recommends which episodes to keep, review, repair, or remove.

---

## Headline result

Across ALOHA Mobile, DROID-100, and real PushT datasets, coverage-based curation consistently outperformed random selection at the same episode budget across three policy families — BC-MLP, ACT, and Diffusion Policy. **Method rankings were stable across architectures (Spearman ρ ≥ 0.86).**

**Mean improvement over random selection (5 seeds, 30% retention, 3 datasets):**

| Method | BC-MLP | ACT | Diffusion Policy |
|---|---:|---:|---:|
| Diversity-only | **+29.5%** | **+26.5%** | +11.9% |
| Calibra full | +24.5% | +23.7% | **+13.8%** |
| K-Center | +24.0% | +23.1% | +10.1% |
| Facility Location | +21.5% | +18.4% | +8.7% |
| Random | 0.0% | 0.0% | 0.0% |

Results use five shared seeds, identical selected episodes across policy architectures, and paired comparisons against random selection. Metric: offline first-action prediction error (state-based policies). [Full ablation, robustness checks, and retention curves →](#empirical-validation)

The central finding: **behavioral coverage is a robust default for robot-data selection**. Quality filtering improves results only when measurable corruption is present — on clean teleop data it is a drag across all three architectures.

On contact-rich data (real PushT), a Calibra coreset at 10% of the full dataset **outperforms training on the full dataset by 41.7%**, using 90% fewer training episodes and proportionally fewer training steps under the episode-scaled benchmark protocol.

---

## What Calibra provides

### Audit

Detect timestamp jitter, frame drops, motion discontinuities, weak coverage, short episodes, and task-structure anomalies. Flags include 95% bootstrap confidence intervals and per-episode outlier detection.

```bash
calibra audit /data/demos.h5
calibra audit /data/demos.h5 --html-out report.html
calibra audit lerobot/pusht --policy diffusion
```

### Prune

Select a smaller training set using quality-aware behavioral coverage. Two-stage pipeline: remove corrupted episodes first, then maximize behavioral diversity from the remainder.

```bash
calibra prune /data/demos.h5 --keep 0.3 --out coreset.json
calibra prune /data/demos.h5 --keep 0.3 --strategy world-model
```

### Predict

Estimate training readiness before launching an expensive experiment. Record outcomes after training to improve future predictions from your lab's actual history.

```bash
calibra predict /data/demos.h5 --policy diffusion
calibra predict /data/demos.h5 --record-outcome 0.82
```

### Monitor

Give teleoperators immediate feedback when a newly recorded episode contains detectable problems — within seconds of saving, not hours later during training.

```bash
calibra watch /data/session/ --remediate
python collect_demos.py | calibra watch --stream --remediate
```

### World-model curation

Retain clean trajectories with high latent prediction error instead of repeatedly training on already-covered dynamics.

```bash
calibra prune /data/demos --keep 0.3 --strategy world-model
```

### Evidence, not opaque scoring

Every aggregate score decomposes into its underlying metrics, per-episode findings, confidence intervals, and remediation recommendations. Calibra separates directly measured dataset signals, evidence-backed interpretations, and predictive estimates with explicit confidence levels. Every interpretation in `calibra compare` output is backed by a falsifiable claim in `calibra/claims/` with an evidence count, confidence rating, and a stated falsification condition.

---

## License

[Business Source License 1.1](LICENSE) — free for research and internal use, converts to Apache 2.0 on 2030-06-30. Offering Calibra as a hosted or managed commercial service requires a separate license. See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md). Contact: omertahtoko@gmail.com

---

## Commands

| Command | Description |
|---|---|
| `calibra` (default) | Full diagnostic audit report with dataset health score dashboard; `--cache-dir` for incremental analysis |
| `calibra compare` | Evidence-backed cross-dataset comparison |
| `calibra certify` | Structured pass/fail certification; `--report` writes a CalibraReport JSON |
| `calibra prune` | Two-stage coreset selection; `--report` writes episode verdicts; `--cache-dir` for incremental analysis |
| `calibra corrupt` | Inject synthetic corruptions to validate metric sensitivity |
| `calibra retarget` | Convert absolute EEF actions to relative delta actions |
| `calibra predict` | Predict training outcome before spending GPU time |
| `calibra card` | Generate a HuggingFace dataset quality card |
| `calibra watch` | Real-time teleoperation quality monitor |
| `calibra score` | Composite 0–100 quality score across four dimensions |
| `calibra sim2real` | Quantify sim-to-real distribution gap |
| `calibra transfer` | Cross-embodiment compatibility scoring |
| `calibra cure` | Automatic data remediation (smoothing, resampling, trimming) |
| `calibra serve` | Local REST API server and web dashboard |
| `calibra audit-all` | Bulk-audit an entire HF org or explicit dataset list; writes CalibraReport JSONs |
| `calibra site` | Generate a static leaderboard website from `audit-all` results |

### 1. `audit` — full diagnostic report

```bash
calibra /data/robot_demos.h5
calibra lerobot/pusht --policy diffusion
calibra /data/demo.h5 --policy act --json
calibra /data/robot_demos.h5 --html-out report.html   # save visual HTML dashboard
calibra /data/demos.h5 --cache-dir .calibra/cache     # incremental analysis
```

Runs four analyzers over every episode and flags anomalies with bootstrap confidence intervals and per-episode outlier detection. The `--html-out` dashboard includes a **Dataset Health Score panel** — a composite 0–100 score derived from diagnostic flags, broken down into four sub-scores: Quality, Synchrony, Coverage, and Integrity (color-coded green/yellow/red).

`--cache-dir DIR` enables incremental analysis: the pipeline result is stored in a file-based cache keyed by a SHA-256 fingerprint of the episode manifest. On unchanged data, the next run returns instantly from cache. Useful when collecting daily demos and re-auditing the same dataset repeatedly.

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

Every interpretation is backed by a falsifiable claim in `calibra/claims/` with an evidence count, confidence rating, and a stated falsification condition. Calibra separates directly measured signals, evidence-backed interpretations, and predictive estimates with explicit confidence levels.

### 3. `certify` — structured pass/fail certification

```bash
calibra certify /data/my_demos
calibra certify /data/my_demos --reference aloha --policy diffusion --strict
calibra certify hf://lerobot/my_dataset --json   # for CI pipelines
calibra certify /data/my_demos --report results/my_demos/latest.json  # write CalibraReport JSON
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

`--report PATH` writes a schema-versioned **CalibraReport** JSON to the given path (e.g. `results/lerobot/pusht/latest.json`). This is the same structured format produced by `audit-all` and consumed by `calibra site` — use it to integrate individual certification runs into the leaderboard pipeline.

### 4. `prune` — coreset selection

```bash
calibra prune /data/100k_episodes --keep 0.3 --out coreset.json
calibra prune /data/my_ds --keep 0.5 --quality-only
calibra prune /data/my_ds --keep 0.25 --max-spike-rate 0.03 --max-vel-disc-rate 0.08

# Write a schema-versioned CalibraReport with per-episode verdicts (recommended)
calibra prune /data/demos.h5 --keep 0.3 --report results/my_ds/latest.json

# GR00T fine-tuning: strict quality thresholds + entropy-weighted diversity
calibra prune demos.hdf5 --keep 0.3 --policy gr00t --report results/franka/latest.json

# Incremental analysis: skip re-running the pipeline on unchanged episodes
calibra prune /data/demos.h5 --keep 0.3 --cache-dir .calibra/cache --report results/latest.json
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

**`--report PATH`** writes a schema-versioned **CalibraReport JSON** to the given path. This is the stable machine-readable contract for all downstream systems (training pipelines, CI, HuggingFace metadata). The report includes `episode_verdicts` — a structured list of approved and rejected episode IDs, per-episode reason codes (e.g. `jerk_spike`, `diversity_pruned`), quality scores, and per-episode SHA-256 content hashes for future change detection. See the [Integrations](#integrations) section for how to consume this report.

**`--cache-dir DIR`** caches the diagnostic pipeline result keyed by a SHA-256 fingerprint of the episode manifest. On repeated runs with unchanged data, the pipeline is skipped and coreset selection proceeds immediately from cache — typically 10–50× faster on large datasets collected incrementally.

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

### 10. `score` — composite 0–100 quality score

```bash
calibra score /data/robot_demos.h5
calibra score lerobot/my_dataset --policy diffusion
calibra score /data/my_ds --reference aloha --json
calibra score hf://lerobot/pusht_image --badge   # print markdown badge for dataset cards
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dataset  : my_demos
  Episodes : 120  ·  Steps: 180000

────────────────────────────────────────────────────────────
  🟢  78.0 / 100  —  Good
────────────────────────────────────────────────────────────

  Temporal Stability       22.00/25  [█████████████████░░░]  88%
     jitter_cv: 0.038
     dropout_rate: 0.003

  Control Smoothness       26.00/35  [██████████████░░░░░░]  74%
     ldlj: -10.6
     spike_rate: 0.021
     vel_disc_rate: 0.027

  Coverage / Diversity     19.00/25  [███████████████░░░░░]  76%
     action_entropy_bits_per_dim: 2.9

  Task Structure           11.00/15  [██████████████░░░░░░]  73%
     trajectory_diversity: 0.31
     short_episode_fraction: 0.04

  0 critical flags  ·  3 warnings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Aggregates all four diagnostic dimensions into a single 0–100 number: Temporal Stability (25 pts), Control Smoothness (35 pts), Coverage/Diversity (25 pts), and Task Structure (15 pts). Score categories: 90–100 Excellent, 75–89 Good, 60–74 Fair, 40–59 Poor, 0–39 Critical. Use `--badge` to generate a shields.io markdown badge for HuggingFace dataset cards. Every aggregate score decomposes into its underlying per-metric findings and confidence intervals. Exit codes: `0` = Good or better (≥75), `1` = Fair or Poor (40–74), `2` = Critical (<40).

### 11. `sim2real` — sim-to-real distribution gap

```bash
calibra sim2real /data/sim_demos.h5 /data/real_demos.h5
calibra sim2real lerobot/sim_dataset /data/real.h5 --policy pi0
calibra sim2real /data/sim.h5 /data/real.h5 --json
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA SIM-TO-REAL GAP ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Sim dataset  : isaac_lab_pick  (500 eps)
  Real dataset : real_pick       (120 eps)

────────────────────────────────────────────────────────────
  🟡  Overall Transfer Risk: MEDIUM
  📊  Pre-training Alignment Index (PAI): 71.3%
────────────────────────────────────────────────────────────

  🟢 Ldlj Gap                             [LOW]
     Sim: -6.2   Real: -8.1   Δ = 1.9
     → Real motions are smoother than sim.

  🟡 Action Kl Divergence                 [MEDIUM]
     Value: 0.73
     → KL(sim||real) = 0.730. Significant action distribution mismatch.

  🟢 Sim Coverage Of Real                 [LOW]
     Value: 0.81
     → Sim covers 81% of the real action space. Good coverage.

  🟢 Control Frequency Gap                [LOW]
     Sim: 50.0   Real: 50.0   Δ = 0.0
     → Sim runs at 50 Hz, real at 50 Hz. Frequency match is good.

────────────────────────────────────────────────────────────
  RECOMMENDATIONS
────────────────────────────────────────────────────────────
  • Consider collecting a small real dataset (50–200 episodes) for
    fine-tuning or domain randomisation in sim.
  • Use `calibra prune` to select the sim episodes closest to the
    real distribution before training.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Measures the distribution gap between a simulation and real-robot dataset across action-space KL divergence, trajectory smoothness delta, coverage overlap, transition dynamics, and control frequency mismatch. Reports an overall transfer risk level (LOW / MEDIUM / HIGH / CRITICAL) and a Pre-training Alignment Index (PAI, 0–100%) summarising how well the sim distribution covers real-world conditions. Exit codes: `0` = LOW or MEDIUM, `1` = HIGH, `2` = CRITICAL.

### 12. `transfer` — cross-embodiment compatibility

```bash
calibra transfer /data/source_robot.h5 /data/target_robot.h5
calibra transfer lerobot/aloha_mobile_cabinet lerobot/svla_so100_pickplace
calibra transfer /data/source.h5 /data/target.h5 --json
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA CROSS-EMBODIMENT TRANSFER SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Source : aloha_mobile_cabinet  (85 eps)
  Target : franka_pick           (60 eps)

────────────────────────────────────────────────────────────
  🟡  Transfer Compatibility: ADAPT
────────────────────────────────────────────────────────────

  🟡 Action Dimensionality               [ADAPT]
     Source has 14D actions, target has 7D. Subset retargeting
     (drop extra dims) may work — use `calibra retarget` to convert.

  ✅ Control Frequency                   [DIRECT]
     Control frequencies are similar (50 Hz vs 50 Hz).

  ✅ Trajectory Smoothness               [DIRECT]
     Similar smoothness profiles (ΔLDLJ = 1.80).

  ✅ Episode Length                      [DIRECT]
     Similar episode lengths (410 vs 390 steps).

  🟡 Action Range Overlap                [ADAPT]
     Source covers 63% of target action range. Some target actions
     have no source demonstrations.

────────────────────────────────────────────────────────────
  RECOMMENDATIONS
────────────────────────────────────────────────────────────
  • Normalise action spaces before mixing source and target data.
  • Use `calibra retarget` if action dims differ.
  • Consider weighting source data lower (e.g. 0.3×) than target data.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Scores the compatibility of reusing source-robot demonstrations to train a policy for a different target robot across five dimensions: action dimensionality, control frequency, trajectory smoothness, episode length, and action range overlap. Levels: DIRECT (mix freely), ADAPT (normalise or retarget first), DIFFICULT (targeted domain adaptation required), INCOMPATIBLE (structural mismatch). Exit codes: `0` = DIRECT or ADAPT, `1` = DIFFICULT, `2` = INCOMPATIBLE.

### 13. `cure` — automatic data remediation

```bash
calibra cure /data/robot_demos.h5 --out cured/
calibra cure /data/demos.h5 --remedy smooth,trim --out cured/
calibra cure lerobot/pusht --hz 10 --out cured/ --format lerobot
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  calibra cure — my_demos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Episodes cured    : 120
  Output directory  : /data/cured/
  Manifest written  : /data/cured/cure_manifest.json
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Automatically applies kinematic and temporal fixes to every episode and writes cleaned per-episode `.npz` files. The default remedy pipeline is `smooth,interpolate,trim`: Savitzky-Golay filtering removes jerk spikes, uniform resampling resolves packet drops and timing jitter, and dead-time trimming cuts leading/trailing static segments. Use `--remedy` to apply a subset, `--hz` to pin the output control frequency, and `--trim-threshold` to tune the motion-detection sensitivity. A `cure_manifest.json` records original and cured step counts and Hz for every episode.

### 14. `audit-all` — bulk dataset auditor

```bash
calibra audit-all --org lerobot                        # audit every dataset in an HF org
calibra audit-all --org lerobot --out ./results --workers 8
calibra audit-all --dataset lerobot/pusht lerobot/aloha_sim_insertion_human
calibra audit-all --org lerobot --force                # re-audit even if cached
calibra audit-all --org lerobot --limit 5 --dry-run    # preview without running
```

```
Discovering datasets (org=lerobot) ...
Found 47 dataset(s).
[1/47] lerobot/pusht  auditing ...
[1/47] lerobot/pusht  OK  score=74.2 grade=C cert=provisional  8.3s
[2/47] lerobot/aloha_sim_insertion_human  auditing ...
...

Done.  audited=45  skipped=2  failed=0  mean_score=71.8
Manifest: results/manifest.json
```

Bulk-audits a HuggingFace org or an explicit list of dataset repo IDs in parallel. For each dataset it runs `Pipeline().analyze_path()`, assembles a schema-validated **CalibraReport** JSON, and writes it to a revision-stamped path:

```
results/<org>/<slug>/<revision-sha[:8]>/<timestamp>.json
results/<org>/<slug>/latest.json    ← always up-to-date symlink
```

A `manifest.json` is written to the output root with an aggregate summary. Skips datasets whose current revision is already cached; use `--force` to re-audit. Supports `--workers N` for parallel execution (default 4) and `--format` to override the adapter for all datasets. Requires `pip install huggingface-hub`.

### 15. `site` — static leaderboard website

```bash
calibra site --results ./results --out ./site
calibra site --results ./results --out ./site --title "My Robot Lab Leaderboard"
```

```
Building site from 47 report(s) → site/
  leaderboard  → site/index.html
  dataset page → site/lerobot/pusht/index.html
  dataset page → site/lerobot/aloha_sim_insertion_human/index.html
  ...
Done. 47 dataset(s) → site/index.html
```

Reads the `results/` directory tree produced by `audit-all` and generates a self-contained static website with no server or Python runtime required:

| Output | Description |
|---|---|
| `site/index.html` | Sortable, filterable dataset leaderboard with search and grade/cert filters |
| `site/<org>/<slug>/index.html` | Per-dataset detail page: quality dimensions, findings, policy recommendations |
| `site/<org>/<slug>/badge.svg` | Embeddable shields.io-style quality badge |
| `site/<org>/<slug>/history.json` | Score history across dataset revisions (for external consumers) |

The leaderboard is fully client-side (no build step, no dependencies) and can be hosted on GitHub Pages, Netlify, or any static file server.

### 16. `serve` — local REST API server

```bash
calibra serve                    # start on localhost:7842
calibra serve --port 8000
calibra serve --host 0.0.0.0
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA SERVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Web dashboard : http://localhost:7842
  REST API      : http://localhost:7842/api/v1
  Press Ctrl+C to stop.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Starts a local HTTP server that exposes all Calibra diagnostics as a REST API and serves the visual web dashboard at `http://localhost:7842`. Useful for programmatic access from scripts, CI pipeline integrations, and browsing dataset metrics in a browser without the terminal. Use `--host 0.0.0.0` to expose the server on all network interfaces.

---

## Integrations

Calibra bridges the gap between dataset curation and training pipelines. The public `CalibraReport` JSON contract is the stable handoff between them — produced by `calibra prune --report` and consumed by training code without importing any Calibra internals.

### LeRobot (HuggingFace)

```python
from calibra.integrations.lerobot import (
    recommended_episode_ids,
    rejected_episode_ids,
    rejection_reason_codes,
    load_dataset,
    filter_by_report,
)

# 1. Get approved episode IDs from a CalibraReport
ids = recommended_episode_ids("results/pusht/latest.json")
# → ['0', '4', '7', '12', ...]

# 2. Load a HuggingFace LeRobot dataset filtered to approved episodes
ds = load_dataset("lerobot/pusht", report_path="results/pusht/latest.json")
# ds is a datasets.Dataset with only the Calibra-approved episodes

# 3. Filter an already-loaded dataset
from datasets import load_dataset as hf_load
full = hf_load("lerobot/pusht", split="train")
ds = filter_by_report(full, "results/pusht/latest.json")

# 4. Inspect rejection reasons
codes = rejection_reason_codes("results/pusht/latest.json")
# → {"42": ["jerk_spike", "timestamp_dropout"], "7": ["diversity_pruned"]}
```

**End-to-end workflow:**

```bash
# 1. Record demos with LeRobot
lerobot-record --robot-type so100 --repo-id $HF_USER/my_dataset

# 2. Curate and write the report
calibra prune /path/to/my_dataset --keep 0.3 --report results/my_dataset/latest.json

# 3. Train on the coreset
python - <<'EOF'
from calibra.integrations.lerobot import load_dataset
ds = load_dataset("/path/to/my_dataset", report_path="results/my_dataset/latest.json")
ds.save_to_disk("./my_dataset_coreset")
EOF

lerobot-train policy=act dataset_repo_id=./my_dataset_coreset
```

**Benchmark:** `experiments/lerobot_coreset_benchmark.py` runs the full sweep on any LeRobot v2 HuggingFace dataset, comparing Calibra coreset vs. random vs. full at keep fractions 10–100%, evaluated by held-out trajectory MSE.

```bash
pip install calibra-robotics datasets torch matplotlib
python experiments/lerobot_coreset_benchmark.py --dataset lerobot/pusht --keep 0.1 0.3 0.5
```

### Isaac Lab → GR00T (NVIDIA)

```python
from calibra.integrations.isaac_lab import (
    recommended_demo_indices,
    rejected_demo_indices,
    rejection_reason_codes,
    export_gr00t_manifest,
    filter_hdf5,
)

# 1. Get approved demo indices (0-based integers matching HDF5 group order)
indices = recommended_demo_indices("results/franka/latest.json")
# → [0, 3, 7, 11, ...]

# 2. Export a GR00T training manifest
manifest_path = export_gr00t_manifest(
    report_path="results/franka/latest.json",
    demos_path="demos/franka_pick.hdf5",
    out_path="gr00t_manifest.json",
)

# 3. Write a filtered HDF5 with only Calibra-approved demos
filter_hdf5(
    src="demos/franka_pick.hdf5",
    report_path="results/franka/latest.json",
    dst="demos/franka_pick_coreset.hdf5",
)
```

**End-to-end workflow:**

```bash
# 1. Record Isaac Lab demos (produces demos.hdf5)
python scripts/robosuite_collect_data.py

# 2. Curate with GR00T-tuned thresholds and write the report
calibra prune demos.hdf5 --keep 0.3 --policy gr00t --report results/franka/latest.json

# 3. Export GR00T manifest and filtered HDF5
python - <<'EOF'
from calibra.integrations.isaac_lab import export_gr00t_manifest, filter_hdf5
export_gr00t_manifest("results/franka/latest.json", demos_path="demos.hdf5")
filter_hdf5("demos.hdf5", "results/franka/latest.json", "demos_coreset.hdf5")
EOF

# 4. Fine-tune GR00T on the coreset
python -m gr00t.train --manifest gr00t_manifest.json --demo-file demos_coreset.hdf5
```

**GR00T manifest format:**
```json
{
  "schema_version": "1.0.0",
  "calibra_report": "/abs/path/results/franka/latest.json",
  "dataset_path": "/abs/path/demos/franka_pick.hdf5",
  "n_demos_total": 200,
  "n_demos_selected": 60,
  "keep_fraction": 0.30,
  "method": "calibra-diversity",
  "demo_indices": [0, 3, 7, 11, "..."],
  "demo_ids": ["demo_0", "demo_3", "demo_7", "demo_11", "..."],
  "reason_codes": {"42": ["jerk_spike"], "7": ["diversity_pruned"]},
  "created_at": "2026-07-23T12:00:00+00:00"
}
```

**Benchmark:** `experiments/isaac_lab_gr00t_benchmark.py` runs the full sweep on synthetic 7-DOF arm data, comparing Calibra (with GR00T thresholds) vs. random selection at multiple keep fractions.

```bash
pip install calibra-robotics torch numpy matplotlib
python experiments/isaac_lab_gr00t_benchmark.py --n-demos 300 --keep 0.2 0.3 0.5
```

### Incremental analysis (daily pipelines)

When collecting demos daily and re-auditing the same dataset, use `--cache-dir` to skip the pipeline on unchanged episodes:

```bash
# Day 1: full audit takes ~30s
calibra prune /data/demos --keep 0.3 --cache-dir .calibra/cache --report results/latest.json

# Day 2: 10 new episodes added — cache hit for unchanged episodes, re-runs only on new ones
calibra prune /data/demos --keep 0.3 --cache-dir .calibra/cache --report results/latest.json
# → [cache hit]  pipeline skipped (fingerprint unchanged for 90% of data)
```

The cache is keyed by a 24-char SHA-256 fingerprint of sorted `(episode_id, content_hash)` pairs plus the policy family. Any episode change — re-recording, re-processing, or adding episodes — invalidates the fingerprint for the full batch, triggering a fresh pipeline run.

The `episode_hashes` field in the CalibraReport JSON records each episode's SHA-256[:16] content hash so downstream systems can detect future changes without re-running Calibra.

---

## Empirical Validation

Calibra is a **dataset observability framework** for diagnosing, predicting, and curating robotics training data. Its diversity-based selection component is *competitive with established coreset methods* (K-Center, Facility Location) rather than a claimed improvement over them, and its offline metrics provide *measurable, moderate* pre-training predictive signal. The results below are reported at the strength the evidence supports — seeded, paired comparisons are the headline; single-seed runs are labelled exploratory.

### Deployed GPU Benchmarks (RTX 2080)

To validate Calibra's real-world impact, we ran full policy training and evaluation loops on an RTX 2080 GPU under two critical validation setups.

#### 1. Coreset Curation Benchmark (`gym_pusht/PushT-v0`) — *exploratory single-seed case study*
Trains a Behavior Cloning (BC) policy on a curated **30% Calibra coreset** vs. the full raw dataset and a random 30% baseline.

| Curation Condition | Training Steps | Avg Coverage | Success Rate ($\text{SR} \ge 50\%$) | Optimization Steps Saved | CUDA Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Raw Dataset (100%)** | 150,000 | 21.9% | 2.0% | Base (0.0% saved) | 186.7s |
| **Calibra 30% Coreset** | **6,300** | **23.3%** | **8.0%** | **95.8% fewer steps** | **7.8s** |
| **Random 30% Baseline** | 45,000 | 23.8% | 6.0% | 69.5% fewer steps | 56.9s |

> ⚠️ **Reported as exploratory evidence, not a general result.** This is a **single-seed** run and absolute success rates are low (2–8%). A 10-seed variance sweep (`experiments/check_pusht_variance.py`) shows the clean coreset averages **14.8% ± 6.3%**, so single-run gaps at this scale are within training-seed noise. The seeded, paired-*t* ablation below (5 seeds, 3 datasets) is the result to rely on.
>
> **On the 95.8% figure:** the coreset used 6,300 optimization steps vs. 150,000 for full-data under this benchmark's *episode-scaled* step schedule — i.e. 95.8% fewer optimization steps and 95.8% lower measured train time *under this protocol*. This does **not** imply that retaining 30% of episodes universally cuts training cost by 95.8%; the saving depends on the training-budget protocol.

To reproduce:
```bash
python experiments/pusht_real_benchmark.py
```

#### 2. Failure Prediction & Correlation Benchmark (`calibra predict`)
Validates that Calibra can intercept dataset failures *before* wasting GPU compute. Tested across 15 PushT dataset variants corrupted with controlled noise (frame drops, joystick spikes, noisy episodes) at varying severity levels.

* **Predictive Correlation (L6, within-PushT corruption severity):** Spearman **$\rho = 0.6749$** ($p = 0.0057$) between offline scores and downstream success *across the 15 corruption-severity variants of a single task*. This measures sensitivity to corruption severity — it is a different question from the cross-dataset ranking correlation reported below.
* **Failure Prediction Accuracy (L4):** **73.3%** (11/15 conditions correctly classified as PASS/FAIL prior to training).
* **Root-Cause Accuracy (L4):** **88.9%** (8/9 single-fault modes correctly identified, pinpointing teleoperation spikes and packet loss).

To reproduce:
```bash
python experiments/failure_prevention_benchmark.py --save-fig --out-json results_l4l6.json
```

---

### Predictor success correlation

Offline predicted success probabilities (`calibra predict`) achieve a **Spearman rank correlation (ρ) of 0.5971** (p = 0.0146, statistically significant) with actual downstream policy success rates across **seven verified dataset conditions** (ALOHA sim × 4, PushT image, Mobile ALOHA × 2). This is a *cross-dataset ranking* correlation and is distinct from the within-PushT corruption-severity correlation (ρ = 0.6749) reported above. DROID-100 and BridgeData V2 are **excluded** from this correlation due to a control-mode mismatch (velocity-command datasets have structurally high vel_disc that the current rubric over-penalises).

> Calibra ships **reference profiles for 16 datasets** — a separate count from the seven datasets used in this correlation.

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
| Quality-filter only | +8.8% (unchanged) | −5.8% (unchanged) | **+32.0%** |
| Diversity-only | +13.0% (unchanged) | +16.9% (unchanged) | +47.4% (unchanged) |
| **Calibra full pipeline** | **+22.6%** (unchanged) | **+16.9%** (unchanged) | **+39.5%** |

The contact-aware filter detects that real PushT's velocity discontinuities are contact-driven (vel_disc/spike ratio = 24.4) and relaxes the vel_disc threshold by 3×. This transforms a −30.5% failure into a +39.5% gain while leaving ALOHA and DROID completely unchanged (their ratios are 1.9 and 1.6 — well below the 3.0 threshold).

**Quality filtering failure modes (without contact-aware fix):**

1. **Heterogeneous datasets (DROID)** — quality-only collapses coverage of rare robot morphologies. Removing the noisiest episodes inadvertently removes the only representatives of certain robot types.

2. **Contact-rich tasks (real PushT)** — velocity discontinuities during contact events (block-hits, pushes) are classified as noise. The quality filter removes the most contact-rich demonstrations, which have the highest learning signal.

The contact-aware fix addresses failure mode 2 by using the vel_disc/spike ratio as a contact detector. Failure mode 1 (morphology collapse) remains an open problem — see `calibra/strategy.py` for the regime-dependent configuration that mitigates it via relaxed thresholds.

---

### Coreset baseline comparison — vs. published selection methods

The ablation above compares Calibra against random selection. To test it against the coreset / data-selection literature, we add three published baselines — **K-Center greedy** (Gonzalez farthest-point sampling), **Herding** (mean-matching), and **Facility Location** (greedy submodular coverage) — all run on the *same* per-episode behavioral features as Calibra's diversity stage, so the only variable is the selection algorithm. Every condition is trained over **5 shared seeds** with per-seed MSEs paired across conditions, enabling a paired *t*-test. Keep fraction 30%, BC-MLP, RTX 2080.

**Per-dataset improvement vs. random (5-seed mean):**

| Method (keep 30%) | ALOHA mobile | DROID-100 | PushT real |
|---|---:|---:|---:|
| Full dataset (100%) | +41.4% | +12.0% | +12.8% |
| K-Center greedy | +21.4% | +3.9% | +46.8% |
| Herding | −17.0% | +2.5% | −19.7% |
| Facility Location | +17.3% | +3.4% | +43.9% |
| Quality-filter only | +5.5% | +7.0% | +36.2% |
| **Diversity-only** | +17.8% | **+24.3%** | +46.4% |
| **Calibra full** | +18.7% | +16.3% | +38.5% |

**Aggregate across the 3 datasets** (`experiments/aggregate_ablation.py`):

| Method | Mean rank ↓ | Mean vs-random |
|---|---:|---:|
| **Diversity-only** | **2.00** | **+29.5%** |
| K-Center greedy | 2.00 | +24.0% |
| Calibra full | 2.67 | +24.5% |
| Facility Location | 4.00 | +21.5% |
| Quality-filter only | 4.33 | +16.2% |
| Random | 6.33 | 0.0% |
| Herding | 6.67 | −11.4% |
| *Oracle (best method per dataset)* | *1.00* | *+30.8%* |

**Calibra full — Win/Tie/Loss vs. each method** (paired *t*-test, α = 0.05):

| Opponent | W-T-L | Notes |
|---|:---:|---|
| Random | 2-1-0 | never loses |
| K-Center greedy | 1-1-1 | tie on ALOHA (p=0.37), win DROID, loss PushT |
| Herding | 3-0-0 | dominates |
| Facility Location | 1-1-1 | wins DROID, loses PushT |
| Quality-filter only | 2-1-0 | quality+diversity ≥ quality alone |
| Diversity-only | 0-1-2 | **loses to its own diversity stage on DROID & PushT** |

**Takeaways:**

1. **Calibra's diversity (coverage) selection is the strongest method** — best mean improvement (**+29.5%**), tied-best mean rank (2.00), and more *consistent* than K-Center (worst rank 3 vs. K-Center's rank-4 on DROID). It never significantly loses to any published baseline.
2. **The quality filter is regime-dependent, not a universal default.** On these clean teleop datasets it drags the full pipeline below diversity-only (significant on DROID and PushT). It pays off only when real corruption is present (see the failure-prediction and `corrupt` benchmarks). The recommended default is diversity-only, with quality-filtering gated on detected corruption.
3. **Adaptive-across-methods headroom is small** — the oracle (+30.8%) beats the best fixed method (diversity-only, +29.5%) by only **1.3 points**, so no complex regime-adaptive selector is warranted on this evidence.

To reproduce:
```bash
python experiments/ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet --seeds 5 --json results/ablation_aloha.json
python experiments/ablation_benchmark.py --dataset lerobot/droid_100 --seeds 5 --json results/ablation_droid.json
python experiments/ablation_benchmark.py --dataset lerobot/columbia_cairlab_pusht_real --seeds 5 --json results/ablation_pusht_real.json
python experiments/aggregate_ablation.py results/ablation_*.json
```

---

### Cross-architecture check — BC-MLP, ACT, and Diffusion Policy

The ablation above uses a BC-MLP learner. To test whether the findings are an
artifact of that one architecture, we re-ran the **entire** benchmark — identical
datasets, splits, seeds, and byte-identical coreset selection — swapping only the
downstream learner across three families:

- **BC-MLP** — deterministic 3-layer regressor (the original ablation).
- **ACT** — Action Chunking Transformer: a state-conditioned CVAE with a
  transformer encoder–decoder that predicts a 16-step action chunk (masked L1 + KL).
  `experiments/act_ablation_benchmark.py`.
- **Diffusion Policy** — a state-conditioned DDPM that denoises a 16-step action
  chunk. `experiments/diffusion_ablation_benchmark.py`.

All at 5 seeds, keep 30%, RTX 2080. Every coreset is identical across the three;
only the policy changes.

> **Scope.** These are *state-based* policies scored on **offline first-action
> prediction error** (the action each policy executes), not image-conditioned
> policies evaluated by simulator rollout. They test whether the *selection
> ranking* transfers across learners, not task-level success rate. Diffusion is a
> stochastic sampler, so it is evaluated by single-sample first-action MSE and is
> only compared **relative to Random / by rank within the diffusion learner** —
> absolute MSE is never compared across policy families.

**Mean improvement over Random (5 seeds, keep 30%, 3 datasets; per-method mean rank in parentheses):**

| Method (keep 30%) | BC-MLP | ACT | Diffusion |
|---|---:|---:|---:|
| **Diversity-only** | +29.5% (2.0) | +26.5% (2.0) | +11.9% (2.7) |
| **Calibra full** | +24.5% (2.7) | +23.7% (3.0) | **+13.8% (1.3)** |
| K-Center greedy | +24.0% (2.0) | +23.1% (2.7) | +10.1% (4.0) |
| Facility Location | +21.5% (4.0) | +18.4% (3.7) | +8.7% (4.3) |
| Quality-filter only | +16.2% (4.3) | +17.4% (4.7) | +11.1% (3.3) |
| Random | 0.0% (6.3) | 0.0% (5.7) | 0.0% (5.7) |
| Herding | −11.4% (6.7) | −7.5% (6.3) | −5.5% (6.7) |

**Rank agreement of the method ordering (Spearman ρ):** BC↔ACT **1.00**, BC↔Diffusion **0.86**, ACT↔Diffusion **0.86** (Kendall τ = 1.00 / 0.71 / 0.71). Changing the learner from an MLP to a transformer (BC→ACT) did not change the method ordering at all; changing to a generative policy (→Diffusion) altered some details but preserved most of the ranking.

**What holds across all three, and the one thing that shifts:**

1. **The core ranking is architecture-robust** (ρ ≥ 0.86). Coverage-based selection (Diversity-only / Calibra full) **consistently outperformed Random across all three evaluated policy architectures**; K-Center and Facility Location are mid-pack; and **Herding was worst under every learner**. Calibra full was *never significantly worse than the best published coreset baseline* (K-Center) under any learner.
2. **Diversity-only (coverage) selection is the best-or-tied-best method under all three learners.** In the single-sample diffusion run above, Calibra full's quality filter appeared to lift it to the top rank — but two robustness checks (below) overturn that. Under both equal compute and multi-sample-averaged evaluation, **Diversity-only is again the top method under diffusion** and Calibra full significantly loses to it on DROID and PushT (p<0.001) — the *same* quality-filter drag seen on BC-MLP and ACT. So the honest conclusion is that **the quality filter is a drag on clean data under all three architectures**, and the apparent diffusion exception did not survive robustness checks.
3. **Practical takeaway:** diversity/coverage selection is the safe default across all three architectures; quality filtering pays off only under detected corruption (see the failure-prediction and `corrupt` benchmarks), not on clean teleop data.

**Robustness checks on the diffusion learner (5 key conditions, 5 seeds).** Because diffusion is a stochastic sampler evaluated by single-sample first-action MSE, we ran two controls — reported *alongside* the single-sample numbers, not replacing them:

| Method | Single-sample (default) | Equal compute (5000 steps) | Multi-sample (10× avg) |
|---|---:|---:|---:|
| **Diversity-only** | +11.9% (rank 2.7) | +23.4% (**1.67**) | +22.9% (**1.67**) |
| Calibra full | +13.8% (1.33) | +19.3% (2.33) | +21.2% (2.00) |
| K-Center greedy | +10.1% (4.0) | +22.0% (2.00) | +20.2% (2.33) |

- **Equal compute** (`--max-steps 5000`, identical optimizer steps for every condition): Full-dataset's default-protocol advantage was largely a compute artifact — its vs-random gap collapses from +59/+14/+36% to +2.6/−6.4/+3.6% (ALOHA/DROID/PushT) once steps are equalized, while the coreset selection ranking is unchanged. The coreset advantage is not a compute artifact.
- **Multi-sample** (`--eval-samples 10`, averaging sampler noise): the method ranking is stable and matches the equal-compute ranking — confirming the single-sample diffusion result's *ranking* was not merely sampler noise, even though the specific "Calibra full first" ordering was.

To reproduce:
```bash
# ACT
python experiments/act_ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet --seeds 5 --json results/act_ablation_aloha.json
python experiments/act_ablation_benchmark.py --dataset lerobot/droid_100 --seeds 5 --json results/act_ablation_droid.json
python experiments/act_ablation_benchmark.py --dataset lerobot/columbia_cairlab_pusht_real --seeds 5 --json results/act_ablation_pusht_real.json
python experiments/aggregate_ablation.py results/act_ablation_*.json
# Diffusion Policy
python experiments/diffusion_ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet --seeds 5 --json results/diffusion_ablation_aloha.json
python experiments/diffusion_ablation_benchmark.py --dataset lerobot/droid_100 --seeds 5 --json results/diffusion_ablation_droid.json
python experiments/diffusion_ablation_benchmark.py --dataset lerobot/columbia_cairlab_pusht_real --seeds 5 --json results/diffusion_ablation_pusht_real.json
python experiments/aggregate_ablation.py results/diffusion_ablation_*.json
# Diffusion robustness checks (per dataset; --conditions restricts to the 5 key methods)
#   equal compute:  add  --max-steps 5000
#   multi-sample:   add  --eval-samples 10
python experiments/diffusion_ablation_benchmark.py --dataset lerobot/droid_100 --seeds 5 \
    --max-steps 5000 --conditions "Full dataset,K-Center greedy,Diversity-only,Calibra full" \
    --json results/diffusion_equalstep_droid.json
# NOTE: aggregate_ablation.py takes explicit paths — shell globs like results/*.json are not
# expanded by PowerShell, so pass filenames (or run from bash).
```

---

### Rare-mode preservation — controlled multimodal benchmark

The coreset benchmark above measures average success rate. A tougher test is **worst-group success**: can a heavily compressed coreset still cover a rare behavioural mode that random selection frequently misses?

**Setup:** 2D point-mass navigation with a vertical wall and two goals — Goal A (upper-right, 140 episodes) and Goal B (lower-right, 20 episodes), a 7:1 imbalance. At low retention budgets, random selection often draws zero or one B episode, making B a silent failure. Calibra's diversity-aware selector explicitly samples from underrepresented regions of the action space. Worst-group success = min(A success, B success). 5 random seeds per fraction.

**Results (BC policy, 5-seed mean; `experiments/multigoal_obstacle_benchmark.py`):**

| Keep | Episodes | Cal Overall | Cal Worst | Rand Overall | Rand Worst | Episodes saved |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **5%** | **8** | **99.3%** | **98.7%** | 68.3% ±21% | 36.5% | 95% |
| **10%** | **16** | **100.0%** | **100.0%** | 84.5% ±17.6% | 69.1% | 90% |
| 20% | 32 | 100.0% | 100.0% | 95.5% ±7.2% | 91.2% | 80% |
| 30% | 48 | 100.0% | 100.0% | 99.9% ±0.3% | 99.7% | 70% |
| 50% | 80 | 100.0% | 100.0% | 99.6% ±0.5% | 99.2% | 50% |
| 100% | 160 | 100.0% | 100.0% | 99.9% ±0.3% | 99.7% | — |

At **5% retention** (8 episodes), Calibra achieves 98.7% worst-group vs. 36.5% for random — a **62-point gap**. Calibra selected 3 Goal-B episodes out of 8 total slots, overrepresenting the rare mode 4.5× relative to its population share. At **10% retention** (16 episodes), Calibra matches full-dataset performance on both goals while random still misses B in roughly 1 in 3 seeds.

To reproduce:
```bash
pip install calibra-robotics torch numpy
python experiments/multigoal_obstacle_benchmark.py
```

**Diagnostic validity sweep (same environment):** each corruption family is swept in isolation to verify that each metric responds to the defect it targets:

| Corruption | Score behaviour | Status |
|---|---|:---:|
| Frame dropout (0 → 80%) | 57.9 → 24.8 (monotone decrease) | Correct |
| Jerk spikes (0 → 20%) | 57.9 → 42.0 (saturates at 5% spike rate) | Correct (saturation documented) |
| Duplicates (0 → 90%) | 57.9 → 44.6 (monotone decrease) | Correct |
| Episode truncation (100% → 10%) | 57.9 → 68.6 (inversion at 10%) | Known gap — short episodes only expose the smooth approach phase; task-progress signals would fix this |
| Observation lag (0 → 16 frames) | 57.9 → 57.9 (no response) | Known gap — current temporal metrics do not detect shifted observation windows |

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
| PushT real | **24.4** | HIGH NOISE (contact) | **−30.5%** | **+39.5%** |

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

**PushT real (contact-aware) — Calibra full vs. random:**

| Keep | k | Calibra vs. Random | Calibra vs. Full |
|---|---|---|---|
| 10% | 11 | +56.6% | +41.7% |
| 20% | 22 | +49.1% | +38.4% |
| 30% | 33 | +42.5% | +31.5% |
| 50% | 54 | +32.2% | +17.6% |
| 70% | 76 | +15.0% | +10.4% |

On contact-rich tasks like PushT real, a Calibra coreset at 10% of the dataset size outperforms training on the full unpruned dataset by 41.7%, using 90% fewer training episodes and proportionally fewer training steps under the episode-scaled benchmark protocol.

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

You can profile additional datasets locally using `scripts/profile_dataset.py`.

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
**Works out of the box** — no extra install required.

Stage 2 scores each quality-passing episode by latent prediction error and keeps the highest-surprise fraction. Two backends compute that score, chosen automatically:

- **Lightweight baseline (default, core install):** a closed-form encoder (PCA / random projection) and a closed-form linear next-latent predictor — no gradient descent, no GPU, fit in milliseconds. See `calibra/world_model/surprise.py`.
- **Trained JEPA (advanced, `pip install torch`):** a small MLP encoder/predictor trained with a VICReg objective (`calibra/models/robot_jepa.py`) for higher-fidelity surprise scores on larger datasets.

`calibra prune --strategy world-model` and `calibra watch --world-model` use the trained JEPA when torch is installed and fall back to the lightweight baseline otherwise — same CLI flags, same output shape, no code changes required either way. The result is a dataset that pushes the world model toward unexplored dynamics rather than reinforcing what it already knows.

```
WORLD-MODEL CURATION SUMMARY

Original episodes: 1000
Quality failures: 87
High-surprise kept: 300
Low-surprise pruned: 613

Top novel episodes:
  ep_104  surprise=0.82  reason="unusual contact dynamics"
  ep_511  surprise=0.77  reason="rare state-space excursion"
  ep_208  surprise=0.74  reason="long-horizon recovery"
```

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

- **Not only a dataset score** — every aggregate score decomposes into specific metrics, per-episode findings, confidence intervals, and actionable evidence.
- **Not an AI assistant** — it runs deterministic mathematical estimators, not a language model.
- **Not a cloud service** — it runs entirely locally against your files.
- **Not a replacement for domain expertise** — it tells you *what* to look at; you decide *what to do*.

---

## Formats supported

| Format | Extra install | Notes |
|---|---|---|
| LeRobot v2/v3 (Parquet shards) | `calibra[lerobot]` | DuckDB lazy scan — image columns never enter RAM; v3 uses the same fast path as v2 |
| LeRobot v1 (HF datasets) | `calibra[lerobot]` | HuggingFace `datasets` + pandas groupby |
| HuggingFace Hub | `calibra[lerobot]` | `lerobot/pusht`, `hf://lerobot/pusht` |
| HDF5 (Isaac Lab, Robomimic) | `calibra[hdf5]` | Convention A + B |
| RLDS / TF Datasets | `calibra[rlds]` | tensorflow-datasets |
| MCAP / ROS2 bags | `calibra[mcap]` | mcap + mcap-ros2-support |

---

## Contributing

Calibra is not open to external pull requests (PRs) or contributions at this time.

---

## Development

### Repository layout

```
calibra/
├── core/               # Public API: LazyDatasetReader, SchemaNormalizer, mappings.yaml
├── metrics/            # Standalone pure-numpy functions (no pipeline needed)
├── analyzers/          # Pipeline analyzers: temporal, smoothness, coverage, task_structure
├── ingestion/          # Format adapters (lerobot v1/v2/v3, hdf5, rlds, mcap) + registry
├── comparison/         # DatasetComparator, EpisodeCurator
├── world_model/        # Lightweight (non-torch) world-model surprise scoring + curation
├── models/             # RobotJEPA — trained world model (optional, requires torch)
├── schema/
│   ├── report.py           # EpisodeBatch, DiagnosticReport, normalization layer (internal)
│   ├── public_report.py    # CalibraReport — stable public JSON contract (versioned)
│   └── scoring.py          # Scoring rubric v1.0: dimension weights, grade thresholds
├── audit_all.py        # calibra audit-all — bulk HF org / dataset auditor
├── report_json.py      # Assemble CalibraReport from DiagnosticReport
├── report_html.py      # HTML dashboard (includes Dataset Health Score panel)
├── site.py             # calibra site — static leaderboard + per-dataset page generator
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

### Development setup

```bash
git clone https://github.com/omerTT/Calibra
pip install -e '.[all,dev]'
pytest              # 596 tests
ruff check .        # zero errors expected
```

---

## License

[Business Source License 1.1](LICENSE) — free for internal use, open-source under Apache 2.0 on 2030-06-30. Commercial hosting requires a license: omertahtoko@gmail.com
